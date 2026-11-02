"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Structural redesign: uses support-function Minkowski sums for pairwise separation,
LP for outer containment, and L-BFGS-B for global optimization of positions/angles.
"""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
DEADLINE_MARGIN = 15.0  # seconds before the 150s limit
SQRT3 = math.sqrt(3)


def _hex_vertices_local(angle_deg):
    """Return 6 local vertices of a unit regular hexagon rotated by angle_deg."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((math.cos(a), math.sin(a)))
    return verts


def _support_func(angle_deg, nx, ny):
    """Support function of a unit hexagon rotated by angle_deg in direction (nx,ny)."""
    verts = _hex_vertices_local(angle_deg)
    max_val = 0.0
    for vx, vy in verts:
        val = vx * nx + vy * ny
        if val > max_val:
            max_val = val
    return max_val


def _separation_required(a1, a2, nx, ny):
    """Minimum separation along direction (nx,ny) for two unit hexagons with angles a1,a2."""
    # For disjointness: (c2-c1).(nx,ny) >= support(a1,nx,ny) + support(a2,-nx,-ny)
    s1 = _support_func(a1, nx, ny)
    s2 = _support_func(a2, -nx, -ny)
    return s1 + s2


def _pairwise_max_overlap(pos, angles, i, j, n_dirs=24):
    """Max overlap violation between hexagons i and j using 24 sample directions."""
    max_viol = 0.0
    dx = pos[2*j] - pos[2*i]
    dy = pos[2*j+1] - pos[2*i+1]
    for k in range(n_dirs):
        theta = 2.0 * math.pi * k / n_dirs
        nx = math.cos(theta)
        ny = math.sin(theta)
        req = _separation_required(angles[i], angles[j], nx, ny)
        proj = dx * nx + dy * ny
        viol = req - proj
        if viol > max_viol:
            max_viol = viol
    return max_viol


def _all_separations_ok(pos, angles, n_dirs=24, tol=-1e-9):
    """Check all pairs have non-negative separation in all directions."""
    n = len(angles)
    for i in range(n):
        for j in range(i+1, n):
            if _pairwise_max_overlap(pos, angles, i, j, n_dirs) > tol:
                return False
    return True


def _min_outer_lp(pos, angles, oangle_deg):
    """LP to find minimum outer hexagon side given inner positions/angles and outer angle."""
    c = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    
    a_rad = math.radians(oangle_deg)
    
    for i in range(N):
        verts = _hex_vertices_local(angles[i])
        for vx, vy in verts:
            wx = pos[2*i] + vx
            wy = pos[2*i+1] + vy
            for k in range(6):
                theta = math.radians(oangle_deg + 30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-wx * nx - wy * ny)
    
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if res.success:
        t = res.x[2]
        side = t * 2.0 / SQRT3
        center = (res.x[0], res.x[1])
        return side, center
    return float('inf'), (0.0, 0.0)


def _objective(state, oangle_deg):
    """Objective: outer side + penalty for overlaps."""
    pos = state[:2*N]
    angles = state[2*N:]
    
    side, center = _min_outer_lp(pos, angles, oangle_deg)
    if not np.isfinite(side):
        return 1e6
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            viol = _pairwise_max_overlap(pos, angles, i, j, 12)
            if viol > 0:
                penalty += viol * viol
    
    return side + 500.0 * penalty


def _build_lattice_positions():
    """Build initial hexagonal lattice positions for 11 hexagons."""
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    # Pick 4 from ring2 that form a good cluster (topologically balanced)
    selected = [ring2[0], ring2[3], ring2[6], ring2[9]]  # (2,0),(1,-2),(-2,-2),(-2,1)
    
    full_axial = pos_axial + selected
    
    pos_xy = []
    for q, r in full_axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        pos_xy.append((x, y))
    return pos_xy


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Build initial configuration
    pos_xy = _build_lattice_positions()
    
    best_side = float('inf')
    best_pos = np.zeros(2 * N)
    best_angles = np.zeros(N)
    best_center = (0.0, 0.0)
    best_oangle = 0.0
    
    # Try multiple outer orientations
    for oangle in [0.0, 15.0, 30.0, -15.0, -30.0, 5.0, -5.0, 10.0, -10.0]:
        if time.time() > deadline:
            break
        
        # Initialize state from lattice
        state = np.zeros(3 * N)
        for i in range(N):
            state[2*i] = pos_xy[i][0]
            state[2*i+1] = pos_xy[i][1]
            state[2*N+i] = 0.0
        
        # Quick LP to get baseline
        side, center = _min_outer_lp(state[:2*N], state[2*N:], oangle)
        if np.isfinite(side) and side < best_side:
            best_side = side
            best_pos = state[:2*N].copy()
            best_angles = state[2*N:].copy()
            best_center = center
            best_oangle = oangle
    
    if not np.isfinite(best_side):
        # Fallback: use centroid
        best_pos = np.array([pos_xy[i][0] for i in range(N)] + [pos_xy[i][1] for i in range(N)])
        best_angles = np.zeros(N)
        best_side, best_center = _min_outer_lp(best_pos, best_angles, 0.0)
        best_oangle = 0.0
    
    # Nonlinear optimization with L-BFGS-B
    if time.time() < deadline - 30:
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Perturb initial state for diverse restarts
            if restart == 0:
                init_state = np.concatenate([best_pos, best_angles])
            else:
                init_state = np.concatenate([best_pos, best_angles])
                rng = np.random.default_rng(42 + restart)
                noise_pos = rng.normal(0, 0.2, size=2*N)
                noise_ang = rng.normal(0, 5.0, size=N)
                init_state[:2*N] += noise_pos
                init_state[2*N:] += noise_ang
            
            try:
                result = minimize(
                    _objective, init_state, args=(best_oangle,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                
                if result.fun < best_side + 100:
                    final_pos = result.x[:2*N]
                    final_angles = result.x[2*N:]
                    
                    # Verify separations
                    if _all_separations_ok(final_pos, final_angles, 24):
                        side, center = _min_outer_lp(final_pos, final_angles, best_oangle)
                        if np.isfinite(side) and side < best_side:
                            best_side = side
                            best_pos = final_pos.copy()
                            best_angles = final_angles.copy()
                            best_center = center
            except Exception:
                pass
    
    # Try to optimize outer angle with fixed inner positions
    if time.time() < deadline - 10:
        for oangle in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                       -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            if time.time() > deadline:
                break
            side, center = _min_outer_lp(best_pos, best_angles, oangle)
            if np.isfinite(side) and side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle
    
    # Final verification: ensure all hexagons are inside outer
    outer_side = best_side
    outer_center = best_center
    outer_angle = best_oangle
    
    # Check containment
    for i in range(N):
        verts = _hex_vertices_local(best_angles[i])
        for vx, vy in verts:
            wx = best_pos[2*i] + vx
            wy = best_pos[2*i+1] + vy
            # Check if (wx,wy) is inside outer hexagon
            inside = True
            for k in range(6):
                theta = math.radians(outer_angle + 30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = (wx - outer_center[0]) * nx + (wy - outer_center[1]) * ny
                apothem = outer_side * SQRT3 / 2
                if proj > apothem + 1e-9:
                    inside = False
                    break
            if not inside:
                # Expand outer hexagon slightly
                outer_side *= 1.01
                break
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([best_pos[2*i], best_pos[2*i+1], math.degrees(best_angles[i])])
    
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
