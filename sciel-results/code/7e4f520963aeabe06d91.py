"""Hexagon packing via projected gradient descent with simulated annealing."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 120.0

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(a + i * np.pi / 3),
                      cy + side * np.sin(a + i * np.pi / 3)] for i in range(6)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx, dy = px - cx, py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx, ly = ca * dx - sa * dy, sa * dx + ca * dy
    s = side
    sqrt3 = np.sqrt(3)
    # Six half-plane inequalities for regular hexagon centered at origin
    # Vertices at angles 0,60,120,180,240,300
    # Edge normals (inward) and offsets:
    # n0=(1,0) offset s; n1=(-1,0) offset s; n2=(sqrt3/2,1/2) offset s; etc.
    # Actually for vertices at (s,0),(s/2,s*sqrt3/2),...:
    # Supporting lines:
    # x <= s, -x <= s, sqrt3*x+y <= 2*s*sqrt3/2... let me use the standard form
    # For a regular hexagon with circumradius = side (vertices at distance s from center):
    # The 6 supporting lines (outward normals, distance = s*cos(30)=s*sqrt3/2... no)
    # Let me just check: vertex (s,0). Edge from (s,0) to (s/2, s*sqrt3/2).
    # Direction: (-s/2, s*sqrt3/2). Outward normal: (s*sqrt3/2, s/2) -> normalize: (sqrt3,1)/2
    # Line: sqrt3*x + y = sqrt3*s + 0 = sqrt3*s? Check at (s,0): sqrt3*s. Yes.
    # At (s/2, s*sqrt3/2): sqrt3*s/2 + s*sqrt3/2 = sqrt3*s. Yes.
    # So constraint: sqrt3*x + y <= sqrt3*s
    # Similarly for all 6 edges.
    d1 = s - lx                    # x <= s
    d2 = s + lx                    # -x <= s
    d3 = sqrt3 * s - (sqrt3 * lx + ly)    # sqrt3*x + y <= sqrt3*s
    d4 = sqrt3 * s - (-sqrt3 * lx + ly)   # -sqrt3*x + y <= sqrt3*s
    d5 = sqrt3 * s - (-sqrt3 * lx - ly)   # -sqrt3*x - y <= sqrt3*s
    d6 = sqrt3 * s - (sqrt3 * lx - ly)    # sqrt3*x - y <= sqrt3*s
    return min(d1, d2, d3, d4, d5, d6) >= -1e-9

def _min_side_analytic(positions, angles, outer_cx, outer_cy, outer_angle):
    """Compute minimum outer side length analytically using support function."""
    a_outer = np.radians(outer_angle)
    # The 6 outward normals of the outer hexagon
    normals = []
    for i in range(6):
        theta = a_outer + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)  # shape (6,2)
    
    # For each inner hexagon vertex, compute the maximum over all 6 normals of
    # (dot(v - center, n)) / (s * cos(pi/6)) where s is the side length.
    # Actually, the support function of a regular hexagon with side s in direction n is:
    # h(n) = s * cos(pi/6) * max over 6 vertices... 
    # Simpler: a point p is inside hexagon(center, s, angle) iff for all 6 edge normals n_i:
    #   dot(p - center, n_i) <= s * cos(pi/6) = s * sqrt(3)/2
    # Wait, that's for the inradius. The inradius of a regular hexagon with side s is s*sqrt(3)/2.
    # So the constraint is dot(p-center, n_i) <= s * sqrt(3)/2 for all i.
    # Therefore s >= dot(p-center, n_i) / (sqrt(3)/2) for all i, all p.
    # Minimum s = max over all p, all i of dot(p-center, n_i) / (sqrt(3)/2)
    
    sqrt3 = np.sqrt(3)
    inradius_factor = sqrt3 / 2.0  # inradius / side = sqrt(3)/2
    
    max_val = 0.0
    for i in range(len(positions)):
        x, y = positions[i]
        ang = angles[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        for v in verts:
            rel = np.array([v[0] - outer_cx, v[1] - outer_cy])
            dots = normals @ rel  # (6,)
            # Each dot must be <= s * sqrt(3)/2
            required_s = dots.max() / inradius_factor
            if required_s > max_val:
                max_val = required_s
    
    return max(max_val, 1.0)

def _hexagon_overlap_fast(verts1, verts2, tol=1e-9):
    """SAT overlap check for two convex hexagons."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(6):
            edge = verts[(i+1)%6] - verts[i]
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    for axis in axes:
        proj1 = verts1 @ axis
        proj2 = verts2 @ axis
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts_list.append(_hex_vertices(x, y, 1.0, ang))
    for i in range(len(verts_list)):
        for j in range(i+1, len(verts_list)):
            if _hexagon_overlap_fast(verts_list[i], verts_list[j]):
                return False
    for verts in verts_list:
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                return False
    return True

def _initial_packing():
    """Hexagonal close packing initial configuration."""
    positions = []
    angles = []
    sqrt3 = np.sqrt(3)
    # 3-3-3-2 arrangement or similar for 11 hexagons
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons (offset)
    # Actually let's do a more compact arrangement
    # Center + ring of 6 + outer partial
    positions.append([0, 0])
    angles.append(0.0)
    for k in range(6):
        theta = k * np.pi / 3
        positions.append([2.0 * np.cos(theta), 2.0 * np.sin(theta)])
        angles.append(0.0)
    # 5 more for total 11
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi/5
        r = 2 * 2.0 * np.cos(np.pi/3) + 1.0  # roughly 3
        positions.append([r * np.cos(theta), r * np.sin(theta)])
        angles.append(0.0)
    return np.array(positions[:N]), np.array(angles[:N])

def _objective(params, outer_cx, outer_cy, outer_angle):
    """Objective: minimum outer side + overlap penalty."""
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    side = _min_side_analytic(pos, ang, outer_cx, outer_cy, outer_angle)
    # Overlap penalty
    penalty = 0.0
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i]))
    for i in range(n):
        for j in range(i+1, n):
            if _hexagon_overlap_fast(verts_list[i], verts_list[j]):
                penalty += 10.0
    return side + penalty

def construct():
    global DEADLINE
    
    # Initial packing
    pos, ang = _initial_packing()
    n = N
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Build initial parameter vector
    params = np.concatenate([pos.flatten(), ang])
    
    current_side = _min_side_analytic(pos, ang, outer_cx, outer_cy, outer_angle)
    best_params = params.copy()
    best_side = current_side
    
    # Check initial validity
    if not _check_validity(
        [[pos[i][0], pos[i][1], ang[i]] for i in range(n)],
        (outer_cx, outer_cy), current_side, outer_angle
    ):
        # Spread out more
        for i in range(n):
            pos[i] *= 1.5
        params = np.concatenate([pos.flatten(), ang])
        current_side = _min_side_analytic(pos, ang, outer_cx, outer_cy, outer_angle)
        best_params = params.copy()
        best_side = current_side
    
    # Simulated annealing phase
    T = 2.0  # Initial temperature
    T_min = 0.001
    cooling = 0.995
    step_size = 0.1
    
    if time.time() < DEADLINE - 20.0:
        while T > T_min and time.time() < DEADLINE - 20.0:
            # Random perturbation
            new_params = best_params + np.random.randn(len(best_params)) * step_size
            
            # Keep angles in reasonable range
            new_params[2*n:] = np.mod(new_params[2*n:], 2*np.pi)
            
            new_obj = _objective(new_params, outer_cx, outer_cy, outer_angle)
            cur_obj = _objective(best_params, outer_cx, outer_cy, outer_angle)
            
            if new_obj < cur_obj or np.random.random() < np.exp(-(new_obj - cur_obj) / T):
                best_params = new_params
                if new_obj < best_side:
                    best_side = new_obj
                # Adapt step size
                if new_obj < cur_obj:
                    step_size = min(step_size * 1.05, 0.5)
                else:
                    step_size *= 0.98
            else:
                step_size *= 0.99
            
            T *= cooling
    
    # Local polish with steepest descent
    if time.time() < DEADLINE - 5.0:
        step = 0.01
        improved = True
        while improved and time.time() < DEADLINE - 5.0:
            improved = False
            cur_obj = _objective(best_params, outer_cx, outer_cy, outer_angle)
            
            for i in range(len(best_params)):
                if time.time() > DEADLINE - 5.0:
                    break
                # Central difference
                h = 1e-4
                p_plus = best_params.copy()
                p_minus = best_params.copy()
                p_plus[i] += h
                p_minus[i] -= h
                # Fix angle bounds
                if i >= 2*n:
                    p_plus[i] = np.mod(p_plus[i], 2*np.pi)
                    p_minus[i] = np.mod(p_minus[i], 2*np.pi)
                
                f_plus = _objective(p_plus, outer_cx, outer_cy, outer_angle)
                f_minus = _objective(p_minus, outer_cx, outer_cy, outer_angle)
                grad = (f_plus - f_minus) / (2*h)
                
                if abs(grad) < 1e-6:
                    continue
                
                # Try moving in negative gradient direction
                new_params = best_params.copy()
                new_params[i] -= step * np.sign(grad)
                if i >= 2*n:
                    new_params[i] = np.mod(new_params[i], 2*np.pi)
                
                new_obj = _objective(new_params, outer_cx, outer_cy, outer_angle)
                if new_obj < cur_obj - 1e-8:
                    best_params = new_params
                    cur_obj = new_obj
                    improved = True
                    if new_obj < best_side:
                        best_side = new_obj
                    break  # Restart coordinate scan
            
            if not improved:
                step *= 0.7
                if step < 0.0001:
                    break
    
    # Extract final result
    final_pos = best_params[:2*n].reshape(n, 2)
    final_ang = best_params[2*n:]
    final_side = _min_side_analytic(final_pos, final_ang, outer_cx, outer_cy, outer_angle)
    
    # Final validity check
    inner_list = [[float(final_pos[i][0]), float(final_pos[i][1]), float(final_ang[i])] for i in range(n)]
    if _check_validity(inner_list, (outer_cx, outer_cy), final_side, outer_angle):
        return inner_list, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle)
    
    # Fallback: valid but loose packing
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2 + 1)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
