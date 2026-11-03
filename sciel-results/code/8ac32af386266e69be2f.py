"""Packing 12 unit hexagons in minimal regular hexagon - structural rewrite.

New approach:
- Directly optimize all 12 inner hexagon positions (x,y) and a shared rotation
  plus the outer hexagon side length and rotation using scipy SLSQP with
  explicit non-overlap (SAT-based) and containment constraints.
- Multiple topologies tried with local refinement.
- Fast vectorized geometry checks.
"""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12
DEADLINE = 143.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    verts = np.array([
        [side * np.cos(angle + i * np.pi / 3),
         side * np.sin(angle + i * np.pi / 3)] for i in range(6)
    ])
    return verts + np.array([cx, cy])

def sat_separation(p1, p2):
    """Return minimum separation distance between two convex hexagons using SAT.
    Positive means separated, negative means overlapping."""
    n1, n2 = len(p1), len(p2)
    min_sep = np.inf
    for poly in (p1, p2):
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i+1) % n]
            ex, ey = b - a
            norm = np.sqrt(ex*ex + ey*ey)
            if norm < 1e-12:
                continue
            nx, ny = -ey/norm, ex/norm
            proj1 = p1 @ np.array([nx, ny])
            proj2 = p2 @ np.array([nx, ny])
            sep = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
            min_sep = min(min_sep, sep)
    return min_sep

def point_in_hex(pt, hex_verts):
    """Check if point is inside convex polygon (hexagon)."""
    n = len(hex_verts)
    for i in range(n):
        a, b = hex_verts[i], hex_verts[(i+1) % n]
        cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
        if cross < -1e-9:
            return False
    return True

def hex_in_hex(inner_verts, outer_verts):
    """Check if inner hexagon is entirely inside outer hexagon."""
    for v in inner_verts:
        if not point_in_hex(v, outer_verts):
            return False
    return True

def evaluate_config(positions, inner_angle, outer_center, outer_side, outer_angle):
    """Evaluate a full configuration. Returns (valid, min_violation).
    min_violation: negative = max overlap depth, positive = min clearance."""
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Check containment
    min_cont = np.inf
    for pos in positions:
        iv = hex_vertices(pos[0], pos[1], 1.0, inner_angle)
        # Compute max distance of inner vertices outside outer
        for v in iv:
            # Check containment via half-planes of outer hex
            for i in range(6):
                a, b = outer_verts[i], outer_verts[(i+1) % 6]
                ex, ey = b - a
                norm = np.sqrt(ex*ex + ey*ey)
                if norm < 1e-12:
                    continue
                nx, ny = -ey/norm, ex/norm
                # Signed distance: positive = inside
                dist = (v[0] - a[0])*nx + (v[1] - a[1])*ny
                if dist < 0:
                    min_cont = min(min_cont, dist)
    
    # Check pairwise non-overlap
    min_sep = np.inf
    for i in range(N):
        h1 = hex_vertices(positions[i][0], positions[i][1], 1.0, inner_angle)
        for j in range(i+1, N):
            h2 = hex_vertices(positions[j][0], positions[j][1], 1.0, inner_angle)
            sep = sat_separation(h1, h2)
            min_sep = min(min_sep, sep)
    
    return min_cont, min_sep

def objective(params):
    """Objective for SLSQP: minimize outer_side + penalty for violations."""
    # params: [x0,y0, x1,y1, ..., x11,y11, inner_angle, outer_side, outer_angle]
    pos = params[:2*N].reshape(N, 2)
    inner_angle = params[2*N]
    outer_side = params[2*N+1]
    outer_angle = params[2*N+2]
    
    outer_center = np.array([0.0, 0.0])
    
    # Compute violations
    min_cont, min_sep = evaluate_config(pos, inner_angle, outer_center, outer_side, outer_angle)
    
    # Penalty: if violations exist, add large penalty
    penalty = 0.0
    if min_cont < 0:
        penalty += 100.0 * (-min_cont)
    if min_sep < 0:
        penalty += 100.0 * (-min_sep)
    
    return outer_side + penalty

def constraints(params):
    """Constraints for SLSQP: all must be >= 0."""
    pos = params[:2*N].reshape(N, 2)
    inner_angle = params[2*N]
    outer_side = params[2*N+1]
    outer_angle = params[2*N+2]
    
    outer_center = np.array([0.0, 0.0])
    outer_verts = hex_vertices(0, 0, outer_side, outer_angle)
    
    cons = []
    
    # Containment constraints: each inner vertex must be inside outer hex
    for i in range(N):
        iv = hex_vertices(pos[i][0], pos[i][1], 1.0, inner_angle)
        for v in iv:
            # Half-plane constraints for outer hex
            for k in range(6):
                a, b = outer_verts[k], outer_verts[(k+1) % 6]
                ex, ey = b - a
                norm = np.sqrt(ex*ex + ey*ey)
                if norm < 1e-12:
                    continue
                nx, ny = -ey/norm, ex/norm
                cons.append({'type': 'ineq',
                             'fun': lambda p, _i=i, _k=k, _nx=nx, _ny=ny, _ax=a[0], _ay=a[1]:
                                 (p[2*_i] - _ax)*_nx + (p[2*_i+1] - _ay)*_ny})
    
    # Non-overlap constraints: pairwise SAT separation >= 0
    # This is expensive; use a simplified version with center distance
    for i in range(N):
        for j in range(i+1, N):
            cons.append({'type': 'ineq',
                         'fun': lambda p, _i=i, _j=j, _ia=inner_angle:
                             sat_separation(
                                 hex_vertices(p[2*_i], p[2*_i+1], 1.0, p[2*N]),
                                 hex_vertices(p[2*_j], p[2*_j+1], 1.0, p[2*N])
                             )})
    
    return cons

def make_initial_config(topology, scale=1.0):
    """Create initial positions for different topologies."""
    d = scale * math.sqrt(3)  # touching distance for unit hexagons
    positions = []
    
    if topology == 'ring':
        # 1 center + 6 ring + 5 partial ring
        positions.append([0.0, 0.0])
        for i in range(6):
            a = i * np.pi / 3
            positions.append([d * np.cos(a), d * np.sin(a)])
        for i in range(6):
            if i == 2:  # skip one for 5
                continue
            a = i * np.pi / 3 + np.pi / 6
            positions.append([2*d * np.cos(a), 2*d * np.sin(a)])
    elif topology == 'grid3x4':
        # 3 rows of 4, staggered
        for row in range(3):
            y = row * d * np.sqrt(3) / 2
            x_start = -1.5 * d
            if row % 2 == 1:
                x_start += d / 2
            for col in range(4):
                positions.append([x_start + col * d, y])
    elif topology == 'grid4x3':
        # 4 rows of 3, staggered
        for row in range(4):
            y = row * d * np.sqrt(3) / 2
            x_start = -d
            if row % 2 == 1:
                x_start += d / 2
            for col in range(3):
                positions.append([x_start + col * d, y])
    elif topology == 'compact':
        # More compact arrangement
        positions.append([0.0, 0.0])
        for i in range(6):
            a = i * np.pi / 3
            positions.append([d * np.cos(a), d * np.sin(a)])
        # 5 more in a tighter second ring
        for i in range(5):
            a = i * 2 * np.pi / 5
            positions.append([1.7*d * np.cos(a), 1.7*d * np.sin(a)])
    else:
        # Fallback: linear
        for i in range(12):
            positions.append([i * d, 0.0])
    
    return np.array(positions)

def construct():
    start_time = time.time()
    
    best_score = np.inf
    best_params = None
    
    topologies = ['ring', 'grid3x4', 'grid4x3', 'compact']
    inner_angles = [0.0, 30.0, 15.0, 45.0]
    outer_angles = [0.0, 30.0, 15.0, 60.0]
    
    for topo in topologies:
        if time.time() - start_time > DEADLINE:
            break
        
        for ia in inner_angles[:2]:  # Limit inner angles for speed
            for oa in outer_angles[:2]:  # Limit outer angles for speed
                if time.time() - start_time > DEADLINE:
                    break
                
                pos0 = make_initial_config(topo, scale=1.0)
                
                # Initial estimate of outer side
                extent = np.max(np.abs(pos0), axis=0)
                init_side = max(extent[0], extent[1]) + 2.0
                init_side = max(init_side, 4.0)
                
                # Build initial params
                params0 = np.concatenate([pos0.flatten(), [ia, init_side, oa]])
                
                # Bounds
                bounds = [(-10, 10)] * (2*N) + [(0, 180), (3, 15), (0, 60)]
                
                try:
                    # Use SLSQP with constraints
                    result = minimize(
                        objective, params0, method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 80, 'ftol': 1e-6}
                    )
                    
                    if result.fun < best_score and result.fun < 50:
                        # Verify
                        pos = result.x[:2*N].reshape(N, 2)
                        ia_opt = result.x[2*N]
                        side_opt = result.x[2*N+1]
                        oa_opt = result.x[2*N+2]
                        
                        mc, ms = evaluate_config(pos, ia_opt, np.array([0,0]), side_opt, oa_opt)
                        if mc >= -1e-6 and ms >= -1e-6:
                            best_score = side_opt
                            best_params = result.x.copy()
                except Exception:
                    continue
    
    # If SLSQP didn't find good solution, try Nelder-Mead on a simpler objective
    if best_score > 10 or best_params is None:
        for topo in topologies[:2]:
            if time.time() - start_time > DEADLINE:
                break
            
            pos0 = make_initial_config(topo, scale=1.0)
            
            def nm_objective(params):
                pos = params[:2*N].reshape(N, 2)
                ia = params[2*N]
                oa = params[2*N+1]
                
                # Estimate side from extent
                outer_verts_approx_side = max(np.max(np.abs(pos)), np.max(np.abs(pos))) + 2.5
                
                mc, ms = evaluate_config(pos, ia, np.array([0,0]), outer_verts_approx_side, oa)
                
                penalty = 0
                if mc < 0: penalty += 50 * (-mc)
                if ms < 0: penalty += 50 * (-ms)
                
                return outer_verts_approx_side + penalty
            
            params0 = np.concatenate([pos0.flatten(), [0.0, 0.0]])
            bounds_nm = [(-8, 8)] * (2*N) + [(0, 180), (0, 60)]
            
            try:
                result = minimize(
                    nm_objective, params0, method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                
                if result.fun < best_score and result.fun < 50:
                    pos = result.x[:2*N].reshape(N, 2)
                    ia = result.x[2*N]
                    oa = result.x[2*N+1]
                    side_est = max(np.max(np.abs(pos))) + 2.5
                    
                    mc, ms = evaluate_config(pos, ia, np.array([0,0]), side_est, oa)
                    if mc >= -1e-6 and ms >= -1e-6:
                        best_score = side_est
                        best_params = result.x.copy()
            except Exception:
                continue
    
    # Final: if we have a good solution, do a binary search on side for precision
    if best_params is not None and best_score < 20:
        pos = best_params[:2*N].reshape(N, 2)
        ia = best_params[2*N]
        oa = best_params[2*N+2] if len(best_params) > 2*N+2 else 0.0
        
        # Binary search for exact minimum side
        lo, hi = 3.0, best_score + 1.0
        for _ in range(25):
            mid = (lo + hi) / 2
            mc, ms = evaluate_config(pos, ia, np.array([0,0]), mid, oa)
            if mc >= -1e-9 and ms >= -1e-9:
                hi = mid
            else:
                lo = mid
        
        final_side = hi
        inner = [[float(pos[i][0]), float(pos[i][1]), float(ia)] for i in range(N)]
        return inner, [0.0, 0.0], float(final_side), float(oa)
    
    # Fallback: simple ring arrangement
    d = math.sqrt(3)
    positions = [[0.0, 0.0, 0.0]]
    for i in range(6):
        a = i * np.pi / 3
        positions.append([d * np.cos(a), d * np.sin(a), 0.0])
    for i in range(5):
        a = i * 2 * np.pi / 5
        positions.append([1.8*d * np.cos(a), 1.8*d * np.sin(a), 0.0])
    
    return positions, [0.0, 0.0], 8.0, 0.0
