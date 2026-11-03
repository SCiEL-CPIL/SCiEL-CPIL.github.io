import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def _hexagon_vertices(cx, cy, angle, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation.
    angle is the rotation of the hexagon. Vertices at angle + k*pi/3 for k=0..5.
    """
    angles = angle + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hexagon_overlap(v1, v2, tol=1e-9):
    """Check if two hexagons (as vertex arrays of shape (6,2)) overlap using SAT."""
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    all_edges = np.vstack([edges1, edges2])
    
    for edge in all_edges:
        nx, ny = -edge[1], edge[0]
        norm = math.hypot(nx, ny)
        if norm < 1e-12:
            continue
        nx, ny = nx / norm, ny / norm
        
        proj1 = v1[:, 0] * nx + v1[:, 1] * ny
        proj2 = v2[:, 0] * nx + v2[:, 1] * ny
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 + tol < min2 or max2 + tol < min1:
            return False
    
    return True

def _fast_overlap_check(pts, inner_angle, tol=1e-9):
    """Check if any pair of unit hexagons at positions pts with rotation inner_angle overlap.
    Returns True if overlap detected, False otherwise.
    """
    n = len(pts)
    # Precompute vertices for all hexagons
    vertices = np.zeros((n, 6, 2))
    for i in range(n):
        vertices[i] = _hexagon_vertices(pts[i, 0], pts[i, 1], inner_angle)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Quick bounding box check
            v1 = vertices[i]
            v2 = vertices[j]
            if np.max(v1[:,0]) + tol < np.min(v2[:,0]) or np.max(v2[:,0]) + tol < np.min(v1[:,0]):
                continue
            if np.max(v1[:,1]) + tol < np.min(v2[:,1]) or np.max(v2[:,1]) + tol < np.min(v1[:,1]):
                continue
            if _hexagon_overlap(v1, v2, tol):
                return True
    return False

def _min_enclosing_hexagon_size(pts, inner_angle, outer_angle):
    """Compute the minimum side length S of a regular hexagon with rotation outer_angle
    that contains all inner hexagons (unit side, rotation inner_angle) centered at pts.
    Returns (S, center_x, center_y) or (inf, None, None) if infeasible.
    """
    # Collect all vertices of all inner hexagons
    n_hex = len(pts)
    all_vertices = np.zeros((n_hex * 6, 2))
    for i in range(n_hex):
        all_vertices[i*6:(i+1)*6] = _hexagon_vertices(pts[i, 0], pts[i, 1], inner_angle)
    
    normals = np.zeros((6, 2))
    for k in range(6):
        ang = outer_angle + math.pi / 6.0 + k * math.pi / 3.0
        normals[k, 0] = math.cos(ang)
        normals[k, 1] = math.sin(ang)
    
    n_v = len(all_vertices)
    A_ub_list = []
    b_ub_list = []
    
    for k in range(6):
        nx, ny = normals[k]
        for v in all_vertices:
            # (v - center) · n <= S * APOTHEM
            # v·n - center·n - S*APOTHEM <= 0
            A_ub_list.append([v[0]*nx, v[1]*nx, -nx, -ny, -APOTHEM])
            b_ub_list.append(0.0)
            # -(v - center) · n <= S * APOTHEM
            # -v·n + center·n - S*APOTHEM <= 0
            A_ub_list.append([-v[0]*nx, -v[1]*nx, nx, ny, -APOTHEM])
            b_ub_list.append(0.0)
    
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    c = [0, 0, 0, 0, 1]  # Minimize S (5th variable, index 4)
    bounds = [(None, None), (None, None), (None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            S = res.x[4]
            cx = res.x[2]
            cy = res.x[3]
            return S, cx, cy
        else:
            return float('inf'), None, None
    except Exception:
        return float('inf'), None, None

def _force_directed_layout(n_hex, max_iter=1000, dt=0.05, repulse_strength=2.0, attract_strength=0.1, seed=None):
    """
    Force-directed layout to find a compact non-overlapping arrangement of n_hex unit hexagons.
    Returns array of (x, y) positions.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize on hexagonal lattice
    positions = np.zeros((n_hex, 2))
    idx = 0
    for r in range(-4, 5):
        for q in range(-4, 5):
            if idx >= n_hex:
                break
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions[idx] = [x, y]
            idx += 1
        if idx >= n_hex:
            break
    
    # Add small random perturbation
    positions += np.random.uniform(-0.1, 0.1, size=(n_hex, 2))
    
    for iteration in range(max_iter):
        forces = np.zeros((n_hex, 2))
        
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    diff = np.random.uniform(-0.01, 0.01, size=2)
                    dist = np.linalg.norm(diff)
                
                # Two hexagons with same orientation just touch when center distance = 2 * APOTHEM = sqrt(3)
                threshold = math.sqrt(3)
                if dist < threshold:
                    force_mag = repulse_strength * (threshold - dist) / threshold
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
                elif dist < threshold * 2:
                    force_mag = attract_strength * (dist - threshold) / threshold
                    forces[i] -= force_mag * diff / dist
                    forces[j] += force_mag * diff / dist
        
        centroid = np.mean(positions, axis=0)
        for i in range(n_hex):
            forces[i] += -0.05 * (positions[i] - centroid)
        
        positions += dt * forces
        dt *= 0.998
    
    positions -= np.mean(positions, axis=0)
    return positions

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_S = float('inf')
    best_pts = None
    best_inner_angle = 0.0
    best_outer_angle = 0.0
    best_center = (0.0, 0.0)

    # Generate candidate positions
    candidates = []
    
    # Candidate 1: Center + ring of 6 + 4 more
    pts1 = [(0, 0)]
    for k in range(6):
        ang = k * math.pi / 3
        x = math.sqrt(3) * math.cos(ang)
        y = math.sqrt(3) * math.sin(ang)
        pts1.append((x, y))
    for k in range(4):
        ang = k * math.pi / 2
        x = 2 * math.sqrt(3) * math.cos(ang)
        y = 2 * math.sqrt(3) * math.sin(ang)
        pts1.append((x, y))
    candidates.append(np.array(pts1))
    
    # Candidate 2: Force-directed with various seeds
    for seed in range(30):
        pos = _force_directed_layout(N, max_iter=400, dt=0.03, 
                                       repulse_strength=1.5 + (seed % 5) * 0.3,
                                       attract_strength=0.05 + (seed % 3) * 0.02,
                                       seed=seed)
        candidates.append(pos)
    
    # Try several inner and outer rotations
    inner_angles = [0.0, math.pi/6, math.pi/3, math.pi/4, math.pi/12, 5*math.pi/12, math.pi/2, 7*math.pi/12]
    outer_angles = [0.0, math.pi/6, math.pi/3, math.pi/4, math.pi/12, 5*math.pi/12, math.pi/2, 7*math.pi/12]
    
    for pts in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        for inner_ang in inner_angles:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            if _fast_overlap_check(pts, inner_ang, tol=1e-9):
                continue
            
            for outer_ang in outer_angles:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                S, cx, cy = _min_enclosing_hexagon_size(pts, inner_ang, outer_ang)
                if S < best_S:
                    best_S = S
                    best_pts = pts.copy()
                    best_inner_angle = inner_ang
                    best_outer_angle = outer_ang
                    best_center = (cx, cy)
    
    # Second pass: Optimize positions for the best configuration
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        
        def objective(params):
            n_pts = N
            pts = params[:2*n_pts].reshape(n_pts, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            outer_ang = params[2*n_pts+2]
            
            outer_ang = ((outer_ang % (math.pi/3)) + math.pi/3) % (math.pi/3)
            
            S, _, _ = _min_enclosing_hexagon_size(pts, best_inner_angle, outer_ang)
            if S == float('inf'):
                return 1e6
            
            penalty = 0.0
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    dx = pts[i,0] - pts[j,0]
                    dy = pts[i,1] - pts[j,1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < 3.0 - 1e-6:
                        v1 = _hexagon_vertices(pts[i,0], pts[i,1], best_inner_angle)
                        v2 = _hexagon_vertices(pts[j,0], pts[j,1], best_inner_angle)
                        if _hexagon_overlap(v1, v2, tol=1e-9):
                            penalty += 10.0 * (3.0 - dist_sq)
            
            return S + penalty
        
        init_params = list(best_pts.flatten()) + [best_center[0], best_center[1], best_outer_angle]
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize(objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if res.fun < best_S:
                    new_pts = res.x[:2*N].reshape(N, 2)
                    new_cx = res.x[2*N]
                    new_cy = res.x[2*N+1]
                    new_outer = ((res.x[2*N+2] % (math.pi/3)) + math.pi/3) % (math.pi/3)
                    
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            v1 = _hexagon_vertices(new_pts[i,0], new_pts[i,1], best_inner_angle)
                            v2 = _hexagon_vertices(new_pts[j,0], new_pts[j,1], best_inner_angle)
                            if _hexagon_overlap(v1, v2, tol=1e-9):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        S_final, cx_final, cy_final = _min_enclosing_hexagon_size(new_pts, best_inner_angle, new_outer)
                        if S_final < best_S:
                            best_S = S_final
                            best_pts = new_pts
                            best_center = (cx_final, cy_final)
                            best_outer_angle = new_outer
        except Exception:
            pass
    
    # Final result
    if best_pts is None:
        best_pts = np.zeros((N, 2))
        for i in range(N):
            best_pts[i] = [i * math.sqrt(3), 0]
        best_S = 10.0
        best_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_inner_angle = 0.0
    
    cx, cy = best_center
    inner_angle_deg = math.degrees(best_inner_angle)
    outer_angle_deg = math.degrees(best_outer_angle)
    
    S_final, cx_final, cy_final = _min_enclosing_hexagon_size(best_pts, best_inner_angle, best_outer_angle)
    if S_final < float('inf'):
        best_S = S_final
        best_center = (cx_final, cy_final)
    
    cx, cy = best_center
    inner_angle_deg = math.degrees(best_inner_angle)
    outer_angle_deg = math.degrees(best_outer_angle)
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(inner_angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(outer_angle_deg)
    
    return inner, center, side, rotation
