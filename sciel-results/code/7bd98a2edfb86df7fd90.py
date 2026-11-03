import math
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate all axial coordinates within distance 3 from origin
    # Axial distance: max(abs(q), abs(r), abs(-q-r)) <= 3
    lattice_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if max(abs(q), abs(r), abs(-q-r)) <= 3:
                lattice_points.append((q, r))
    
    # We need to choose 11 points from this set such that they are pairwise disjoint
    # Disjointness in axial coordinates for unit hexagons:
    # Two hexagons at (q1, r1) and (q2, r2) are disjoint if their distance > 1.
    # Distance in axial coords: (|q1-q2| + |r1-r2| + |q1+r1-q2-r2|) / 2
    # So we need distance >= 2, i.e., |q1-q2| + |r1-r2| + |q1+r1-q2-r2| >= 4
    
    def axial_dist(q1, r1, q2, r2):
        return (abs(q1-q2) + abs(r1-r2) + abs(q1+r1-q2-r2)) // 2

    # Precompute valid pairs
    n_pts = len(lattice_points)
    valid_pair = [[True]*n_pts for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            d = axial_dist(lattice_points[i][0], lattice_points[i][1], 
                           lattice_points[j][0], lattice_points[j][1])
            if d < 2:
                valid_pair[i][j] = False
                valid_pair[j][i] = False

    # Backtracking search for 11 disjoint points
    # To speed up, we can sort points by some heuristic, e.g., distance from center
    # But simple backtracking with pruning should work for 37 points choosing 11
    
    def search_combinations(index, current_set, remaining):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        if len(current_set) == 11:
            return list(current_set)
        
        if index >= n_pts:
            return None
            
        if len(current_set) + (n_pts - index) < 11:
            return None
            
        # Option 1: Include lattice_points[index]
        if all(valid_pair[index, i] for i in current_set):
            res = search_combinations(index + 1, current_set + [index], remaining - 1)
            if res is not None:
                return res
                
        # Option 2: Exclude lattice_points[index]
        res = search_combinations(index + 1, current_set, remaining)
        if res is not None:
            return res
            
        return None

    # We want to find configurations that minimize the enclosing hexagon size.
    # The size depends on the spread of the points.
    # Let's generate multiple valid configurations and evaluate them.
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta, cx=None, cy=None):
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        if cx is None or cy is None:
            # Use centroid as initial center
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            
        D = 0
        for nx, ny in normals:
            for px, py in pts:
                proj = abs((px - cx) * nx + (py - cy) * ny)
                D = max(D, proj)
        
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Coarse grid search over beta
        for i in range(61):
            beta = i * (math.pi / 6) / 60
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 1000, 'xatol': 1e-16})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    # Instead of finding just one combination, let's find a few diverse ones or the best one.
    # Given the time limit, finding one good one is safer.
    # The "minimal 3x3 block" mentioned in the prompt suggests points close to center.
    # Let's bias the search towards points closer to the origin.
    
    # Sort lattice points by distance from origin to prioritize central packing
    sorted_indices = sorted(range(n_pts), key=lambda i: lattice_points[i][0]**2 + lattice_points[i][1]**2)
    
    # Map sorted indices back to original indices for valid_pair checks
    # Actually, let's just run the search on the sorted list but keep track of original indices
    # Or simpler: just use the search function but it might pick far points.
    # Let's modify search to prioritize central points.
    
    def search_combinations_prioritized(index, current_set, remaining):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        if len(current_set) == 11:
            return list(current_set)
        
        if index >= len(sorted_indices):
            return None
            
        if len(current_set) + (len(sorted_indices) - index) < 11:
            return None
            
        orig_idx = sorted_indices[index]
        
        # Option 1: Include
        if all(valid_pair[orig_idx, curr] for curr in current_set):
            res = search_combinations_prioritized(index + 1, current_set + [orig_idx], remaining - 1)
            if res is not None:
                return res
                
        # Option 2: Exclude
        res = search_combinations_prioritized(index + 1, current_set, remaining)
        if res is not None:
            return res
            
        return None

    found_indices = search_combinations_prioritized(0, [], 11)
    
    if found_indices is None:
        # Fallback to a known good configuration
        axial_fallback = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        pts = axial_to_cartesian(axial_fallback)
    else:
        axial_found = [lattice_points[i] for i in found_indices]
        pts = axial_to_cartesian(axial_found)

    best_global_S, best_global_beta, best_global_center = optimize_shape(pts)
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Safety scale
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
