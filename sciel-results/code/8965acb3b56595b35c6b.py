import math
import time
import numpy as np
from scipy.optimize import linprog, minimize
from itertools import combinations

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    # Reuse from current program
    pass

def _generate_lattice_configs():
    # Returns list of lists of (x,y) for 11 hexagons
    # Center
    center = (0.0, 0.0)
    
    # Lattice basis
    u = np.array([_SQRT3, 0.0])
    v = np.array([_SQRT3/2, 1.5])
    
    def get_pos(i, j):
        return tuple(i*u + j*v)
        
    # Ring 1 (i,j)
    ring1_coords = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring1_pos = [get_pos(i,j) for i,j in ring1_coords]
    
    # Ring 2
    # Generate neighbors of ring1
    ring2_set = set()
    for i,j in ring1_coords:
        neighbors = [
            (i+1, j), (i-1, j),
            (i, j+1), (i, j-1),
            (i+1, j-1), (i-1, j+1)
        ]
        for ni, nj in neighbors:
            if (ni, nj) not in ring1_coords and (ni, nj) != (0,0):
                ring2_set.add((ni, nj))
                
    # Verify count is 12
    # ring2_pos = [get_pos(i,j) for i,j in ring2_set]
    
    # Generate all combinations of 4 from ring2
    configs = []
    ring2_list = list(ring2_set)
    for combo in combinations(range(len(ring2_list)), 4):
        selected_coords = [ring2_list[i] for i in combo]
        positions = [center] + ring1_pos + [get_pos(i,j) for i,j in selected_coords]
        configs.append(positions)
        
    return configs

def construct():
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1: Combinatorial Lattice Search
    configs = _generate_lattice_configs()
    candidates = []
    
    for idx, positions in enumerate(configs):
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Inner angle 0 or 30? Lattice hexagons are usually aligned.
        # Let's try 0.0 and 30.0? 
        # Actually, the orientation of inner hexagons relative to each other is fixed by the lattice.
        # The lattice vectors define the orientation.
        # A hexagon at (0,0) with angle 0 has vertices at angles 0, 60, 120...
        # A hexagon at (1,0) (dist sqrt(3))...
        # In the honeycomb lattice, all hexagons are aligned.
        # So inner_angle = 0.0 is the natural choice.
        
        inner_angle = 0.0
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        
        # Coarse angle search
        best_local_side = float('inf')
        best_local_angle = 0.0
        best_local_center = (0.0, 0.0)
        
        # Optimization: The optimal outer angle is likely close to the symmetry axes of the cluster.
        # We can just sweep 0-30 deg.
        for outer_angle_deg in np.arange(0, 30.0, 1.0):
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_angle = outer_angle_deg
                    best_local_center = (cx, cy)
                    
        # Fine search
        if best_local_side < float('inf'):
            for outer_angle_deg in np.arange(max(0, best_local_angle - 1.0), min(30.0, best_local_angle + 1.0), 0.05):
                res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_local_side:
                        best_local_side = side
                        best_local_angle = outer_angle_deg
                        best_local_center = (cx, cy)
                        
        if best_local_side < float('inf'):
            candidates.append((best_local_side, positions, best_local_center, best_local_angle))
            
    if not candidates:
        # Fallback
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Sort candidates
    candidates.sort(key=lambda x: x[0])
    
    # Take top 5 for polishing
    top_candidates = candidates[:5]
    
    for side, positions, center, angle in top_candidates:
        if time.time() > _DEADLINE - 10.0:
            break
        if side >= best_side:
            continue # Already worse than best found? No, best_side is global.
            # Actually, we want to polish the best ones.
            
        # Initialize params for L-BFGS-B
        x0 = []
        for px, py in positions:
            x0.append(px)
            x0.append(py)
            x0.append(0.0) # angle rad
            
        # Objective function (reuse from current program logic)
        # We need to define _objective and _all_pairwise_dists and _softmin
        # I will copy them from the current program to ensure they work.
        
        try:
            res = minimize(_objective, x0, args=([], angle, 100.0),
                          method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_side and res.fun < 100:
                # Verify and update
                new_positions = []
                new_angles = []
                for i in range(N):
                    new_positions.append((res.x[3*i], res.x[3*i+1]))
                    new_angles.append(math.degrees(res.x[3*i+2]))
                    
                dists = _all_pairwise_dists(new_positions, [math.radians(a) for a in new_angles])
                if len(dists) > 0 and np.min(dists) < -1e-4:
                    continue
                    
                pos_arr = np.array(new_positions)
                ang_deg_arr = np.array(new_angles)
                base_angles = np.radians(ang_deg_arr)[:, None] + np.arange(6) * math.pi / 3.0
                cos_a = np.cos(base_angles)
                sin_a = np.sin(base_angles)
                all_vertices = np.column_stack([
                    pos_arr[:, 0:1].repeat(6, axis=1) + cos_a,
                    pos_arr[:, 1:2].repeat(6, axis=1) + sin_a
                ])
                cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, angle)
                
                if side < best_side:
                    best_side = side
                    best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                    best_center = (cx, cy)
                    best_outer_angle = angle
                    best_params = res.x[:]
                    
        except Exception:
            pass
            
    # Fallback if no improvement found
    if best_inner is None:
        # Use the best lattice config
        side, positions, center, angle = candidates[0]
        best_side = side
        best_inner = [[px, py, 0.0] for px, py in positions]
        best_center = center
        best_outer_angle = angle
        best_params = []
        for px, py in positions:
            best_params.append(px)
            best_params.append(py)
            best_params.append(0.0)

    # Final sanity check
    if best_inner is None or best_side >= float('inf'):
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    best_inner = best_inner[:N]
    while len(best_inner) < N:
        best_inner.append([0.0, 0.0, 0.0])
        
    return best_inner, best_center, best_side, best_outer_angle

# Helper functions from current program
def _hex_distance_fast(p1, a1, p2, a2):
    d = np.linalg.norm(p1 - p2)
    if d < 1e-10:
        return 0.0
    dir_vec = (p2 - p1) / d
    theta = math.atan2(dir_vec[1], dir_vec[0])
    vals1 = np.cos(a1 + np.arange(6) * math.pi / 3.0 - theta)
    vals2 = np.cos(a2 + np.arange(6) * math.pi / 3.0 - (theta + math.pi))
    h1 = np.max(vals1)
    h2 = np.max(vals2)
    return d - h1 - h2

def _all_pairwise_dists(positions, angles):
    n = len(positions)
    pos_arr = np.array(positions)
    ang_arr = np.array(angles)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = _hex_distance_fast(pos_arr[i], ang_arr[i], pos_arr[j], ang_arr[j])
            dists.append(d)
    return np.array(dists)

def _softmin(vals, beta=100.0):
    m = np.min(vals)
    shifted = vals - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta

def _objective(params, base_positions, outer_angle_deg, beta=80.0):
    n = len(base_positions)
    positions = []
    angles_rad = []
    for i in range(n):
        positions.append((params[3*i], params[3*i+1]))
        angles_rad.append(params[3*i+2])
    
    dists = _all_pairwise_dists(positions, angles_rad)
    if len(dists) == 0:
        dists = np.array([1.0])
    
    sm = _softmin(dists, beta)
    barrier = max(0.0, -sm) ** 2 * 50.0
    
    pos_arr = np.array(positions)
    ang_deg = np.degrees(np.array(angles_rad))
    base_angles = np.radians(ang_deg)[:, None] + np.arange(6) * math.pi / 3.0
    cos_a = np.cos(base_angles)
    sin_a = np.sin(base_angles)
    all_vertices = np.column_stack([
        pos_arr[:, 0:1].repeat(6, axis=1) + cos_a,
        pos_arr[:, 1:2].repeat(6, axis=1) + sin_a
    ])
    
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    return side + barrier
