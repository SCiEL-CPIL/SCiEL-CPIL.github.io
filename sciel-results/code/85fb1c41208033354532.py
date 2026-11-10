# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import differential_evolution, minimize

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, angle angle_deg."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def get_edge_normals(verts):
    """Get outward-pointing edge normals for a hexagon given its vertices."""
    normals = []
    center = np.mean(verts, axis=0)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)

def check_separation_fast(v1, v2, tol=1e-6):
    """
    Fast separation check.
    """
    n1 = get_edge_normals(v1)
    n2 = get_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap = max(min2 - max1, min1 - max2)
        if gap > tol:
            return True
    return False

def calculate_min_side_fast(inner_hex_data, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of an outer regular hexagon centered at origin.
    """
    # Normals of the outer hexagon
    # Vertices at outer_angle_deg + 60i. Normals at outer_angle_deg + 30 + 60i.
    normal_angles = np.radians(outer_angle_deg + 30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    all_vertices = []
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        verts = hexagon_vertices(x, y, 1.0, angle)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # Project all vertices onto the 6 normals
    projs = all_vertices @ normals.T
    max_projs = np.max(projs, axis=0)
    max_proj = np.max(max_projs)
    
    side = 2.0 * max_proj / math.sqrt(3)
    return side

def objective_de(params, outer_angle_deg=0.0):
    """
    Objective for Differential Evolution.
    Uses a soft penalty for overlap based on center distance for speed,
    and the exact bounding box size.
    """
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = params[3*i]
        inner_hex_data[i, 1] = params[3*i+1]
        inner_hex_data[i, 2] = params[3*i+2]
        
    side = calculate_min_side_fast(inner_hex_data, outer_angle_deg)
    
    penalty = 0.0
    # Coarse overlap check: distance between centers
    # Min distance for non-overlapping unit hexes is sqrt(3) if aligned.
    # With rotation, it can be less, but > 1.5 roughly.
    # Use 1.8 as a safe threshold for penalty activation.
    min_dist_sq = 0
    for i in range(11):
        for j in range(i+1, 11):
            dx = inner_hex_data[i, 0] - inner_hex_data[j, 0]
            dy = inner_hex_data[i, 1] - inner_hex_data[j, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 3.24: # 1.8^2
                penalty += (3.24 - dist_sq)
                
    return side + 10.0 * penalty

def objective_refined(params, outer_angle_deg=0.0):
    """
    Objective for local refinement.
    Uses exact separation check.
    """
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = params[3*i]
        inner_hex_data[i, 1] = params[3*i+1]
        inner_hex_data[i, 2] = params[3*i+2]
        
    side = calculate_min_side_fast(inner_hex_data, outer_angle_deg)
    
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            v1 = hexagon_vertices(x1, y1, 1.0, a1)
            v2 = hexagon_vertices(x2, y2, 1.0, a2)
            
            # Check separation
            n1 = get_edge_normals(v1)
            n2 = get_edge_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            max_gap = -np.inf
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                gap = max(min2 - max1, min1 - max2)
                if gap > max_gap:
                    max_gap = gap
                    
            if max_gap < 1e-6:
                penalty += (1e-6 - max_gap) ** 2
                
    return side + 100.0 * penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    # Initial guess: 4-3-4 grid
    h = math.sqrt(3) / 2
    coords = []
    for i in range(4):
        coords.append([i * 3.0, 0.0])
    for i in range(3):
        coords.append([1.5 + i * 3.0, 2 * h])
    for i in range(4):
        coords.append([i * 3.0, 4 * h])
        
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
        
    init_params = np.zeros(33)
    for i, c in enumerate(coords):
        init_params[3*i] = c[0]
        init_params[3*i+1] = c[1]
        init_params[3*i+2] = 0.0
        
    # Try a few outer angles
    best_side = float('inf')
    best_params = init_params.copy()
    best_outer_angle = 0.0
    
    # Phase 1: Differential Evolution
    # Bounds for x, y: -4 to 4
    # Bounds for angle: -30 to 30 (symmetry allows this range)
    bounds = [(-4.0, 4.0), (-4.0, 4.0), (-30.0, 30.0)] * 11
    
    # Run DE for a limited time
    de_start = time.time()
    de_deadline = 300.0 # Use 300s for DE
    
    try:
        # Use a population size that fits in time
        # maxiter=100, popsize=10 -> 1000 evals
        # Each eval is fast (coarse penalty)
        result = differential_evolution(
            objective_de,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=0.01,
            polish=False, # We will polish manually
            init='sobol'
        )
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
    except Exception:
        pass
        
    # If DE didn't run long enough or failed, use init
    if best_params is init_params:
         best_side = objective_de(init_params)
         
    # Phase 2: Local Refinement
    # Refine the best solution found
    if time.time() - start_time < deadline - safety_margin:
        # Try refining with different outer angles
        for oa in [0.0, 15.0, 30.0]:
            if time.time() - start_time > deadline - safety_margin:
                break
            
            # Adjust initial params if needed? No, just optimize.
            # The objective depends on oa.
            x0 = best_params.copy()
            
            try:
                result = minimize(
                    objective_refined,
                    x0,
                    args=(oa,),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if result.fun < best_side:
                    # Verify validity
                    inner_hex_data = np.zeros((11, 3))
                    for i in range(11):
                        inner_hex_data[i, 0] = result.x[3*i]
                        inner_hex_data[i, 1] = result.x[3*i+1]
                        inner_hex_data[i, 2] = result.x[3*i+2]
                        
                    valid = True
                    for i in range(11):
                        for j in range(i+1, 11):
                            if not check_separation_fast(inner_hex_data[i], inner_hex_data[j]):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        test_side = calculate_min_side_fast(inner_hex_data, oa)
                        if test_side < best_side:
                            best_side = test_side
                            best_params = result.x.copy()
                            best_outer_angle = oa
                            
            except Exception:
                continue

    # Phase 3: Final Polish
    if time.time() - start_time < deadline - safety_margin:
        # Fine tune outer angle
        for oa in np.arange(max(0, best_outer_angle - 5.0), min(30, best_outer_angle + 5.0) + 0.01, 0.5):
            if time.time() - start_time > deadline - safety_margin:
                break
            x0 = best_params.copy()
            try:
                result = minimize(
                    objective_refined,
                    x0,
                    args=(oa,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12}
                )
                if result.fun < best_side:
                     inner_hex_data = np.zeros((11, 3))
                     for i in range(11):
                         inner_hex_data[i, 0] = result.x[3*i]
                         inner_hex_data[i, 1] = result.x[3*i+1]
                         inner_hex_data[i, 2] = result.x[3*i+2]
                         
                     valid = True
                     for i in range(11):
                         for j in range(i+1, 11):
                             if not check_separation_fast(inner_hex_data[i], inner_hex_data[j]):
                                 valid = False
                                 break
                         if not valid:
                             break
                     
                     if valid:
                         test_side = calculate_min_side_fast(inner_hex_data, oa)
                         if test_side < best_side:
                             best_side = test_side
                             best_params = result.x.copy()
                             best_outer_angle = oa
            except Exception:
                continue

    # Construct return values
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = best_params[3*i]
        inner_hex_data[i, 1] = best_params[3*i+1]
        inner_hex_data[i, 2] = best_params[3*i+2]
        
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, best_side

# EVOLVE-BLOCK-END
