import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def _sat_overlap_depth(v1, v2):
    """
    Compute the minimum translation distance to separate two convex polygons v1, v2.
    Returns 0 if no overlap, positive value if overlapping.
    v1, v2 are (6, 2) arrays of vertices.
    """
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            edge = p2 - p1
            # Normal is perpendicular to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            # Ensure normal points outward
            # Midpoint of edge
            mid = (p1 + p2) / 2
            # Vector from center to midpoint
            center = np.mean(verts, axis=0)
            if np.dot(normal, center - mid) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    if min_overlap == float('inf'):
        return 0.0
    
    return min_overlap

def _containment_violation(verts, outer_center, outer_side, outer_angle):
    """
    Compute the maximum violation of the vertices being inside the outer hexagon.
    Returns 0 if all vertices are inside, otherwise the max distance outside.
    """
    cx, cy = outer_center
    a = np.radians(outer_angle)
    apothem = outer_side * np.sqrt(3) / 2.0
    # Normals for the outer hexagon boundaries
    normal_angles = a + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    
    max_viol = 0.0
    for v in verts:
        dx = v[0] - cx
        dy = v[1] - cy
        for na in normal_angles:
            proj = dx * np.cos(na) + dy * np.sin(na)
            if proj > apothem:
                viol = proj - apothem
                if viol > max_viol:
                    max_viol = viol
                # No need to check other normals for this vertex if we just want max viol?
                # Actually, a vertex can stick out of multiple corners, but max proj is the one.
                break
    return max_viol

def _objective(params, outer_side):
    """
    Objective function for the optimizer.
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, outer_angle]
    Returns sum of squared penalties.
    """
    # Unpack
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append((x, y, a))
    
    outer_angle = params[N*3]
    outer_center = (0.0, 0.0)
    
    penalty = 0.0
    
    # 1. Disjointness Penalty
    for i in range(N):
        v1 = _hex_verts(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, N):
            v2 = _hex_verts(inner[j][0], inner[j][1], 1.0, inner[j][2])
            depth = _sat_overlap_depth(v1, v2)
            if depth > 0:
                penalty += depth * depth
    
    # 2. Containment Penalty
    for i in range(N):
        v = _hex_verts(inner[i][0], inner[i][1], 1.0, inner[i][2])
        viol = _containment_violation(v, outer_center, outer_side, outer_angle)
        if viol > 0:
            penalty += viol * viol
            
    return penalty

def _try_pack(outer_side, time_budget=5.0):
    """
    Try to find a valid packing for the given outer_side.
    Returns (success, best_params, best_penalty).
    """
    start_time = time.time()
    best_params = None
    best_penalty = float('inf')
    
    np.random.seed(42) # Fixed seed for reproducibility within this call? 
                       # Actually, we want different restarts.
    
    # We will do multiple random restarts
    # Initial guess: Hexagonal grid
    # 3 rows of 4, or 4 rows of 3
    # Let's try a few structured initializations
    
    def get_initial_config():
        # Random scatter
        config = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = np.random.uniform(0, 60)
            config.extend([x, y, a])
        config.append(np.random.uniform(0, 60)) # Outer angle
        return np.array(config)
        
    def get_grid_config(rows, cols, spacing_x, spacing_y):
        config = []
        count = 0
        for r in range(rows):
            for c in range(cols):
                if count >= N:
                    break
                x = c * spacing_x + (r % 2) * spacing_x / 2
                y = r * spacing_y
                # Center the grid roughly
                x -= (cols-1) * spacing_x / 2
                y -= (rows-1) * spacing_y / 2
                config.extend([x, y, 0.0])
                count += 1
        # Pad if less than N (should not happen)
        while len(config) < N*3:
             config.extend([np.random.uniform(-3,3), np.random.uniform(-3,3), 0.0])
        config.append(0.0) # Outer angle
        return np.array(config[:N*3+1])

    initial_configs = [
        get_grid_config(3, 4, 2.1, 1.8),
        get_grid_config(4, 3, 2.1, 1.8),
        get_grid_config(2, 6, 2.1, 1.8),
        get_initial_config(),
        get_initial_config()
    ]
    
    for init in initial_configs:
        if time.time() - start_time > time_budget:
            break
            
        try:
            res = minimize(
                _objective, 
                init, 
                args=(outer_side,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if res.fun < best_penalty:
                best_penalty = res.fun
                best_params = res.x
                
                # If penalty is effectively zero, we found a solution
                if best_penalty < 1e-5:
                    return True, best_params, best_penalty
                    
        except Exception:
            continue
            
    return False, best_params, best_penalty

def construct():
    start_time = time.time()
    
    # Binary search for the minimum side length
    low = 3.5
    high = 6.0
    
    best_params = None
    best_side = high
    
    # We will run for a fixed number of iterations or until time runs out
    # Let's do 15 iterations of binary search
    # Each iteration tries to pack. If it fails, we might need to be more careful.
    
    # First, ensure we have a valid packing at 'high'
    # If not, increase high
    success, params, pen = _try_pack(high, time_budget=10.0)
    if not success or pen > 1e-4:
         # Try larger
         high = 7.0
         success, params, pen = _try_pack(high, time_budget=10.0)
         if not success or pen > 1e-4:
             # Fallback: return a very large hexagon with a valid grid
             # Grid 3x4 with spacing 2.1, 1.8
             # Width approx 3*2.1 + 1.05 = 7.35
             # Height approx 2*1.8 + 1.73 = 5.33
             # Side length needs to cover this.
             # Apothem = S * sqrt(3)/2.
             # For width W, we need S such that 2*S >= W? No.
             # Let's just return a safe large value.
             inner = []
             for r in range(3):
                 for c in range(4):
                     x = c * 2.1 + (r % 2) * 1.05
                     y = r * 1.8
                     x -= 3 * 2.1 / 2
                     y -= 2 * 1.8 / 2
                     inner.append([x, y, 0.0])
             return inner, (0.0, 0.0), 8.0, 0.0

    best_params = params
    best_side = high
    
    # Binary Search
    # We have 145s total. Spent ~20s so far.
    # Remaining ~120s.
    # Let's do 10 steps. 12s per step.
    
    for i in range(10):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        mid = (low + high) / 2.0
        success, params, pen = _try_pack(mid, time_budget=10.0)
        
        if success and pen < 1e-5:
            # Feasible
            high = mid
            best_side = mid
            best_params = params
        else:
            # Infeasible
            low = mid
            
    # Extract final result from best_params
    # best_params: [x1, y1, a1, ..., x12, y12, a12, outer_angle]
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
        
    outer_angle = best_params[N*3]
    
    return inner, (0.0, 0.0), best_side, outer_angle

if __name__ == "__main__":
    # Simple test
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}")
