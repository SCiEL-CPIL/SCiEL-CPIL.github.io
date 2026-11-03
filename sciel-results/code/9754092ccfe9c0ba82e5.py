# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    # P = u*(0,0) + v*(1,0) + w*(0.5, sqrt(3)/2)
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _min_triangle_area_vec(points):
    """
    Vectorized calculation of the minimum triangle area.
    points: (N, 2) array
    """
    n = len(points)
    # Generate all unique triplets i < j < k
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    vec1 = p_j - p_i
    vec2 = p_k - p_i
    cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas) if areas.size > 0 else 0.0

def _objective_bary(x):
    """
    Objective function for barycentric coordinates.
    x: flat array of length 2*N, representing (u1, v1, u2, v2, ...)
    """
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Check constraints: u >= 0, v >= 0, u+v <= 1
    # Penalty for violations
    pen1 = np.maximum(0, -u)
    pen2 = np.maximum(0, -v)
    pen3 = np.maximum(0, u + v - 1.0)
    
    penalty = np.sum(pen1**2) + np.sum(pen2**2) + np.sum(pen3**2)
    
    if penalty > 1e-4:
        return penalty
    
    # Convert to Cartesian
    points = _bary_to_cart(u, v)
    
    # Calculate min area
    area = _min_triangle_area_vec(points)
    
    # We want to maximize area, so minimize -area
    # Normalize by triangle area? The evaluator measures raw value.
    # "The evaluator measures the raw value... maximize the smallest area... divided by its area"
    # The prompt says: "maximize the smallest area of a triangle formed by three points. The notebook uses an equilateral triangle and divides by its area."
    # But the interface says: "The evaluator measures the raw value (higher is better)."
    # And "search score is value/seed".
    # The current program returns points. The evaluator calculates the score.
    # I should maximize the raw area.
    return -area

def _make_structured_starts():
    """
    Generate several structured initial configurations in barycentric coordinates.
    """
    starts = []
    
    # 1. Random starts
    rng = np.random.RandomState(42)
    for _ in range(30):
        u = rng.random(N)
        v = rng.random(N)
        # Normalize to be inside triangle? No, just random in unit square and rely on penalty/optimization?
        # Better: sample uniformly in triangle.
        u_raw = rng.random(N)
        v_raw = rng.random(N)
        s = u_raw + v_raw
        s[s == 0] = 1.0
        u = u_raw / s
        v = v_raw / s
        starts.append(np.column_stack((u, v)).flatten())
        
    # 2. Grid-based starts
    # Try to place points on a coarse grid of barycentric coordinates
    # e.g., u, v in {0, 0.25, 0.5, 0.75, 1.0} subject to u+v<=1
    coords = [0.0, 0.25, 0.5, 0.75, 1.0]
    valid_points = []
    for u in coords:
        for v in coords:
            if u >= 0 and v >= 0 and u + v <= 1.0:
                valid_points.append((u, v))
                
    # Shuffle valid points and take N
    rng.shuffle(valid_points)
    if len(valid_points) >= N:
        starts.append(np.array(valid_points[:N]).flatten())
        
    # 3. Symmetric starts
    # Center and vertices
    center = (1/3, 1/3)
    v0 = (1.0, 0.0) # u=1, v=0 -> V0
    v1 = (0.0, 1.0) # u=0, v=1 -> V1
    v2 = (0.0, 0.0) # u=0, v=0 -> V2 (since w=1)
    
    # Wait, mapping:
    # P = u V0 + v V1 + (1-u-v) V2
    # If u=1, v=0 -> V0. Correct.
    # If u=0, v=1 -> V1. Correct.
    # If u=0, v=0 -> V2. Correct.
    
    sym_points = [center, v0, v1, v2]
    # Add midpoints
    mid01 = ((1+0)/2, (0+1)/2) # 0.5, 0.5 -> u+v=1, w=0. Midpoint of V0, V1.
    mid12 = ((0+0)/2, (1+0)/2) # 0, 0.5 -> Midpoint of V1, V2.
    mid20 = ((0+1)/2, (0+0)/2) # 0.5, 0 -> Midpoint of V2, V0.
    
    sym_points.extend([mid01, mid12, mid20])
    
    # Need 11 points. Add some random perturbations to the symmetric set?
    # Or just use the first N from a combined list.
    # Let's add more grid points to fill up to 11 if needed.
    all_starts = sym_points + valid_points
    if len(all_starts) >= N:
        starts.append(np.array(all_starts[:N]).flatten())
        
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_x = None
    best_score = -float('inf')
    
    inits = _make_structured_starts()
    
    # Ensure we have at least a few inits
    if not inits:
        u = np.ones(N) / N
        v = np.ones(N) / N
        inits.append(np.column_stack((u, v)).flatten())

    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Try Nelder-Mead
            res = minimize(
                _objective_bary,
                init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                x_opt = res.x.reshape(N, 2)
                # Check validity
                u = x_opt[:, 0]
                v = x_opt[:, 1]
                if np.all(u >= -1e-6) and np.all(v >= -1e-6) and np.all(u + v <= 1.0 + 1e-6):
                    # Clamp for final output
                    u_clamped = np.clip(u, 0, 1)
                    v_clamped = np.clip(v, 0, 1)
                    # Ensure u+v <= 1
                    excess = np.maximum(0, u_clamped + v_clamped - 1.0)
                    # Simple clamp: scale down if needed? Or just project?
                    # For evaluation, we just need points inside.
                    # If slightly out, we can project to nearest valid point.
                    # But let's just use the clamped version if it's close.
                    
                    points = _bary_to_cart(u_clamped, v_clamped)
                    score = _min_triangle_area_vec(points)
                    
                    if score > best_score:
                        best_score = score
                        best_x = points
                        
            # Try Powell as well if time permits
            if time.time() < deadline - 5.0:
                res_powell = minimize(
                    _objective_bary,
                    init,
                    method='Powell',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_powell.success or res_powell.fun < best_score:
                    x_opt = res_powell.x.reshape(N, 2)
                    u = x_opt[:, 0]
                    v = x_opt[:, 1]
                    if np.all(u >= -1e-6) and np.all(v >= -1e-6) and np.all(u + v <= 1.0 + 1e-6):
                        u_clamped = np.clip(u, 0, 1)
                        v_clamped = np.clip(v, 0, 1)
                        points = _bary_to_cart(u_clamped, v_clamped)
                        score = _min_triangle_area_vec(points)
                        if score > best_score:
                            best_score = score
                            best_x = points
                            
        except Exception:
            pass

    if best_x is None:
        # Fallback: Centroid
        best_x = np.zeros((N, 2))
        # Place one at centroid, others near it?
        # Better fallback: uniform grid
        u = np.linspace(0, 1, N)
        v = np.linspace(0, 1, N)
        # This is invalid (u+v>1).
        # Just return vertices repeated?
        # Let's use the first valid init if it exists
        if inits:
             x_opt = inits[0].reshape(N, 2)
             u = np.clip(x_opt[:, 0], 0, 1)
             v = np.clip(x_opt[:, 1], 0, 1)
             best_x = _bary_to_cart(u, v)
        else:
             best_x = np.array([[0.5, np.sqrt(3)/6]] * N)

    return best_x
# EVOLVE-BLOCK-END
