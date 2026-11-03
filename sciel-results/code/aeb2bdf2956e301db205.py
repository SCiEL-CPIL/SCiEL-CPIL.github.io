import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3)

# Precompute triangle pairs
_i, _j = np.triu_indices(N, k=1)
_PAIRS_I = _i
_PAIRS_J = _j
_PAIRS_COUNT = len(_i)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates (should sum to 1).
    Returns the minimum triangle area (in absolute value, half the cross product).
    """
    # Convert barycentric to Cartesian: x = u1 + 0.5*u2, y = (sqrt(3)/2)*u2
    # For area of triangle (a, b, c) in barycentric coords:
    # Area = 0.5 * |det of matrix with rows (x_a, y_a, 1), (x_b, y_b, 1), (x_c, y_c, 1)|
    # But in barycentric coords for equilateral triangle, area = 0.5 * |det|
    # Actually, the area of a triangle with barycentric coords (u1,v1,w1), (u2,v2,w2), (u3,v3,w3)
    # is 0.5 * area_of_big_triangle * |det| of the matrix of barycentric coords.
    # Since area_of_big_triangle = sqrt(3)/4, and we're dividing by it, the normalized area
    # is 0.25 * |det| of barycentric matrix.
    # Actually: normalized_area = |det(bary_matrix)| / 2
    # Let me verify: for the three vertices (1,0,0), (0,1,0), (0,0,1), det = 1, area = 0.5*1 = 0.5.
    # The full triangle has normalized area 1.0. So 0.5 is correct for a triangle that is half the full triangle.
    # Yes, normalized area = 0.5 * |det| of barycentric coordinate matrix.
    
    # Precompute all pair differences for speed
    # For each triple (i,j,k), area = 0.5*|det([p_i, p_j, p_k])| where p are barycentric
    
    # Vectorized: compute all 165 determinants
    # Build the 3x3 matrices
    # For triple (i, j, k): det = p_i[0]*(p_j[1]*p_k[2] - p_j[2]*p_k[1]) - p_i[1]*(p_j[0]*p_k[2] - p_j[2]*p_k[0]) + p_i[2]*(p_j[0]*p_k[1] - p_j[1]*p_k[0])
    
    # Let's just loop over triples but vectorize where possible
    min_area = np.inf
    for i in range(N - 2):
        pi = p[i]
        for j in range(i + 1, N - 1):
            pj = p[j]
            # For all k > j
            ps_k = p[j+1:]
            # det = pi[0]*(pj[1]*ps_k[:,2] - pj[2]*ps_k[:,1]) 
            #     - pi[1]*(pj[0]*ps_k[:,2] - pj[2]*ps_k[:,0])
            #     + pi[2]*(pj[0]*ps_k[:,1] - pj[1]*ps_k[:,0])
            dets = (pi[0]*(pj[1]*ps_k[:,2] - pj[2]*ps_k[:,1])
                   - pi[1]*(pj[0]*ps_k[:,2] - pj[2]*ps_k[:,0])
                   + pi[2]*(pj[0]*ps_k[:,1] - pj[1]*ps_k[:,0]))
            areas = 0.5 * np.abs(dets)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y) for equilateral triangle."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """Project barycentric coordinates to be valid (non-negative, sum to 1)."""
    b = bary.copy()
    # Normalize
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    # Clip negative values
    b = np.clip(b, 0, 1)
    # Re-normalize
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_area = 0.0
    best_bary = None
    
    def _is_time_up():
        return time.time() > deadline - 5.0
    
    # Multiple restart strategies
    # Strategy 1: Random uniform in barycentric
    # Strategy 2: Grid-based
    # Strategy 3: Centroid + ring
    
    for restart in range(30):
        if _is_time_up():
            break
        
        # Initialize
        if restart == 0:
            # Regular grid-like
            bary = np.zeros((N, 3))
            # Place points in a structured way
            # 3 points near each vertex, 1 near center, 2 more
            for i in range(N):
                # Random but biased
                bary[i] = rng.dirichlet([1, 1, 1])
        elif restart == 1:
            # More structured
            bary = np.zeros((N, 3))
            # Corners
            bary[0] = [0.9, 0.05, 0.05]
            bary[1] = [0.05, 0.9, 0.05]
            bary[2] = [0.05, 0.05, 0.9]
            # Edge midpoints
            bary[3] = [0.45, 0.45, 0.1]
            bary[4] = [0.1, 0.45, 0.45]
            bary[5] = [0.45, 0.1, 0.45]
            # Center
            bary[6] = [1/3, 1/3, 1/3]
            # A few more
            for i in range(7, N):
                bary[i] = rng.dirichlet([1, 1, 1])
        else:
            # Random
            bary = rng.dirichlet([1, 1, 1], size=N)
        
        current_bary = _project_bary(bary)
        current_area = _min_area_bary(current_bary)
        
        if current_area > best_area:
            best_area = current_area
            best_bary = current_bary.copy()
        
        # Simulated Annealing in barycentric space
        T = 0.03
        T_min = 1e-7
        cooling = 0.997
        step_size = 0.08
        
        while T > T_min and not _is_time_up():
            # Perturb
            new_bary = current_bary.copy()
            num_perturb = rng.integers(1, 4)
            indices = rng.choice(N, size=num_perturb, replace=False)
            for idx in indices:
                new_bary[idx] += rng.normal(0, step_size, size=3)
            
            new_bary = _project_bary(new_bary)
            new_area = _min_area_bary(new_bary)
            
            if new_area > current_area:
                current_bary = new_bary
                current_area = new_area
                if current_area > best_area:
                    best_area = current_area
                    best_bary = current_bary.copy()
            else:
                delta = new_area - current_area
                if delta > -T * 5:  # Only compute exp if not too negative
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_bary = new_bary
                        current_area = new_area
            
            T *= cooling
            step_size *= 0.998
        
        # Local refinement with coordinate descent
        if not _is_time_up():
            for _ in range(50):
                if _is_time_up():
                    break
                improved = False
                for i in range(N):
                    for dim in range(3):
                        # Try small perturbations
                        for scale in [0.01, 0.005, 0.001]:
                            for direction in [-1, 1]:
                                trial = current_bary.copy()
                                trial[i, dim] += direction * scale
                                trial = _project_bary(trial)
                                trial_area = _min_area_bary(trial)
                                if trial_area > current_area:
                                    current_bary = trial
                                    current_area = trial_area
                                    improved = True
                                    if current_area > best_area:
                                        best_area = current_area
                                        best_bary = current_bary.copy()
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if not improved:
                    break
    
    if best_bary is None:
        # Fallback
        bary = np.zeros((N, 3))
        for i in range(N):
            bary[i] = [1/3, 1/3, 1/3] + rng.normal(0, 0.1, size=3)
        best_bary = _project_bary(bary)
    
    # Convert to Cartesian
    cart = _bary_to_cart(best_bary)
    return cart
