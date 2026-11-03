"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _precompute_lp_matrices(n):
    """
    Precompute the A_ub matrix for the LP that maximizes sum(radii)
    given pairwise distances.
    
    Variables: r_0, ..., r_{n-1}
    Constraints:
      1. r_i + r_j <= d_ij  for all i < j
      2. r_i <= b_i (handled by bounds)
      3. r_i >= 0 (handled by bounds)
    
    We return A_ub (shape [n(n-1)/2, n]) and a helper to update b_ub.
    """
    num_constraints = n * (n - 1) // 2
    A_ub = np.zeros((num_constraints, n))
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            idx += 1
            
    c = -np.ones(n)  # Minimize -sum(r)
    
    return c, A_ub

def _compute_radii(pos, n, c, A_ub):
    """
    Compute optimal radii for given positions using LP.
    Returns (sum_radii, radii_array).
    """
    # Boundary distances: min(x, y, 1-x, 1-y)
    b = np.minimum(np.minimum(pos[:, 0], pos[:, 1]), 1.0 - np.minimum(pos[:, 0], pos[:, 1]))
    b = np.maximum(b, 0.0)  # Ensure non-negative upper bounds
    
    # Pairwise distances
    # Use broadcasting for speed
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Extract upper triangle distances for b_ub
    # The order must match A_ub construction: i=0,j=1; i=0,j=2...; i=1,j=2...
    b_ub = np.zeros(n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            b_ub[idx] = dist[i, j]
            idx += 1
            
    bounds = [(0, b[i]) for i in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            # Fallback if LP fails (should be rare with valid positions)
            # Greedy assignment
            radii = np.zeros(n)
            for _ in range(10):
                changed = False
                for i in range(n):
                    max_r = b[i]
                    for j in range(n):
                        if i != j:
                            max_r = min(max_r, dist[i, j] - radii[j])
                    if max_r > radii[i] + 1e-9:
                        radii[i] = max(0, max_r)
                        changed = True
                if not changed:
                    break
            return np.sum(radii), radii
    except Exception:
        # Last resort fallback
        return 0.0, np.zeros(n)

def _objective(params, n, c, A_ub):
    """
    Objective function for position optimization.
    params: flattened array of shape (2*n,)
    Returns negative sum of radii.
    """
    pos = params.reshape(n, 2)
    # Clip positions to [0,1] to ensure boundary distances are well-defined
    # L-BFGS-B handles bounds, but clipping inside obj prevents numerical issues
    # if the optimizer steps slightly out (though bounds should prevent this).
    pos = np.clip(pos, 0.0, 1.0)
    
    sum_r, _ = _compute_radii(pos, n, c, A_ub)
    return -sum_r

def _is_valid(pos, radii, n, tol=1e-5):
    """Check if packing is valid."""
    # Boundary
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    # Overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_and_validate(pos, radii, n, c, A_ub):
    """
    Take a candidate solution, re-solve LP to get exact max radii,
    then check validity.
    """
    # Re-solve LP to ensure radii are optimal for these positions
    sum_r, radii_opt = _compute_radii(pos, n, c, A_ub)
    
    # Check validity with the optimized radii
    if _is_valid(pos, radii_opt, n):
        return sum_r, pos, radii_opt
    else:
        # If invalid (numerical error), shrink radii slightly
        radii_safe = radii_opt * 0.999
        if _is_valid(pos, radii_safe, n):
            return np.sum(radii_safe), pos, radii_safe
        else:
            return -np.inf, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    c, A_ub = _precompute_lp_matrices(N)
    
    best_score = -np.inf
    best_result = None
    
    # Initial configurations to try
    configs = []
    
    # 1. Grid 5x6 (26 circles)
    rows, cols = 5, 6
    pos_grid = []
    for i in range(rows):
        for j in range(cols):
            if len(pos_grid) >= N: break
            pos_grid.append([(j + 0.5) / cols, (i + 0.5) / rows])
    configs.append(np.array(pos_grid))
    
    # 2. Random uniform
    rng = np.random.RandomState(42)
    configs.append(rng.uniform(0.1, 0.9, (N, 2)))
    
    # 3. Random with some structure (clustering)
    centers = rng.uniform(0.2, 0.8, (4, 2))
    pos_clustered = []
    for k in range(N):
        c_idx = k % 4
        pos_clustered.append(centers[c_idx] + rng.uniform(-0.1, 0.1, 2))
    configs.append(np.array(pos_clustered))
    
    # 4. Corners biased
    pos_corner = []
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for k in range(N):
        c_idx = k % 4
        pos_corner.append(corners[c_idx] + rng.uniform(-0.05, 0.05, 2))
    configs.append(np.array(pos_corner))
    
    bounds = [(0, 1)] * (2 * N)
    
    for i, pos0 in enumerate(configs):
        if time.time() > deadline - 20.0:
            break
            
        # Ensure initial positions are within bounds
        pos0 = np.clip(pos0, 0, 1)
        
        try:
            # Use L-BFGS-B for bound-constrained optimization
            res = minimize(
                _objective,
                pos0.flatten(),
                args=(N, c, A_ub),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.x is not None:
                pos_opt = res.x.reshape(N, 2)
                pos_opt = np.clip(pos_opt, 0, 1)
                
                score, pos_final, radii_final = _fix_and_validate(pos_opt, np.zeros(N), N, c, A_ub)
                
                if score > best_score:
                    best_score = score
                    best_result = np.zeros((N, 3))
                    best_result[:, :2] = pos_final
                    best_result[:, 2] = radii_final
                    
        except Exception:
            continue
            
    # If no optimization succeeded, use a simple fallback
    if best_result is None:
        # Fallback: Grid with small radii
        rows, cols = 5, 6
        pos_fallback = []
        for i in range(rows):
            for j in range(cols):
                if len(pos_fallback) >= N: break
                pos_fallback.append([(j + 0.5) / cols, (i + 0.5) / rows])
        pos_fallback = np.array(pos_fallback)
        # Calculate max possible radii for this grid
        _, radii_fallback = _compute_radii(pos_fallback, N, c, A_ub)
        # Ensure validity
        radii_fallback = radii_fallback * 0.999
        best_result = np.zeros((N, 3))
        best_result[:, :2] = pos_fallback
        best_result[:, 2] = radii_fallback
        best_score = np.sum(radii_fallback)
        
    return best_result

# EVOLVE-BLOCK-END
