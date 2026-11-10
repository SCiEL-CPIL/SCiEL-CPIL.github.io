# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization using rank-1 (product of marginals)
    parameterization P[y,x] = R[y] * C[x].
    
    Uses differential evolution for global search over the 20-dimensional marginal space,
    followed by a local refinement step. This avoids getting stuck in local optima that
    plague gradient-based methods on this non-convex landscape.
    """
    n = 10
    deadline = time.time() + 590  # Safety margin before 630s limit
    
    # Precompute grid indices
    x_idx = np.arange(n)
    y_idx = np.arange(n)
    
    # Precompute projection values for each slope as integer arrays
    # P[y, x] -> value = x + r*y
    proj_minus1 = (x_idx[None, :] - y_idx[:, None]).astype(int)  # x - y
    proj_0 = x_idx[None, :].astype(int)                          # x
    proj_1 = (x_idx[None, :] + y_idx[:, None]).astype(int)       # x + y
    proj_2 = (x_idx[None, :] + 2 * y_idx[:, None]).astype(int)   # x + 2y
    
    # Precompute unique values and mapping for fast pooling
    # For each projection, we need to sum P[y,x] over cells with same value
    # We'll precompute index lists for each unique value
    
    def build_pool_indices(proj_vals):
        """Return dict mapping value -> list of (y, x) indices."""
        pools = {}
        for y in range(n):
            for x in range(n):
                v = proj_vals[y, x]
                if v not in pools:
                    pools[v] = []
                pools[v].append((y, x))
        return pools
    
    pool_m1 = build_pool_indices(proj_minus1)
    pool_0 = build_pool_indices(proj_0)
    pool_1 = build_pool_indices(proj_1)
    pool_2 = build_pool_indices(proj_2)
    
    # Precompute flattened index arrays for each pool for vectorized sum
    def pools_to_arrays(pools):
        """Convert pool dict to list of (value, flat_indices_array)."""
        result = []
        for v in sorted(pools.keys()):
            idxs = np.array([y * n + x for y, x in pools[v]], dtype=np.int32)
            result.append((v, idxs))
        return result
    
    pool_m1_arr = pools_to_arrays(pool_m1)
    pool_0_arr = pools_to_arrays(pool_0)
    pool_1_arr = pools_to_arrays(pool_1)
    pool_2_arr = pools_to_arrays(pool_2)
    
    def compute_ratio_from_marginals(r, c):
        """
        Compute D/M ratio from normalized row and column marginals.
        r: shape (10,), c: shape (10,), both sum to 1.
        P[y,x] = r[y] * c[x]
        """
        P_flat = (r[:, None] * c[None, :]).flatten()
        
        def entropy_from_pools(pool_arr, P_flat):
            probs = np.array([np.sum(P_flat[idxs]) for _, idxs in pool_arr])
            probs = probs[probs > 1e-15]
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        
        D = entropy_from_pools(pool_m1_arr, P_flat)
        H0 = entropy_from_pools(pool_0_arr, P_flat)
        H1 = entropy_from_pools(pool_1_arr, P_flat)
        H2 = entropy_from_pools(pool_2_arr, P_flat)
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return -1e10
        return D / M
    
    def objective(params):
        """Objective to minimize: negative of D/M ratio."""
        log_r = params[:n]
        log_c = params[n:]
        
        # Convert to probabilities via softmax
        r = np.exp(log_r - np.max(log_r))
        c = np.exp(log_c - np.max(log_c))
        r = r / np.sum(r)
        c = c / np.sum(c)
        
        ratio = compute_ratio_from_marginals(r, c)
        if ratio < -1e9:
            return 1e10
        return -ratio
    
    # Define bounds for log-marginals (log-space, so exp gives positive values)
    # We bound them to avoid extreme values that cause numerical issues
    bounds = [(-5, 5)] * (2 * n)
    
    # Initial population: include uniform and some structured starts
    # Uniform: log_r = log_c = 0
    uniform_init = np.zeros(2 * n)
    
    # Differential evolution for global search
    best_ratio = -np.inf
    best_params = uniform_init.copy()
    
    # First evaluate uniform
    ratio_uniform = compute_ratio_from_marginals(np.ones(n)/n, np.ones(n)/n)
    best_ratio = ratio_uniform
    
    # Run differential evolution
    # We use a moderate population size and allow multiple generations
    # but check time budget
    start_time = time.time()
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-10,
            mut=(0.5, 1.0),
            recomb=0.7,
            polish=False,  # We'll do our own local refinement
            callback=None,
            update_interval=1
        )
        
        if result.fun < -best_ratio:
            best_ratio = -result.fun
            best_params = result.x
            
    except Exception as e:
        # If DE fails, fall back to random search
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # Not enough time for refinement
        pass
    else:
        # Local refinement using Nelder-Mead or L-BFGS-B from best DE result
        from scipy.optimize import minimize
        
        # Try a few local refinements from best and nearby points
        rng = np.random.default_rng(123)
        local_candidates = [best_params.copy()]
        
        # Add a few random perturbations
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            perturbed = best_params + rng.normal(0, 0.5, 2 * n)
            local_candidates.append(perturbed)
        
        for cand in local_candidates:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    objective,
                    cand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < -best_ratio:
                    best_ratio = -res.fun
                    best_params = res.x
            except:
                pass
    
    # Extract best solution
    log_r_best = best_params[:n]
    log_c_best = best_params[n:]
    
    r_best = np.exp(log_r_best - np.max(log_r_best))
    c_best = np.exp(log_c_best - np.max(log_c_best))
    r_best = r_best / np.sum(r_best)
    c_best = c_best / np.sum(c_best)
    
    P = np.outer(r_best, c_best)
    
    # Ensure validity
    P = np.clip(P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        # Fallback to uniform
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / total
    
    return P


# EVOLVE-BLOCK-END
