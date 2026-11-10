# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    We model P[y, x] = R[y] * C[x] (product of independent row and column marginals).
    This reduces the search space from 100 dimensions to 20 (10 rows + 10 cols).
    
    We maximize D / M where:
    - D = H_{-1}: Shannon entropy (bits) of distribution of x - y
    - M = max(H_0, H_1, H_2): max Shannon entropy of x, x+y, x+2y
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative, summing to 1.
    """
    n = 10
    
    # Precompute indices
    y_idx = np.arange(n)[:, None]  # shape (10, 1)
    x_idx = np.arange(n)[None, :]  # shape (1, 10)
    
    # Values for each slope
    # H_{-1}: x - y
    vals_neg1 = x_idx - y_idx  # shape (10, 10)
    # H_0: x
    vals_0 = x_idx  # shape (10, 10)
    # H_1: x + y
    vals_1 = x_idx + y_idx  # shape (10, 10)
    # H_2: x + 2y
    vals_2 = x_idx + 2 * y_idx  # shape (10, 10)
    
    # Precompute the bin indices for each slope to speed up projection
    # We map each cell (y, x) to a bin index for each slope.
    # H_{-1}: v = x - y in [-9, 9]. Index = v + 9.
    bins_neg1 = (vals_neg1 + n - 1).astype(int)
    
    # H_0: v = x in [0, 9]. Index = v.
    bins_0 = vals_0.astype(int)
    
    # H_1: v = x + y in [0, 18]. Index = v.
    bins_1 = vals_1.astype(int)
    
    # H_2: v = x + 2y in [0, 27]. Index = v.
    bins_2 = vals_2.astype(int)
    
    # Number of bins
    n_bins_neg1 = 2 * n - 1
    n_bins_0 = n
    n_bins_1 = 2 * n - 1
    n_bins_2 = 2 * n - 1
    
    def compute_entropies_fast(log_R, log_C):
        """
        Compute D and M from log-marginals using vectorized binning.
        log_R: shape (10,), log_C: shape (10,)
        Returns (D, M) where D = H_{-1}, M = max(H_0, H_1, H_2)
        """
        R = np.exp(log_R - np.max(log_R))
        C = np.exp(log_C - np.max(log_C))
        
        R_norm = R / np.sum(R)
        C_norm = C / np.sum(C)
        
        P = np.outer(R_norm, C_norm)
        
        # Helper to compute entropy from a bin index array and probability matrix
        def get_entropy(bins, p_matrix):
            # bins shape (10, 10), p_matrix shape (10, 10)
            # Flatten
            bins_flat = bins.flatten()
            p_flat = p_matrix.flatten()
            
            # Use np.bincount to sum probabilities into bins
            # We need to know the number of bins.
            # For safety, we pass the max bin + 1.
            n_bins = bins_flat.max() + 1
            bin_probs = np.bincount(bins_flat, weights=p_flat, minlength=n_bins)
            
            # Filter out zeros to avoid log(0)
            mask = bin_probs > 1e-15
            if not np.any(mask):
                return 0.0
            p_nonzero = bin_probs[mask]
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        D = get_entropy(bins_neg1, P)
        H_0 = get_entropy(bins_0, P)
        H_1 = get_entropy(bins_1, P)
        H_2 = get_entropy(bins_2, P)
        
        M = max(H_0, H_1, H_2)
        return D, M

    def objective(params):
        """
        Minimize negative ratio D/M.
        params: 20-dim vector, first 10 are log_R, next 10 are log_C
        """
        log_R = params[:n]
        log_C = params[n:]
        
        D, M = compute_entropies_fast(log_R, log_C)
        
        if M < 1e-9:
            return 1e10  # penalty
        
        return -D / M
    
    # Initial guess: uniform distribution
    log_R0 = np.zeros(n)
    log_C0 = np.zeros(n)
    x0 = np.concatenate([log_R0, log_C0])
    
    # Use differential evolution for global optimization
    # Larger population and more iterations for better exploration
    bounds = [(-6, 6)] * (2 * n)
    
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=200,
        tol=1e-10,
        popsize=30,
        polish=False, # We will do our own polish
        init='sobol' # Better initial sampling
    )
    
    best_params = result.x
    best_val = result.fun
    
    # Local refinement from the global best
    res_local = minimize(
        objective,
        best_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
    )
    if res_local.fun < best_val:
        best_val = res_local.fun
        best_params = res_local.x
        
    # Try several heuristic starting points to escape local minima
    # These points are designed to create asymmetries that might favor D over M
    starting_points = [
        x0.copy(),
        # Diagonal concentration
        np.concatenate([np.linspace(-3, 3, n), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.linspace(-3, 3, n)]),
        # Anti-diagonal
        np.concatenate([np.linspace(3, -3, n), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.linspace(3, -3, n)]),
        # Corners
        np.concatenate([np.array([3, 2, 1, 0, -1, -2, -3, -3, -2, -1]), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.array([3, 2, 1, 0, -1, -2, -3, -3, -2, -1])]),
        # Random perturbations from uniform
        x0 + np.random.normal(0, 1, 2*n),
        x0 + np.random.normal(0, 1, 2*n),
        x0 + np.random.normal(0, 1, 2*n),
    ]
    
    for sp in starting_points:
        res = minimize(
            objective,
            sp,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    # Extract optimal marginals
    log_R_opt = best_params[:n]
    log_C_opt = best_params[n:]
    
    R = np.exp(log_R_opt - np.max(log_R_opt))
    C = np.exp(log_C_opt - np.max(log_C_opt))
    R = R / np.sum(R)
    C = C / np.sum(C)
    
    P = np.outer(R, C)
    
    # Ensure valid: clip negatives, normalize
    P = np.clip(P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        # Fallback to uniform
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / total
    
    return P

# EVOLVE-BLOCK-END
