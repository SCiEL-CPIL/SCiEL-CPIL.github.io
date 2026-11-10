# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


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
    
    # For product distribution P[y,x] = R[y]*C[x], we can compute the projected distributions
    # analytically from R and C.
    
    def compute_entropies(log_R, log_C):
        """
        Compute D and M from log-marginals.
        log_R: shape (10,), log_C: shape (10,)
        Returns (D, M) where D = H_{-1}, M = max(H_0, H_1, H_2)
        """
        R = np.exp(log_R - np.max(log_R))  # avoid overflow
        C = np.exp(log_C - np.max(log_C))
        
        # Normalize so that sum(R)*sum(C) = 1
        # Actually, P[y,x] = R[y]*C[x] must sum to 1.
        # Let's work with normalized R and C such that sum(R)=1, sum(C)=1
        R_norm = R / np.sum(R)
        C_norm = C / np.sum(C)
        
        # P[y,x] = R_norm[y] * C_norm[x]
        # Now compute projected distributions.
        
        # For H_0 (x): distribution of X is just C_norm
        p_x = C_norm
        H_0 = -np.sum(p_x[p_x > 1e-15] * np.log2(p_x[p_x > 1e-15]))
        
        # For H_1 (x+y): 
        # p_{x+y}[v] = sum_{y: x+y=v} R[y]*C[x] = sum_{y} R[y]*C[v-y] where x=v-y in [0,9]
        # y in [0,9], x = v-y in [0,9] => y in [max(0, v-9), min(9, v)]
        p_xy = np.zeros(2 * n - 1)  # v ranges from 0 to 18
        for v in range(2 * n - 1):
            y_min = max(0, v - (n - 1))
            y_max = min(n - 1, v)
            y_vals = np.arange(y_min, y_max + 1)
            x_vals = v - y_vals
            p_xy[v] = np.sum(R_norm[y_vals] * C_norm[x_vals])
        H_1 = -np.sum(p_xy[p_xy > 1e-15] * np.log2(p_xy[p_xy > 1e-15]))
        
        # For H_2 (x+2y):
        # v = x + 2y, x in [0,9], y in [0,9]
        # x = v - 2y, so v - 2y in [0,9] => 2y in [v-9, v] => y in [ceil((v-9)/2), floor(v/2)]
        # Also y in [0,9]
        # v ranges from 0 to 9 + 18 = 27
        p_x2y = np.zeros(27 + 1)
        for v in range(28):
            y_min = max(0, int(np.ceil((v - (n - 1)) / 2.0)))
            y_max = min(n - 1, int(np.floor(v / 2.0)))
            if y_min > y_max:
                continue
            y_vals = np.arange(y_min, y_max + 1)
            x_vals = v - 2 * y_vals
            # Check x_vals are in [0, n-1]
            valid = (x_vals >= 0) & (x_vals <= n - 1)
            if np.any(valid):
                p_x2y[v] = np.sum(R_norm[y_vals[valid]] * C_norm[x_vals[valid]])
        H_2 = -np.sum(p_x2y[p_x2y > 1e-15] * np.log2(p_x2y[p_x2y > 1e-15]))
        
        # For H_{-1} (x - y):
        # v = x - y, x in [0,9], y in [0,9]
        # x = v + y, so v + y in [0,9] => y in [-v, 9-v] intersect [0,9]
        # v ranges from -9 to 9
        p_xmy = np.zeros(2 * n - 1)  # index 0..18, representing v = -9..9
        for v in range(-(n - 1), n):
            idx = v + (n - 1)
            y_min = max(0, -v)
            y_max = min(n - 1, n - 1 - v)
            if y_min > y_max:
                continue
            y_vals = np.arange(y_min, y_max + 1)
            x_vals = v + y_vals
            p_xmy[idx] = np.sum(R_norm[y_vals] * C_norm[x_vals])
        H_neg1 = -np.sum(p_xmy[p_xmy > 1e-15] * np.log2(p_xmy[p_xmy > 1e-15]))
        
        M = max(H_0, H_1, H_2)
        return H_neg1, M
    
    def objective(params):
        """
        Minimize negative ratio D/M.
        params: 20-dim vector, first 10 are log_R, next 10 are log_C
        """
        log_R = params[:n]
        log_C = params[n:]
        
        D, M = compute_entropies(log_R, log_C)
        
        if M < 1e-9:
            return 1e10  # penalty
        
        return -D / M
    
    # Initial guess: uniform distribution
    log_R0 = np.zeros(n)
    log_C0 = np.zeros(n)
    x0 = np.concatenate([log_R0, log_C0])
    
    # Use differential evolution for global optimization
    bounds = [(-5, 5)] * (2 * n)
    
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=True
    )
    
    # Also try a few more local optimizations from different starting points
    best_params = result.x
    best_val = result.fun
    
    # Try some heuristic starting points
    starting_points = [
        x0.copy(),
        np.concatenate([np.linspace(-2, 2, n), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.linspace(-2, 2, n)]),
        np.concatenate([np.linspace(2, -2, n), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.linspace(2, -2, n)]),
        np.concatenate([np.sin(np.linspace(0, 2*np.pi, n)), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.sin(np.linspace(0, 2*np.pi, n))]),
        np.concatenate([np.cos(np.linspace(0, np.pi, n)), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.cos(np.linspace(0, np.pi, n))]),
    ]
    
    for sp in starting_points:
        res = minimize(
            objective,
            sp,
            method='L-BFGS-B',
            bounds=[(-5, 5)] * (2 * n),
            options={'maxiter': 1000, 'ftol': 1e-12}
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
