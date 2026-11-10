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
    
    def compute_entropies(R, C):
        """
        Compute D and M from normalized marginals R, C (each sums to 1).
        Returns (D, M) where D = H_{-1}, M = max(H_0, H_1, H_2)
        """
        # For H_0 (x): distribution of X is just C
        p_x = C
        H_0 = -np.sum(p_x[p_x > 1e-15] * np.log2(p_x[p_x > 1e-15]))
        
        # For H_1 (x+y): 
        # p_{x+y}[v] = sum_{y} R[y]*C[v-y] where x=v-y in [0,9]
        p_xy = np.zeros(2 * n - 1)  # v ranges from 0 to 18
        for v in range(2 * n - 1):
            y_min = max(0, v - (n - 1))
            y_max = min(n - 1, v)
            y_vals = np.arange(y_min, y_max + 1)
            x_vals = v - y_vals
            p_xy[v] = np.sum(R[y_vals] * C[x_vals])
        H_1 = -np.sum(p_xy[p_xy > 1e-15] * np.log2(p_xy[p_xy > 1e-15]))
        
        # For H_2 (x+2y):
        # v = x + 2y, x in [0,9], y in [0,9]
        p_x2y = np.zeros(28)
        for v in range(28):
            y_min = max(0, int(np.ceil((v - (n - 1)) / 2.0)))
            y_max = min(n - 1, int(np.floor(v / 2.0)))
            if y_min > y_max:
                continue
            y_vals = np.arange(y_min, y_max + 1)
            x_vals = v - 2 * y_vals
            valid = (x_vals >= 0) & (x_vals <= n - 1)
            if np.any(valid):
                p_x2y[v] = np.sum(R[y_vals[valid]] * C[x_vals[valid]])
        H_2 = -np.sum(p_x2y[p_x2y > 1e-15] * np.log2(p_x2y[p_x2y > 1e-15]))
        
        # For H_{-1} (x - y):
        # v = x - y, x in [0,9], y in [0,9]
        p_xmy = np.zeros(2 * n - 1)  # index 0..18, representing v = -9..9
        for v in range(-(n - 1), n):
            idx = v + (n - 1)
            y_min = max(0, -v)
            y_max = min(n - 1, n - 1 - v)
            if y_min > y_max:
                continue
            y_vals = np.arange(y_min, y_max + 1)
            x_vals = v + y_vals
            p_xmy[idx] = np.sum(R[y_vals] * C[x_vals])
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
        
        # Softmax to get positive marginals, then normalize
        R = np.exp(log_R - np.max(log_R))
        C = np.exp(log_C - np.max(log_C))
        R = R / np.sum(R)
        C = C / np.sum(C)
        
        D, M = compute_entropies(R, C)
        
        if M < 1e-9:
            return 1e10  # penalty
        
        return -D / M
    
    # Use differential evolution for global optimization with more iterations
    bounds = [(-8, 8)] * (2 * n)
    
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=300,
        tol=1e-10,
        polish=True,
        popsize=20
    )
    
    best_params = result.x
    best_val = result.fun
    
    # Try multiple heuristic starting points with local optimization
    starting_points = [
        np.zeros(2 * n),
        np.concatenate([np.linspace(-3, 3, n), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.linspace(-3, 3, n)]),
        np.concatenate([np.linspace(3, -3, n), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.linspace(3, -3, n)]),
        np.concatenate([np.sin(np.linspace(0, 2*np.pi, n)) * 2, np.zeros(n)]),
        np.concatenate([np.zeros(n), np.sin(np.linspace(0, 2*np.pi, n)) * 2]),
        np.concatenate([np.cos(np.linspace(0, np.pi, n)) * 2, np.zeros(n)]),
        np.concatenate([np.zeros(n), np.cos(np.linspace(0, np.pi, n)) * 2]),
        # Diagonal concentration patterns
        np.concatenate([np.array([3, 2, 1, 0, 0, 0, 0, 1, 2, 3]), np.zeros(n)]),
        np.concatenate([np.zeros(n), np.array([3, 2, 1, 0, 0, 0, 0, 1, 2, 3])]),
        # Anti-diagonal patterns
        np.concatenate([np.array([3, 2, 1, 0, 0, 0, 0, 1, 2, 3])[::-1], np.zeros(n)]),
        np.concatenate([np.zeros(n), np.array([3, 2, 1, 0, 0, 0, 0, 1, 2, 3])[::-1]]),
    ]
    
    for sp in starting_points:
        res = minimize(
            objective,
            sp,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    # Additional local refinement from the best DE solution
    res = minimize(
        objective,
        best_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-14}
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
