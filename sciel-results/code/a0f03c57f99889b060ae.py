# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximize D / M where D = H_{-1} (entropy of x - y) and M = max(H_0, H_1, H_2).
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    n = 10
    
    def compute_entropies(P):
        """Compute D and M from distribution P[y, x]."""
        # P is shape (10, 10), P[y, x]
        # Projection for slope r: value = x + r*y
        # We need to pool cells with equal value
        
        def shannon_entropy(values, probs):
            """Compute Shannon entropy in bits, ignoring zero probabilities."""
            # Pool by value
            unique_vals, inverse = np.unique(values, return_inverse=True)
            pooled_probs = np.zeros(len(unique_vals))
            for i, idx in enumerate(inverse):
                pooled_probs[idx] += probs[i]
            
            # Filter out zero probabilities
            nonzero_mask = pooled_probs > 1e-15
            if not np.any(nonzero_mask):
                return 0.0
            
            p = pooled_probs[nonzero_mask]
            # Normalize (should already sum to 1, but be safe)
            p_sum = p.sum()
            if p_sum <= 1e-15:
                return 0.0
            p = p / p_sum
            
            return -np.sum(p * np.log2(p))
        
        # Create index arrays
        y_idx, x_idx = np.indices((n, n))
        
        # Flatten
        P_flat = P.flatten()
        x_flat = x_idx.flatten()
        y_flat = y_idx.flatten()
        
        # D = H_{-1}: distribution of x - y
        vals_neg1 = x_flat - y_flat
        D = shannon_entropy(vals_neg1, P_flat)
        
        # H_0: distribution of x
        vals_0 = x_flat
        H_0 = shannon_entropy(vals_0, P_flat)
        
        # H_1: distribution of x + y
        vals_1 = x_flat + y_flat
        H_1 = shannon_entropy(vals_1, P_flat)
        
        # H_2: distribution of x + 2y
        vals_2 = x_flat + 2 * y_flat
        H_2 = shannon_entropy(vals_2, P_flat)
        
        M = max(H_0, H_1, H_2)
        
        return D, M
    
    def objective(params):
        """Objective function to maximize D/M. Returns -D/M for minimization."""
        # params are log-probabilities (logistic transformation)
        P = np.exp(params)
        P_sum = P.sum()
        if P_sum <= 1e-15:
            return 1e10  # Invalid
        P = P / P_sum
        
        D, M = compute_entropies(P)
        
        if M <= 1e-9:
            return 1e10  # Invalid
        
        return -D / M
    
    def objective_and_grad(params):
        """Objective and gradient (numerical)."""
        f = objective(params)
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = objective(params_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return f, grad
    
    # Initial distribution: uniform
    P0 = np.ones((n, n), dtype=np.float64) / (n * n)
    params0 = np.log(P0)
    
    best_params = params0.copy()
    best_score = -compute_entropies(P0)[0] / compute_entropies(P0)[1]
    
    # Try multiple random initializations and optimize
    np.random.seed(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            params = params0.copy()
        else:
            # Perturb initial parameters
            perturbation = np.random.normal(0, 0.5, size=(n, n))
            params = params0 + perturbation
        
        # Use L-BFGS-B optimization
        try:
            result = opt.minimize(
                objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Convert back to probability distribution
    P_best = np.exp(best_params)
    P_best = np.clip(P_best, 0, None)
    P_sum = P_best.sum()
    if P_sum > 1e-15:
        P_best = P_best / P_sum
    else:
        P_best = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Ensure validity
    P_best = np.clip(P_best, 0, None)
    P_best = P_best / P_best.sum()
    
    return P_best


# EVOLVE-BLOCK-END
