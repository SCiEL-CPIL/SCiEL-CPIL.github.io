# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the ratio D / M where:
    D = H_{-1} (entropy of distribution of x - y)
    M = max(H_0, H_1, H_2) (max entropy of x, x+y, x+2y)
    
    P is a 10x10 array P[y, x] for x, y in 0..9.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds budget, keep safety margin
    
    n = 10
    # Grid indices
    # P[y, x] -> row y, col x
    ys = np.arange(n)
    xs = np.arange(n)
    Y, X = np.meshgrid(ys, xs, indexing='ij')  # Y[y, x] = y, X[y, x] = x
    
    # Projection values
    D_vals = X - Y        # x - y
    H0_vals = X           # x
    H1_vals = X + Y       # x + y
    H2_vals = X + 2 * Y   # x + 2y
    
    # Flatten for bincount
    D_flat = D_vals.flatten()
    H0_flat = H0_vals.flatten()
    H1_flat = H1_vals.flatten()
    H2_flat = H2_vals.flatten()
    
    # Unique values and inverse indices
    D_uniq, D_inv = np.unique(D_flat, return_inverse=True)
    H0_uniq, H0_inv = np.unique(H0_flat, return_inverse=True)
    H1_uniq, H1_inv = np.unique(H1_flat, return_inverse=True)
    H2_uniq, H2_inv = np.unique(H2_flat, return_inverse=True)
    
    D_inv_flat = D_inv.flatten()
    H0_inv_flat = H0_inv.flatten()
    H1_inv_flat = H1_inv.flatten()
    H2_inv_flat = H2_inv.flatten()
    
    def compute_entropies(P_flat):
        """Compute D and M from a flat probability vector."""
        # D: entropy of x - y
        D_probs = np.bincount(D_inv_flat, weights=P_flat, minlength=len(D_uniq))
        D_probs = D_probs[D_probs > 1e-15]
        H_D = -np.sum(D_probs * np.log2(D_probs)) if len(D_probs) > 0 else 0.0
        
        # H0: entropy of x
        H0_probs = np.bincount(H0_inv_flat, weights=P_flat, minlength=len(H0_uniq))
        H0_probs = H0_probs[H0_probs > 1e-15]
        H_0 = -np.sum(H0_probs * np.log2(H0_probs)) if len(H0_probs) > 0 else 0.0
        
        # H1: entropy of x + y
        H1_probs = np.bincount(H1_inv_flat, weights=P_flat, minlength=len(H1_uniq))
        H1_probs = H1_probs[H1_probs > 1e-15]
        H_1 = -np.sum(H1_probs * np.log2(H1_probs)) if len(H1_probs) > 0 else 0.0
        
        # H2: entropy of x + 2y
        H2_probs = np.bincount(H2_inv_flat, weights=P_flat, minlength=len(H2_uniq))
        H2_probs = H2_probs[H2_probs > 1e-15]
        H_2 = -np.sum(H2_probs * np.log2(H2_probs)) if len(H2_probs) > 0 else 0.0
        
        M = max(H_0, H_1, H_2)
        return H_D, M
    
    def objective(params):
        """Minimize negative log-ratio: -log(D/M) = log(M) - log(D)"""
        P = np.exp(params)
        P /= np.sum(P)
        P_flat = P.flatten()
        
        H_D, M = compute_entropies(P_flat)
        
        if H_D <= 1e-15 or M <= 1e-15:
            return 1e10
        return np.log(M) - np.log(H_D)
    
    def objective_and_grad(params):
        """Objective and gradient via finite differences."""
        val = objective(params)
        if val > 1e9:
            return val, np.zeros_like(params)
        
        # Numerical gradient
        grad = np.zeros_like(params)
        eps = 1e-7
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            val_plus = objective(params_plus)
            grad[i] = (val_plus - val) / eps
            params_plus[i] -= 2 * eps
            val_minus = objective(params_plus)
            params_plus[i] += eps
            grad[i] = (val_plus - val_minus) / (2 * eps)
            params_plus[i] -= eps
        return val, grad
    
    # Start from uniform distribution
    P_init = np.ones((n, n), dtype=np.float64) / (n * n)
    P_init_flat = P_init.flatten()
    params_init = np.log(P_init_flat)
    
    best_P = P_init.copy()
    best_val = objective(params_init)
    
    # Try multiple optimization approaches
    # 1. L-BFGS-B with gradient
    try:
        result = opt.minimize(
            objective_and_grad,
            params_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < best_val:
            P_opt = np.exp(result.x)
            P_opt /= np.sum(P_opt)
            val_opt = objective(result.x)
            if val_opt < best_val:
                best_val = val_opt
                best_P = P_opt.reshape(n, n)
    except Exception:
        pass
    
    if time.time() > deadline - 60:
        return best_P
    
    # 2. Try Nelder-Mead without gradient (more robust)
    try:
        result = opt.minimize(
            objective,
            params_init,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_val:
            P_opt = np.exp(result.x)
            P_opt /= np.sum(P_opt)
            val_opt = objective(result.x)
            if val_opt < best_val:
                best_val = val_opt
                best_P = P_opt.reshape(n, n)
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_P
    
    # 3. Try Powell method
    try:
        result = opt.minimize(
            objective,
            params_init,
            method='Powell',
            options={'maxiter': 3000, 'ftol': 1e-12, 'xtol': 1e-10}
        )
        if result.fun < best_val:
            P_opt = np.exp(result.x)
            P_opt /= np.sum(P_opt)
            val_opt = objective(result.x)
            if val_opt < best_val:
                best_val = val_opt
                best_P = P_opt.reshape(n, n)
    except Exception:
        pass
    
    # 4. Try random restarts with local optimization
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        P_rand = np.random.dirichlet(np.ones(n * n))
        params_rand = np.log(P_rand)
        
        try:
            result = opt.minimize(
                objective,
                params_rand,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12}
            )
            if result.fun < best_val:
                P_opt = np.exp(result.x)
                P_opt /= np.sum(P_opt)
                val_opt = objective(result.x)
                if val_opt < best_val:
                    best_val = val_opt
                    best_P = P_opt.reshape(n, n)
        except Exception:
            pass
    
    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total > 1e-9:
        best_P /= total
    else:
        best_P = np.ones((n, n)) / (n * n)
    
    return best_P


# EVOLVE-BLOCK-END
