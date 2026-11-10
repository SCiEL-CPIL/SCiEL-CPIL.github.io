# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.special import softmax

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the ratio D / M where:
    D = H_{-1} (entropy of distribution of x - y)
    M = max(H_0, H_1, H_2) (max entropy of x, x+y, x+2y)
    
    P is a 10x10 array P[y, x] for x, y in 0..9.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    n = 10
    # Grid indices
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
    
    # Precompute one-hot matrices for gradient computation
    # Each row corresponds to a cell, each column to a bin
    D_onehot = np.zeros((n*n, len(D_uniq)))
    D_onehot[np.arange(n*n), D_inv_flat] = 1.0
    
    H0_onehot = np.zeros((n*n, len(H0_uniq)))
    H0_onehot[np.arange(n*n), H0_inv_flat] = 1.0
    
    H1_onehot = np.zeros((n*n, len(H1_uniq)))
    H1_onehot[np.arange(n*n), H1_inv_flat] = 1.0
    
    H2_onehot = np.zeros((n*n, len(H2_uniq)))
    H2_onehot[np.arange(n*n), H2_inv_flat] = 1.0

    def compute_entropies(P_flat):
        """Compute D and M from a flat probability vector."""
        # D: entropy of x - y
        D_probs = D_onehot.T @ P_flat
        D_probs = D_probs[D_probs > 1e-15]
        H_D = -np.sum(D_probs * np.log2(D_probs)) if len(D_probs) > 0 else 0.0
        
        # H0: entropy of x
        H0_probs = H0_onehot.T @ P_flat
        H0_probs = H0_probs[H0_probs > 1e-15]
        H_0 = -np.sum(H0_probs * np.log2(H0_probs)) if len(H0_probs) > 0 else 0.0
        
        # H1: entropy of x + y
        H1_probs = H1_onehot.T @ P_flat
        H1_probs = H1_probs[H1_probs > 1e-15]
        H_1 = -np.sum(H1_probs * np.log2(H1_probs)) if len(H1_probs) > 0 else 0.0
        
        # H2: entropy of x + 2y
        H2_probs = H2_onehot.T @ P_flat
        H2_probs = H2_probs[H2_probs > 1e-15]
        H_2 = -np.sum(H2_probs * np.log2(H2_probs)) if len(H2_probs) > 0 else 0.0
        
        M = max(H_0, H_1, H_2)
        return H_D, M, H_0, H_1, H_2

    def objective_and_grad(params, beta=20.0):
        """
        Objective: minimize log(M) - log(D)
        We use a soft-max for M to ensure differentiability.
        """
        # Shift to prevent overflow in exp
        params_shifted = params - np.max(params)
        P = np.exp(params_shifted)
        sum_P = np.sum(P)
        if sum_P <= 0:
            return 1e10, np.zeros_like(params)
        P /= sum_P
        P_flat = P.flatten()
        
        # Compute probabilities for each projection
        D_probs = D_onehot.T @ P_flat
        H0_probs = H0_onehot.T @ P_flat
        H1_probs = H1_onehot.T @ P_flat
        H2_probs = H2_onehot.T @ P_flat
        
        # Compute Entropies
        D_mask = D_probs > 1e-15
        H0_mask = H0_probs > 1e-15
        H1_mask = H1_probs > 1e-15
        H2_mask = H2_probs > 1e-15
        
        H_D = -np.sum(D_probs[D_mask] * np.log2(D_probs[D_mask])) if np.any(D_mask) else 0.0
        H_0 = -np.sum(H0_probs[H0_mask] * np.log2(H0_probs[H0_mask])) if np.any(H0_mask) else 0.0
        H_1 = -np.sum(H1_probs[H1_mask] * np.log2(H1_probs[H1_mask])) if np.any(H1_mask) else 0.0
        H_2 = -np.sum(H2_probs[H2_mask] * np.log2(H2_probs[H2_mask])) if np.any(H2_mask) else 0.0
        
        # Penalize invalid regions
        if H_D < 1e-6 or max(H_0, H_1, H_2) < 1e-6:
            return 1e10, np.zeros_like(params)
        
        # Compute Soft-Max M
        # M_soft = (1/beta) * log( sum(exp(beta * H_i)) )
        # To be stable: max_H + (1/beta) * log( sum(exp(beta * (H_i - max_H))) )
        Hs = np.array([H_0, H_1, H_2])
        max_H = np.max(Hs)
        if max_H == 0:
            return 1e10, np.zeros_like(params)
            
        exp_terms = np.exp(beta * (Hs - max_H))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 1e10, np.zeros_like(params)
            
        M_soft = max_H + (1.0 / beta) * np.log(sum_exp)
        
        # Objective: log(M_soft) - log(H_D)
        val = np.log(M_soft) - np.log(H_D)
        
        # Gradients
        # dH/dp_i = -log2(p_bin) - 1/ln(2) for cells in bin
        
        # Gradient of H_D
        D_coeff = np.zeros(len(D_uniq))
        if np.any(D_mask):
            D_coeff[D_mask] = -(np.log2(D_probs[D_mask]) + 1.0/np.log(2.0))
        grad_H_D = D_onehot @ D_coeff
        
        # Gradient of H_0, H_1, H_2
        H0_coeff = np.zeros(len(H0_uniq))
        if np.any(H0_mask):
            H0_coeff[H0_mask] = -(np.log2(H0_probs[H0_mask]) + 1.0/np.log(2.0))
        grad_H_0 = H0_onehot @ H0_coeff
        
        H1_coeff = np.zeros(len(H1_uniq))
        if np.any(H1_mask):
            H1_coeff[H1_mask] = -(np.log2(H1_probs[H1_mask]) + 1.0/np.log(2.0))
        grad_H_1 = H1_onehot @ H1_coeff
        
        H2_coeff = np.zeros(len(H2_uniq))
        if np.any(H2_mask):
            H2_coeff[H2_mask] = -(np.log2(H2_probs[H2_mask]) + 1.0/np.log(2.0))
        grad_H_2 = H2_onehot @ H2_coeff
        
        # Gradient of M_soft
        # dM_soft/dH_i = exp(beta * (H_i - max_H)) / sum(exp(...))
        # Note: this is exactly softmax(beta * Hs)
        weights = exp_terms / sum_exp  # w0, w1, w2
        
        # Chain rule: grad_M_soft = w0 * grad_H_0 + w1 * grad_H_1 + w2 * grad_H_2
        grad_M_soft = weights[0] * grad_H_0 + weights[1] * grad_H_1 + weights[2] * grad_H_2
        
        # d/dp [ log(M) - log(D) ] = grad_M / M - grad_D / D
        dObj_dP = grad_M_soft / M_soft - grad_H_D / H_D
        
        # Gradient wrt params (log P)
        # dP_i/dparam_j = P_i * (delta_ij - 1) (since sum P = 1)
        # dObj/dparam_j = P_j * (dObj/dP_j - sum_i dObj/dP_i * P_i)
        sum_term = np.sum(dObj_dP * P_flat)
        grad_params = P_flat * (dObj_dP - sum_term)
        
        return val, grad_params

    def objective(params, beta=20.0):
        val, _ = objective_and_grad(params, beta)
        return val

    # Initializations
    initializations = []
    
    # 1. Uniform
    P_init = np.ones((n, n), dtype=np.float64) / (n * n)
    initializations.append(np.log(P_init.flatten()))
    
    # 2. Diagonal bands (spread in x-y)
    for shift in range(-9, 10):
        P_diag = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x - y == shift:
                    P_diag[y, x] = 1.0
        if np.sum(P_diag) > 0:
            P_diag /= np.sum(P_diag)
            initializations.append(np.log(P_diag.flatten()))
            
    # 3. Anti-diagonal bands (spread in x+y)
    for shift in range(0, 2*n):
        P_adiag = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x + y == shift:
                    P_adiag[y, x] = 1.0
        if np.sum(P_adiag) > 0:
            P_adiag /= np.sum(P_adiag)
            initializations.append(np.log(P_adiag.flatten()))

    # 4. Random Dirichlet
    np.random.seed(42)
    for _ in range(20):
        P_rand = np.random.dirichlet(np.ones(n * n))
        initializations.append(np.log(P_rand))
        
    # 5. Structured Random: Pick a few points
    for _ in range(20):
        P_struct = np.zeros((n, n))
        num_points = np.random.randint(3, 10)
        points = np.random.choice(n*n, size=num_points, replace=False)
        P_struct_flat = np.zeros(n*n)
        P_struct_flat[points] = 1.0
        P_struct_flat /= np.sum(P_struct_flat)
        initializations.append(np.log(P_struct_flat))

    best_P = P_init.copy()
    best_val = objective(initializations[0], beta=1000.0) # Use high beta for eval
    
    # Optimize from each initialization
    for idx, init_params in enumerate(initializations):
        if time.time() > deadline - 10:
            break
            
        # 1. L-BFGS-B with smoothed objective
        try:
            result = opt.minimize(
                lambda p: objective_and_grad(p, beta=10.0)[0],
                init_params,
                method='L-BFGS-B',
                jac=lambda p: objective_and_grad(p, beta=10.0)[1],
                options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                P_opt = np.exp(result.x)
                P_opt /= np.sum(P_opt)
                # Verify with exact max (beta=inf equivalent)
                val_exact, _ = objective_and_grad(result.x, beta=1000.0)
                if val_exact < best_val:
                    best_val = val_exact
                    best_P = P_opt.reshape(n, n)
        except Exception:
            pass
            
        if time.time() > deadline - 5:
            break

        # 2. Powell (derivative free, good for non-smooth)
        try:
            result = opt.minimize(
                lambda p: objective(p, beta=10.0),
                init_params,
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-8}
            )
            if result.fun < best_val:
                P_opt = np.exp(result.x)
                P_opt /= np.sum(P_opt)
                val_exact, _ = objective_and_grad(result.x, beta=1000.0)
                if val_exact < best_val:
                    best_val = val_exact
                    best_P = P_opt.reshape(n, n)
        except Exception:
            pass

    # Final polish with L-BFGS-B on best found, using higher beta for sharpness
    try:
        best_params = np.log(best_P.flatten())
        # Annealing beta: start low, go high
        for beta in [10.0, 50.0, 100.0]:
            if time.time() > deadline - 2:
                break
            result = opt.minimize(
                lambda p: objective_and_grad(p, beta=beta)[0],
                best_params,
                method='L-BFGS-B',
                jac=lambda p: objective_and_grad(p, beta=beta)[1],
                options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                P_opt = np.exp(result.x)
                P_opt /= np.sum(P_opt)
                val_exact, _ = objective_and_grad(result.x, beta=1000.0)
                if val_exact < best_val:
                    best_val = val_exact
                    best_P = P_opt.reshape(n, n)
                    best_params = result.x
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
