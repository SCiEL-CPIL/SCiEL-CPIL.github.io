# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Set a global deadline to avoid exceeding the evaluator's wall-clock limit
# The evaluator kills after 630s. We use 590s as a safe deadline.
GLOBAL_DEADLINE = 590.0
START_TIME = time.time()


def _time_left():
    return GLOBAL_DEADLINE - (time.time() - START_TIME)


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the ratio D / M where:
    D = H_{-1} (entropy of distribution of x - y)
    M = max(H_0, H_1, H_2) (max entropy of x, x+y, x+2y)
    
    P is a 10x10 array P[y, x] for x, y in 0..9.
    """
    start_time = time.time()
    deadline = start_time + 590  # 590s safety margin out of 600s
    
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

    def soft_max_entropy(H0, H1, H2, beta):
        """Compute soft max of entropies to make objective differentiable."""
        # Use log-sum-exp for stability
        max_H = max(H0, H1, H2)
        if max_H == 0:
            return 0.0
        exp_terms = np.exp(beta * np.array([H0, H1, H2]) - beta * max_H)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 0.0
        return max_H + (1.0 / beta) * np.log(sum_exp)

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
        M_soft = soft_max_entropy(H_0, H_1, H_2, beta)
        
        # Objective: log(M_soft) - log(H_D)
        val = np.log(M_soft) - np.log(H_D)
        
        # Gradients
        # dH/dp_i = -log2(p_bin) - 1/ln(2) for cells in bin
        # We need grad_D and grad_M_soft
        
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
        
        # Gradient of M_soft = (1/beta) log(sum exp(beta H_i))
        # dM_soft/dH_i = exp(beta H_i) / sum exp(beta H_j)
        exp_H0 = np.exp(beta * H_0)
        exp_H1 = np.exp(beta * H_1)
        exp_H2 = np.exp(beta * H_2)
        sum_exp = exp_H0 + exp_H1 + exp_H2
        
        if sum_exp > 0:
            w0 = exp_H0 / sum_exp
            w1 = exp_H1 / sum_exp
            w2 = exp_H2 / sum_exp
        else:
            w0 = w1 = w2 = 0.0
        
        # Chain rule: grad_M_soft = w0 * grad_H_0 + w1 * grad_H_1 + w2 * grad_H_2
        grad_M_soft = w0 * grad_H_0 + w1 * grad_H_1 + w2 * grad_H_2
        
        # d/dp [ log(M) - log(D) ] = grad_M / M - grad_D / D
        dObj_dP = grad_M_soft / M_soft - grad_H_D / H_D
        
        # Gradient wrt params (log P)
        # dP_i/dparam_j = P_i * (delta_ij - sum_k P_k) -> P_i * (delta_ij - 1) since sum P = 1
        # dObj/dparam_j = P_j * (dObj/dP_j - sum_i dObj/dP_i * P_i)
        sum_term = np.sum(dObj_dP * P_flat)
        grad_params = P_flat * (dObj_dP - sum_term)
        
        return val, grad_params

    def objective(params, beta=20.0):
        val, _ = objective_and_grad(params, beta)
        return val

    def jac(params, beta=20.0):
        _, grad = objective_and_grad(params, beta)
        return grad

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
            # Add small noise to break symmetry and allow movement
            P_noisy = P_diag + 1e-6 * np.random.rand(n, n)
            P_noisy /= P_noisy.sum()
            initializations.append(np.log(P_noisy.flatten()))
            
    # 3. Anti-diagonal bands (spread in x+y)
    for shift in range(0, 19):
        P_adiag = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x + y == shift:
                    P_adiag[y, x] = 1.0
        if np.sum(P_adiag) > 0:
            P_adiag /= np.sum(P_adiag)
            P_noisy = P_adiag + 1e-6 * np.random.rand(n, n)
            P_noisy /= P_noisy.sum()
            initializations.append(np.log(P_noisy.flatten()))
            
    # 4. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        P_rand = rng.random((n, n))
        P_rand /= P_rand.sum()
        initializations.append(np.log(P_rand.flatten()))
        
    # Optimization loop
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = -1e10
    
    def _update_best(P, score):
        nonlocal best_P, best_score
        if score > best_score:
            best_score = score
            best_P = P.copy()
            
    # Evaluate baseline
    ent = compute_entropies(best_P.flatten())
    D = ent[0]
    M = ent[1]
    if M > 1e-9:
        base_ratio = D / M
        _update_best(best_P, base_ratio)
        
    for i, theta0 in enumerate(initializations):
        if _time_left() < 30:
            break
            
        try:
            res = opt.minimize(
                objective,
                theta0,
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            # Extract best P from result
            theta_opt = res.x
            P_opt = np.exp(theta_opt - np.max(theta_opt))
            P_opt = P_opt / P_opt.sum()
            P_opt = P_opt.reshape((n, n))
            
            # Ensure validity
            if P_opt.shape != (n, n):
                continue
            if np.any(np.isnan(P_opt)) or np.any(np.isinf(P_opt)):
                continue
            if np.any(P_opt < -1e-9):
                continue
            if P_opt.sum() <= 1e-9:
                continue
                
            # Clip and normalize
            P_opt = np.clip(P_opt, 0, None)
            P_opt = P_opt / P_opt.sum()
            
            # Evaluate score
            ent = compute_entropies(P_opt.flatten())
            D = ent[0]
            M = ent[1]
            
            if M < 1e-9:
                continue
                
            score = D / M
            _update_best(P_opt, score)
            
        except Exception as e:
            pass
            
    # Final check and return
    P_final = best_P
    P_final = np.clip(P_final, 0, None)
    P_final = P_final / P_final.sum()
    
    return P_final

# EVOLVE-BLOCK-END
