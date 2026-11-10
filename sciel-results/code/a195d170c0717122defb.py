# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp


def _shannon_entropy_bits(probs):
    """Shannon entropy in bits, ignoring zero probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


# Precompute bin assignments for fast ratio computation
_n = 10
_x = np.arange(_n)
_y = np.arange(_n)
_X, _Y = np.meshgrid(_x, _y, indexing='xy')

# H_{-1}: x - y, values range from -9 to 9, shift to 0..18
_bin_minus1 = (_X - _Y + 9).flatten().astype(int)
# H_0: x, values 0..9
_bin_0 = _X.flatten().astype(int)
# H_1: x + y, values 0..18
_bin_1 = (_X + _Y).flatten().astype(int)
# H_2: x + 2y, values 0..27
_bin_2 = (_X + 2 * _Y).flatten().astype(int)

_n_bins_minus1 = 19
_n_bins_0 = 10
_n_bins_1 = 19
_n_bins_2 = 28


def _compute_ratio(P):
    """Compute the ratio D/M for a given 10x10 probability matrix P."""
    P_flat = P.flatten()
    
    # H_{-1}: distribution of x - y
    probs_minus1 = np.bincount(_bin_minus1, weights=P_flat, minlength=_n_bins_minus1)
    H_minus1 = _shannon_entropy_bits(probs_minus1)
    
    # H_0: distribution of x
    probs_0 = np.bincount(_bin_0, weights=P_flat, minlength=_n_bins_0)
    H_0 = _shannon_entropy_bits(probs_0)
    
    # H_1: distribution of x + y
    probs_1 = np.bincount(_bin_1, weights=P_flat, minlength=_n_bins_1)
    H_1 = _shannon_entropy_bits(probs_1)
    
    # H_2: distribution of x + 2y
    probs_2 = np.bincount(_bin_2, weights=P_flat, minlength=_n_bins_2)
    H_2 = _shannon_entropy_bits(probs_2)
    
    M = max(H_0, H_1, H_2)
    if M < 1e-9:
        return 0.0
    return H_minus1 / M


def _entropy_and_grad(probs):
    """
    Compute Shannon entropy and its gradient with respect to the unnormalized weights.
    probs: array of weights (not necessarily summing to 1)
    Returns (H, grad) where grad has same shape as probs.
    """
    probs = np.asarray(probs, dtype=np.float64)
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0, np.zeros_like(probs)
    
    p = probs[mask]
    p_sum = p.sum()
    p_norm = p / p_sum
    
    H = float(-np.sum(p_norm * np.log2(p_norm)))
    
    # Gradient: dH/dp_i = -(log2(p_i/p_sum) + 1) / ln(2) for p_i > 0
    grad = np.zeros_like(probs)
    grad[mask] = -(np.log2(p_norm) + 1.0) / np.log(2.0)
    
    return H, grad


def _compute_ratio_with_grad(P):
    """
    Compute ratio D/M and its gradient with respect to P.
    Returns (ratio, grad) where grad has same shape as P.
    """
    P_flat = P.flatten()
    n = P.shape[0]
    
    # H_{-1}
    probs_minus1 = np.bincount(_bin_minus1, weights=P_flat, minlength=_n_bins_minus1)
    H_minus1, grad_H_minus1 = _entropy_and_grad(probs_minus1)
    
    # Map gradient back to P
    grad_P_minus1 = np.zeros(n*n)
    for i, bin_idx in enumerate(_bin_minus1):
        grad_P_minus1[i] = grad_H_minus1[bin_idx]
    
    # H_0
    probs_0 = np.bincount(_bin_0, weights=P_flat, minlength=_n_bins_0)
    H_0, grad_H_0 = _entropy_and_grad(probs_0)
    grad_P_0 = np.zeros(n*n)
    for i, bin_idx in enumerate(_bin_0):
        grad_P_0[i] = grad_H_0[bin_idx]
    
    # H_1
    probs_1 = np.bincount(_bin_1, weights=P_flat, minlength=_n_bins_1)
    H_1, grad_H_1 = _entropy_and_grad(probs_1)
    grad_P_1 = np.zeros(n*n)
    for i, bin_idx in enumerate(_bin_1):
        grad_P_1[i] = grad_H_1[bin_idx]
    
    # H_2
    probs_2 = np.bincount(_bin_2, weights=P_flat, minlength=_n_bins_2)
    H_2, grad_H_2 = _entropy_and_grad(probs_2)
    grad_P_2 = np.zeros(n*n)
    for i, bin_idx in enumerate(_bin_2):
        grad_P_2[i] = grad_H_2[bin_idx]
    
    M = max(H_0, H_1, H_2)
    if M < 1e-9:
        return 0.0, np.zeros(n*n)
    
    ratio = H_minus1 / M
    
    # Gradient of ratio = D/M:
    # d(ratio)/dP = (dD/dP * M - D * dM/dP) / M^2
    # dM/dP = dH_k/dP where k is argmax(H_0, H_1, H_2)
    
    if H_0 >= H_1 and H_0 >= H_2:
        grad_M = grad_P_0
    elif H_1 >= H_0 and H_1 >= H_2:
        grad_M = grad_P_1
    else:
        grad_M = grad_P_2
    
    grad_ratio = (grad_P_minus1 * M - H_minus1 * grad_M) / (M * M)
    
    return ratio, grad_ratio.reshape(n, n)


def _soft_max_entropy(params, n=10, beta=50.0):
    """
    Compute soft-max of H_0, H_1, H_2 using log-sum-exp for smoothness.
    params: flattened P (100 elements)
    Returns (M_smooth, grad_M) where grad_M has shape (100,).
    """
    P_flat = params.reshape(-1)
    
    probs_0 = np.bincount(_bin_0, weights=P_flat, minlength=_n_bins_0)
    H_0, grad_H_0 = _entropy_and_grad(probs_0)
    grad_P_0 = np.zeros(n*n)
    for i, bin_idx in enumerate(_bin_0):
        grad_P_0[i] = grad_H_0[bin_idx]
    
    probs_1 = np.bincount(_bin_1, weights=P_flat, minlength=_n_bins_1)
    H_1, grad_H_1 = _entropy_and_grad(probs_1)
    grad_P_1 = np.zeros(n*n)
    for i, bin_idx in enumerate(_bin_1):
        grad_P_1[i] = grad_H_1[bin_idx]
    
    probs_2 = np.bincount(_bin_2, weights=P_flat, minlength=_n_bins_2)
    H_2, grad_H_2 = _entropy_and_grad(probs_2)
    grad_P_2 = np.zeros(n*n)
    for i, bin_idx in enumerate(_bin_2):
        grad_P_2[i] = grad_H_2[bin_idx]
    
    Hs = np.array([H_0, H_1, H_2])
    gs = [grad_P_0, grad_P_1, grad_P_2]
    
    # Soft max via log-sum-exp
    scaled = beta * Hs
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    
    if sum_exp < 1e-300:
        M = np.max(Hs)
        if M < 1e-9:
            return 0.0, np.zeros(n*n)
        weights_max = np.zeros(3)
        weights_max[np.argmax(Hs)] = 1.0
        gM = weights_max[0]*gs[0] + weights_max[1]*gs[1] + weights_max[2]*gs[2]
    else:
        M = (max_scaled + np.log(sum_exp)) / beta
        if M < 1e-9:
            return 0.0, np.zeros(n*n)
        weights_max = exp_scaled / sum_exp
        gM = weights_max[0]*gs[0] + weights_max[1]*gs[1] + weights_max[2]*gs[2]
    
    return M, gM


def _objective_general_smooth(params, n=10, beta=50.0):
    """
    Objective for general P, parameterized by log-weights, with soft-max for M.
    params: log(P_{y,x} + epsilon) for all y, x
    """
    log_P = params.reshape(n, n)
    log_P_shifted = log_P - np.max(log_P)
    P = np.exp(log_P_shifted)
    s = P.sum()
    if s < 1e-15:
        return 1e6, np.zeros(n*n)
    P = P / s
    
    P_flat = P.flatten()
    
    # H_{-1}
    probs_minus1 = np.bincount(_bin_minus1, weights=P_flat, minlength=_n_bins_minus1)
    H_minus1, grad_H_minus1 = _entropy_and_grad(probs_minus1)
    grad_P_minus1 = np.zeros(n*n)
    for i, bin_idx in enumerate(_bin_minus1):
        grad_P_minus1[i] = grad_H_minus1[bin_idx]
    
    # Soft-max M
    M, grad_M = _soft_max_entropy(P_flat, n, beta)
    
    if M < 1e-9:
        return 1e6, np.zeros(n*n)
    
    ratio = H_minus1 / M
    grad_ratio = (grad_P_minus1 * M - H_minus1 * grad_M) / (M * M)
    
    obj = -ratio
    grad = -grad_ratio.reshape(n, n)
    
    return obj, grad


def _objective_product(params, n=10):
    """
    Objective for product-form P[y,x] = R[y] * C[x].
    params: [log_R_0, ..., log_R_{n-1}, log_C_0, ..., log_C_{n-1}]
    """
    log_R = params[:n]
    log_C = params[n:2*n]
    
    log_R_shifted = log_R - np.max(log_R)
    log_C_shifted = log_C - np.max(log_C)
    
    R = np.exp(log_R_shifted)
    C = np.exp(log_C_shifted)
    
    R = R / R.sum()
    C = C / C.sum()
    
    P = np.outer(R, C)
    
    ratio = _compute_ratio(P)
    return -ratio


def _objective_rank2(params, n=10):
    """
    Objective for rank-2 factorization P[y,x] = (U[y,0]*V[x,0] + U[y,1]*V[x,1])_+ / Z.
    params: [U_00, U_01, ..., U_91, V_00, V_01, ..., V_91]
    """
    U = params[:2*n].reshape(n, 2)
    V = params[2*n:4*n].reshape(n, 2)
    
    M = U @ V.T
    P_raw = np.clip(M, 0, None)
    
    s = P_raw.sum()
    if s < 1e-15:
        return 1e6
    P = P_raw / s
    
    ratio = _compute_ratio(P)
    return -ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600 - 30  # safety margin
    
    # Start with uniform distribution
    P0 = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P0.copy()
    best_ratio = _compute_ratio(P0)
    
    np.random.seed(42)
    
    # Strategy 1: Product-form optimization P[y,x] = R[y] * C[x]
    for trial in range(15):
        if time.time() > deadline - 120:
            break
        
        scale = np.random.choice([0.3, 0.5, 0.8, 1.0])
        log_R = np.random.randn(n) * scale
        log_C = np.random.randn(n) * scale
        params = np.concatenate([log_R, log_C])
        
        try:
            result = minimize(
                _objective_product,
                params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200},
                args=(n,)
            )
            
            if result.fun < -best_ratio:
                log_R_opt = result.x[:n]
                log_C_opt = result.x[n:2*n]
                
                log_R_shifted = log_R_opt - np.max(log_R_opt)
                log_C_shifted = log_C_opt - np.max(log_C_opt)
                
                R = np.exp(log_R_shifted)
                C = np.exp(log_C_shifted)
                R = R / R.sum()
                C = C / C.sum()
                
                P_opt = np.outer(R, C)
                ratio_opt = _compute_ratio(P_opt)
                if ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_P = P_opt.copy()
        except Exception:
            pass

    # Strategy 2: Rank-2 factorization optimization
    for trial in range(25):
        if time.time() > deadline - 90:
            break
        
        scale = np.random.choice([0.3, 0.5, 0.8])
        U_init = np.random.randn(n, 2) * scale
        V_init = np.random.randn(n, 2) * scale
        params = np.concatenate([U_init.flatten(), V_init.flatten()])
        
        try:
            result = minimize(
                _objective_rank2,
                params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200},
                args=(n,)
            )
            
            if result.fun < -best_ratio:
                U_opt = result.x[:2*n].reshape(n, 2)
                V_opt = result.x[2*n:4*n].reshape(n, 2)
                
                M = U_opt @ V_opt.T
                P_raw = np.clip(M, 0, None)
                s = P_raw.sum()
                if s > 1e-15:
                    P_opt = P_raw / s
                    ratio_opt = _compute_ratio(P_opt)
                    if ratio_opt > best_ratio:
                        best_ratio = ratio_opt
                        best_P = P_opt.copy()
        except Exception:
            pass

    # Strategy 3: General optimization with soft-max surrogate for M
    log_P_init = np.log(best_P + 1e-15)
    
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        
        if trial == 0:
            params = log_P_init.flatten().copy()
        else:
            scale = np.random.choice([0.05, 0.1, 0.2, 0.3])
            params = log_P_init.flatten() + np.random.randn(n*n) * scale
        
        try:
            def obj_only(x):
                return _objective_general_smooth(x, n, beta=50.0)[0]
            
            def grad_only(x):
                return _objective_general_smooth(x, n, beta=50.0)[1].flatten()
            
            result = minimize(
                obj_only,
                params,
                jac=grad_only,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
            )
            
            log_P_opt = result.x.reshape(n, n)
            log_P_shifted = log_P_opt - np.max(log_P_opt)
            P_opt = np.exp(log_P_shifted)
            s_opt = P_opt.sum()
            if s_opt > 1e-15:
                P_opt = P_opt / s_opt
                ratio_opt = _compute_ratio(P_opt)
                if ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_P = P_opt.copy()
                    log_P_init = np.log(best_P + 1e-15)
        except Exception:
            pass
    
    # Strategy 4: Gradient-based local search using analytic gradients
    if time.time() < deadline - 30:
        P_current = best_P.copy()
        P_current = P_current / P_current.sum()
        
        for iteration in range(800):
            if time.time() > deadline - 10:
                break
            
            ratio, grad = _compute_ratio_with_grad(P_current)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = P_current.copy()
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            for step in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
                P_try = P_current + step * grad
                P_try = np.clip(P_try, 1e-15, None)
                s_try = P_try.sum()
                if s_try > 1e-12:
                    P_try = P_try / s_try
                    ratio_try = _compute_ratio(P_try)
                    if ratio_try > best_ratio:
                        best_ratio = ratio_try
                        best_P = P_try.copy()
                        P_current = P_try.copy()
                        break
    
    # Strategy 5: Local perturbations on the best solution
    if time.time() < deadline - 15:
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            
            scale = np.random.choice([0.005, 0.01, 0.02, 0.05, 0.1, 0.15])
            P_pert = best_P + np.random.randn(n, n) * scale
            P_pert = np.clip(P_pert, 1e-15, None)
            s_pert = P_pert.sum()
            if s_pert > 1e-12:
                P_pert = P_pert / s_pert
                ratio_pert = _compute_ratio(P_pert)
                if ratio_pert > best_ratio:
                    best_ratio = ratio_pert
                    best_P = P_pert.copy()
    
    # Strategy 6: Structured initialization based on Kakeya geometry
    if time.time() < deadline - 20:
        for width in [1, 2, 3, 4, 5]:
            P_struct = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    d = abs(x - y)
                    if d <= width:
                        P_struct[y, x] = 1.0 / (d + 1)
            s = P_struct.sum()
            if s > 1e-12:
                P_struct = P_struct / s
                ratio_struct = _compute_ratio(P_struct)
                if ratio_struct > best_ratio:
                    best_ratio = ratio_struct
                    best_P = P_struct.copy()
        
        for width in [1, 2, 3, 4, 5]:
            P_struct = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    d = abs(x + y - (n - 1))
                    if d <= width:
                        P_struct[y, x] = 1.0 / (d + 1)
            s = P_struct.sum()
            if s > 1e-12:
                P_struct = P_struct / s
                ratio_struct = _compute_ratio(P_struct)
                if ratio_struct > best_ratio:
                    best_ratio = ratio_struct
                    best_P = P_struct.copy()
    
    # Final polish: tight gradient-based refinement
    if time.time() < deadline - 10:
        P_current = best_P.copy()
        P_current = P_current / P_current.sum()
        
        for iteration in range(500):
            if time.time() > deadline - 2:
                break
            
            ratio, grad = _compute_ratio_with_grad(P_current)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = P_current.copy()
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            for step in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
                P_try = P_current + step * grad
                P_try = np.clip(P_try, 1e-15, None)
                s_try = P_try.sum()
                if s_try > 1e-12:
                    P_try = P_try / s_try
                    ratio_try = _compute_ratio(P_try)
                    if ratio_try > best_ratio:
                        best_ratio = ratio_try
                        best_P = P_try.copy()
                        P_current = P_try.copy()
                        break
    
    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    s_final = best_P.sum()
    if s_final < 1e-9:
        best_P = P0
    else:
        best_P = best_P / s_final
    
    return best_P


# EVOLVE-BLOCK-END
