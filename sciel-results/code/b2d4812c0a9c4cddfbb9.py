# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_EPS = 1e-12
_SLOPES = [-1.0, 0.0, 1.0, 2.0]

# Precompute projection indices for each slope
# P[y, x] -> index y*N + x
# v = x + slope * y
# We need to map each cell to a bin index for counts.
# v_min and v_max depend on slope.
# Slope -1: v = x - y. Range: -9 to 9. Size 19.
# Slope 0: v = x. Range: 0 to 9. Size 10.
# Slope 1: v = x + y. Range: 0 to 18. Size 19.
# Slope 2: v = x + 2y. Range: 0 to 27. Size 28.

def _get_proj_indices(slope):
    n = _N
    indices = np.zeros((n, n), dtype=np.int32)
    v_min = None
    v_max = None
    for y in range(n):
        for x in range(n):
            v = x + slope * y
            if v_min is None:
                v_min = v
                v_max = v
            else:
                v_min = min(v_min, v)
                v_max = max(v_max, v)
    size = int(round(v_max - v_min)) + 1
    for y in range(n):
        for x in range(n):
            v = x + slope * y
            indices[y, x] = int(round(v - v_min))
    return indices, size

PROJ_INDICES = {}
PROJ_SIZES = {}
for s in _SLOPES:
    idx, sz = _get_proj_indices(s)
    PROJ_INDICES[s] = idx
    PROJ_SIZES[s] = sz

def _shannon_entropy_bits(counts):
    total = counts.sum()
    if total <= _EPS:
        return 0.0
    p = counts / total
    p = p[p > _EPS]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _compute_entropies(weights, support):
    """
    weights: array of length len(support), sum 1.
    support: array of flat indices (y*N + x).
    Returns (H_minus1, H_0, H_1, H_2).
    """
    Hs = []
    for s in _SLOPES:
        counts = np.zeros(PROJ_SIZES[s], dtype=np.float64)
        # Get the bin indices for the cells in support
        bins = PROJ_INDICES[s].ravel()[support]
        np.add.at(counts, bins, weights)
        Hs.append(_shannon_entropy_bits(counts))
    return Hs[0], Hs[1], Hs[2], Hs[3]

def _ratio_from_entropies(Hm1, H0, H1, H2):
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return Hm1 / M

def _compute_entropies_and_grads(weights, support):
    """
    Compute entropies and gradients w.r.t weights for the 4 slopes.
    Returns:
    Hs: [H_m1, H_0, H_1, H_2]
    dHs: [dH_m1, dH_0, dH_1, dH_2] where dH_r[i] is dH_r/dP_i for i in support
    """
    Hs = []
    dHs = []
    for s in _SLOPES:
        counts = np.zeros(PROJ_SIZES[s], dtype=np.float64)
        bins = PROJ_INDICES[s].ravel()[support]
        np.add.at(counts, bins, weights)
        
        total = counts.sum()
        if total <= _EPS:
            Hs.append(0.0)
            dHs.append(np.zeros(len(support)))
            continue
        
        p = counts / total
        p_safe = p[p > _EPS]
        H = -np.sum(p_safe * np.log2(p_safe))
        Hs.append(H)
        
        # Gradient of H_r w.r.t P_i
        # dH/dP_i = -1/ln(2) * (ln(p_bin) + 1)
        # We need ln(p_bin) for each i.
        # p_bin for cell i is counts[bins[i]] / total
        p_bins = counts[bins] / total
        p_bins_safe = np.where(p_bins > _EPS, p_bins, 1.0) # Avoid log(0)
        dH = -(np.log2(p_bins_safe) + 1.0)
        # Where p_bins was 0 (shouldn't happen if weights > 0 and support covers it, but safe)
        # Actually if weight is 0, it's not in support? No, support is fixed.
        # If weight is very small, p_bin might be 0 due to underflow? 
        # Weights are softmax, so > 0. Counts > 0.
        dHs.append(dH)
        
    return Hs, dHs

def _objective_log_ratio(theta, support):
    """
    Returns negative log-ratio: - (ln(H_m1) - LSE(H_0, H_1, H_2))
    """
    z = theta - np.max(theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    if s < _EPS:
        weights = np.ones(len(support)) / len(support)
    else:
        weights = exp_z / s
        
    Hs, _ = _compute_entropies_and_grads(weights, support)
    Hm1, H0, H1, H2 = Hs
    
    if Hm1 <= _EPS:
        return 1e10 # Large penalty
        
    # Soft max for M to ensure smoothness
    # LSE = ln(exp(H0) + exp(H1) + exp(H2))
    # Scale to avoid overflow
    Hs_max = max(H0, H1, H2)
    if Hs_max > 100: # Unlikely for 10x10
        Hs_max = 100
    lse = Hs_max + np.log(np.exp(H0 - Hs_max) + np.exp(H1 - Hs_max) + np.exp(H2 - Hs_max))
    
    log_ratio = np.log2(Hm1) - np.log2(lse)
    # Note: Task asks for D/M. Maximizing D/M is same as maximizing log(D/M).
    # Base of log doesn't matter for optimization.
    return -log_ratio

def _gradient_log_ratio(theta, support):
    """
    Returns gradient of the negative log-ratio w.r.t theta.
    """
    z = theta - np.max(theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    if s < _EPS:
        weights = np.ones(len(support)) / len(support)
    else:
        weights = exp_z / s
        
    Hs, dHs = _compute_entropies_and_grads(weights, support)
    Hm1, H0, H1, H2 = Hs
    dHm1, dH0, dH1, dH2 = dHs
    
    if Hm1 <= _EPS:
        return np.zeros(len(support))
        
    # LSE and weights for max part
    Hs_max = max(H0, H1, H2)
    if Hs_max > 100:
        Hs_max = 100
    exps = np.array([np.exp(H0 - Hs_max), np.exp(H1 - Hs_max), np.exp(H2 - Hs_max)])
    sum_exps = np.sum(exps)
    w0 = exps[0] / sum_exps
    w1 = exps[1] / sum_exps
    w2 = exps[2] / sum_exps
    
    # Gradient of log_ratio w.r.t P
    # d(log Hm1)/dP = dHm1 / (Hm1 * ln(2))  (since H is in bits, ln(H) derivative is 1/H * dH, and dH uses log2, so dH = -sum p log2 p, dH/dP = -(log2 p + 1)/ln(2)? 
    # Wait. H = - sum p log2 p. dH/dp = - (log2 p + 1/ln(2)).
    # My dHs calculation: dH = -(np.log2(p_bins_safe) + 1.0). This is dH/dP_i * (1/total)? 
    # Let's re-verify dH/dP_i.
    # H = - sum_k (N_k/N) log2(N_k/N)
    # dH/dN_i = -1/ln(2) * (log2(N_k/N) + 1)
    # Since P_i = N_i (normalized), and we are treating weights as P directly.
    # So dH/dP_i = -1/ln(2) * (log2(p_bin) + 1).
    # My code: dH = -(np.log2(p_bins_safe) + 1.0). Missing 1/ln(2).
    # Let's fix that in the gradient calculation or the dH calculation.
    # It's better to fix it in the gradient step to keep dHs consistent if used elsewhere? 
    # I'll just multiply by 1/ln(2) here.
    
    inv_ln2 = 1.0 / np.log(2.0)
    
    grad_dHm1 = dHm1 * inv_ln2
    grad_dH0 = dH0 * inv_ln2
    grad_dH1 = dH1 * inv_ln2
    grad_dH2 = dH2 * inv_ln2
    
    # d(log Hm1)/dP = grad_dHm1 / Hm1
    # d(LSE)/dP = w0 * grad_dH0 + w1 * grad_dH1 + w2 * grad_dH2
    # Note: LSE is in bits? No, LSE is ln(sum exp(H)). H is in bits.
    # So LSE is in "ln(bits)" units? 
    # log_ratio = log2(Hm1) - log2(LSE).
    # d(log2(Hm1))/dP = (1/(Hm1 ln 2)) * dHm1/dP.
    # d(log2(LSE))/dP = (1/(LSE ln 2)) * dLSE/dP.
    # dLSE/dP = sum w_j dH_j/dP.
    
    term1 = grad_dHm1 / (Hm1 * np.log(2.0))
    term2 = (w0 * grad_dH0 + w1 * grad_dH1 + w2 * grad_dH2) / (lse_val * np.log(2.0))
    # Wait, I need lse_val.
    lse_val = Hs_max + np.log(sum_exps)
    
    # Correction: 
    # d(log2(Hm1)) = (1/ln 2) * (1/Hm1) * dHm1
    # d(log2(LSE)) = (1/ln 2) * (1/LSE) * dLSE
    # So:
    grad_F_dP = (1.0 / np.log(2.0)) * (grad_dHm1 / Hm1 - (w0 * grad_dH0 + w1 * grad_dH1 + w2 * grad_dH2) / lse_val)
    
    # Gradient w.r.t theta
    # dF/dtheta_k = sum_i dF/dP_i * dP_i/dtheta_k
    # = sum_i dF/dP_i * P_i * (delta_ik - P_k)
    # = dF/dP_k * P_k - P_k * sum_i dF/dP_i * P_i
    
    dot = np.dot(grad_F_dP, weights)
    grad_theta = weights * (grad_F_dP - dot)
    
    # We are minimizing -log_ratio, so return -grad_theta
    return -grad_theta

def _optimize_support(support, deadline):
    """
    Optimize weights for a given support set using SLSQP with analytical gradients.
    Returns (best_ratio, best_P_full).
    """
    K = len(support)
    if K == 0:
        return 0.0, None
    
    # Initial guess: uniform
    theta0 = np.zeros(K)
    
    best_ratio = 0.0
    best_theta = theta0
    
    # Try a few random initializations
    rng = np.random.default_rng(42)
    attempts = 0
    max_attempts = 5 if K < 15 else 2
    
    while time.time() < deadline - 1.0 and attempts < max_attempts:
        attempts += 1
        if attempts == 1:
            theta_start = np.zeros(K)
        else:
            theta_start = rng.standard_normal(K) * 0.5
        
        try:
            res = minimize(
                _objective_log_ratio,
                theta_start,
                args=(support,),
                jac=_gradient_log_ratio,
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun is not None and res.fun < 1e9:
                # Calculate actual ratio for comparison
                z = res.x - np.max(res.x)
                exp_z = np.exp(z)
                weights = exp_z / exp_z.sum()
                Hs, _ = _compute_entropies_and_grads(weights, support)
                Hm1, H0, H1, H2 = Hs
                M = max(H0, H1, H2)
                if M > 1e-9:
                    ratio = Hm1 / M
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_theta = res.x
        except Exception:
            pass
            
    if best_ratio <= 0:
        return 0.0, None
        
    # Construct full P
    z = best_theta - np.max(best_theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    weights = exp_z / s
    
    P = np.zeros(_N * _N, dtype=np.float64)
    P[support] = weights
    P = P.reshape(_N, _N)
    
    return best_ratio, P

def _random_support(rng, min_size=5, max_size=20):
    K = rng.integers(min_size, max_size + 1)
    return rng.choice(_N * _N, size=K, replace=False)

def _structured_supports():
    """Generate some structured supports."""
    supports = []
    n = _N
    # Diagonals
    for d in range(-n+1, n):
        cells = []
        for y in range(n):
            x = y + d
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Anti-diagonals
    for s in range(0, 2*n-1):
        cells = []
        for y in range(n):
            x = s - y
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Slope -2 lines (x + 2y = c)
    for c in range(0, 2*n):
        cells = []
        for y in range(n):
            x = c - 2*y
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Slope 2 lines (x - 2y = c) -> x = 2y + c
    for c in range(-2*n, 2*n):
        cells = []
        for y in range(n):
            x = 2*y + c
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Rows and Cols
    for i in range(n):
        supports.append(np.array([i * n + j for j in range(n)])) # Row i
        supports.append(np.array([j * n + i for j in range(n)])) # Col i
        
    return supports

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0 # 10s margin
    
    best_ratio = 0.0
    best_P = np.ones((_N, _N)) / (_N * _N)
    
    rng = np.random.default_rng(42)
    
    # 1. Try structured supports
    struct_supports = _structured_supports()
    for support in struct_supports:
        if time.time() > deadline - 30.0:
            break
        ratio, P = _optimize_support(support, deadline)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
            
    # 2. Try random supports
    # We can do many of these quickly
    num_random = 0
    while time.time() < deadline - 10.0:
        num_random += 1
        # Vary support size
        if num_random % 10 == 0:
            min_sz, max_sz = 5, 10
        elif num_random % 10 == 5:
            min_sz, max_sz = 10, 15
        else:
            min_sz, max_sz = 15, 25
            
        support = _random_support(rng, min_sz, max_sz)
        ratio, P = _optimize_support(support, deadline)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
            
        # If we've been going for a while, increase support size or change strategy
        if num_random % 20 == 0:
             # Try a support based on current best P's non-zero elements
             if best_ratio > 0:
                 nz = np.nonzero(best_P.ravel() > 1e-6)[0]
                 if len(nz) > 0:
                     # Perturb this support
                     perturbed = nz.copy()
                     # Replace a few elements
                     for _ in range(3):
                         idx_to_remove = rng.integers(len(perturbed))
                         perturbed = np.delete(perturbed, idx_to_remove)
                         new_elem = rng.integers(_N * _N)
                         if new_elem not in perturbed:
                             perturbed = np.append(perturbed, new_elem)
                     ratio, P = _optimize_support(perturbed, deadline)
                     if ratio > best_ratio:
                         best_ratio = ratio
                         best_P = P

    # Final validation and normalization
    P = np.clip(best_P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P = P / s
        
    return P

# EVOLVE-BLOCK-END
