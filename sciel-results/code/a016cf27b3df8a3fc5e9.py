import numpy as np
import time
import scipy.optimize as opt

# Pre-compute the index maps for projections
x_indices = np.arange(10)
y_indices = np.arange(10)
X_grid, Y_grid = np.meshgrid(x_indices, y_indices, indexing='xy')

D_vals = (X_grid - Y_grid)
H0_vals = X_grid
H1_vals = (X_grid + Y_grid)
H2_vals = (X_grid + 2 * Y_grid)

# Flatten indices
D_flat = D_vals.ravel()
H0_flat = H0_vals.ravel()
H1_flat = H1_vals.ravel()
H2_flat = H2_vals.ravel()

# Unique values and inverse maps for bincount
D_uniq, D_inv = np.unique(D_flat, return_inverse=True)
H0_uniq, H0_inv = np.unique(H0_flat, return_inverse=True)
H1_uniq, H1_inv = np.unique(H1_flat, return_inverse=True)
H2_uniq, H2_inv = np.unique(H2_flat, return_inverse=True)

D_bins = len(D_uniq)
H0_bins = len(H0_uniq)
H1_bins = len(H1_uniq)
H2_bins = len(H2_uniq)

def calc_entropy(bin_indices, weights, n_bins):
    q = np.bincount(bin_indices, weights=weights, minlength=n_bins)
    mask = q > 1e-15
    if not np.any(mask):
        return 0.0
    q_nonzero = q[mask]
    return -np.sum(q_nonzero * np.log2(q_nonzero))

def calc_grad_entropy(bin_indices, weights, n_bins):
    q = np.bincount(bin_indices, weights=weights, minlength=n_bins)
    mask = q > 1e-15
    if not np.any(mask):
        return 0.0, np.zeros_like(weights)
    q_nonzero = q[mask]
    H = -np.sum(q_nonzero * np.log2(q_nonzero))
    grad_q = np.zeros(n_bins)
    grad_q[mask] = -np.log2(q_nonzero) - 1.0 / np.log(2.0)
    grad_w = grad_q[bin_indices]
    return H, grad_w

def score_weights(w, supp_D, supp_H0, supp_H1, supp_H2):
    D = calc_entropy(supp_D, w, D_bins)
    H0 = calc_entropy(supp_H0, w, H0_bins)
    H1 = calc_entropy(supp_H1, w, H1_bins)
    H2 = calc_entropy(supp_H2, w, H2_bins)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 0.0
    return D / M

def optimize_weights_for_support(support_indices, beta=30.0):
    """
    support_indices: array of flat indices (0-99)
    Returns optimal weights (array of len K) and the score.
    """
    K = len(support_indices)
    if K == 0:
        return np.array([]), 0.0
    
    # Pre-compute bin indices for this support
    supp_D = D_inv[support_indices]
    supp_H0 = H0_inv[support_indices]
    supp_H1 = H1_inv[support_indices]
    supp_H2 = H2_inv[support_indices]

    # Initial guess: uniform
    w0 = np.ones(K) / K
    
    def obj_and_grad(w):
        D, gD = calc_grad_entropy(supp_D, w, D_bins)
        H0, gH0 = calc_grad_entropy(supp_H0, w, H0_bins)
        H1, gH1 = calc_grad_entropy(supp_H1, w, H1_bins)
        H2, gH2 = calc_grad_entropy(supp_H2, w, H2_bins)
        
        Hs = np.array([H0, H1, H2])
        
        # Soft max
        scaled = beta * Hs
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        sum_exp = np.sum(exp_scaled)
        
        if sum_exp < 1e-300:
            M = np.max(Hs)
            if M < 1e-9:
                return -1e9, np.zeros_like(w)
            weights_max = np.zeros(3)
            weights_max[np.argmax(Hs)] = 1.0
            gM = weights_max[0]*gH0 + weights_max[1]*gH1 + weights_max[2]*gH2
        else:
            M = (max_scaled + np.log(sum_exp)) / beta
            if M < 1e-9:
                return -1e9, np.zeros_like(w)
            weights_max = exp_scaled / sum_exp
            gM = weights_max[0]*gH0 + weights_max[1]*gH1 + weights_max[2]*gH2

        ratio = D / M
        # grad of -ratio = -(M*gD - D*gM)/M^2
        grad_ratio = (M * gD - D * gM) / (M * M)
        
        obj = -ratio
        grad = -grad_ratio
        return obj, grad

    try:
        res = opt.minimize(
            lambda x: obj_and_grad(x)[0],
            w0,
            jac=lambda x: obj_and_grad(x)[1],
            method='SLSQP',
            bounds=[(0, 1)] * K,
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                          'jac': lambda x: np.ones(K)}],
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        w_opt = res.x
        w_opt = np.clip(w_opt, 0, None)
        s = w_opt.sum()
        if s > 0:
            w_opt /= s
            sc = score_weights(w_opt, supp_D, supp_H0, supp_H1, supp_H2)
            return w_opt, sc
    except Exception:
        pass
        
    # Fallback: simple uniform
    w_unif = np.ones(K) / K
    sc = score_weights(w_unif, supp_D, supp_H0, supp_H1, supp_H2)
    return w_unif, sc

def heuristic_score(supp):
    if len(supp) == 0: return -1
    n_d = len(np.unique(D_flat[supp]))
    n_h0 = len(np.unique(H0_flat[supp]))
    n_h1 = len(np.unique(H1_flat[supp]))
    n_h2 = len(np.unique(H2_flat[supp]))
    if n_h0 == 0 or n_h1 == 0 or n_h2 == 0: return -1
    h_d = np.log2(n_d)
    m_h = max(np.log2(n_h0), np.log2(n_h1), np.log2(n_h2))
    return h_d / m_h

def solve_kakeya():
    start_time = time.time()
    timeout = 630
    safety_margin = 20
    
    best_P = np.ones((10, 10)) / 100.0
    def score_P_full(P):
        P_flat = P.ravel()
        D = calc_entropy(D_inv, P_flat, D_bins)
        H0 = calc_entropy(H0_inv, P_flat, H0_bins)
        H1 = calc_entropy(H1_inv, P_flat, H1_bins)
        H2 = calc_entropy(H2_inv, P_flat, H2_bins)
        M = max(H0, H1, H2)
        if M < 1e-9: return 0
        return D / M
        
    best_score = score_P_full(best_P)
    
    rng = np.random.default_rng(42)
    all_indices = np.arange(100)
    
    candidates = [] # List of (score, support_indices, weights)
    
    # Phase 1: Heuristic Pre-filtering
    # Generate many random supports, score them with heuristic, keep top ones
    Ks = [10, 12, 15, 20, 25, 30]
    top_heuristic_supports = []
    
    for K in Ks:
        n_samples = 8000
        for _ in range(n_samples):
            if time.time() - start_time > timeout - safety_margin:
                break
            supp = rng.choice(all_indices, size=K, replace=False)
            h_score = heuristic_score(supp)
            top_heuristic_supports.append((h_score, supp))
            
        top_heuristic_supports.sort(key=lambda x: x[0], reverse=True)
        # Keep top 30 for this K
        for h_s, supp in top_heuristic_supports[:30]:
            w, sc = optimize_weights_for_support(supp)
            if sc > best_score:
                P_new = np.zeros(100)
                P_new[supp] = w
                P_new = P_new.reshape(10,10)
                best_score = sc
                best_P = P_new
            candidates.append((sc, supp.copy(), w.copy()))
            
    # Phase 2: Standard Random Search (reduced)
    for K in Ks:
        if time.time() - start_time > timeout - safety_margin:
            break
        n_attempts = 800
        for _ in range(n_attempts):
            if time.time() - start_time > timeout - safety_margin:
                break
            supp = rng.choice(all_indices, size=K, replace=False)
            w, sc = optimize_weights_for_support(supp)
            if sc > best_score:
                P_new = np.zeros(100)
                P_new[supp] = w
                P_new = P_new.reshape(10,10)
                best_score = sc
                best_P = P_new
            candidates.append((sc, supp.copy(), w.copy()))
                
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 3: Local Search on Supports (Greedy Swap)
    top_candidates = candidates[:80]
    
    for _ in range(30): # 30 rounds of local search
        if time.time() - start_time > timeout - safety_margin:
            break
        new_candidates = []
        for sc, supp, w in top_candidates:
            # Try swapping each point in supp with a point not in supp
            K = len(supp)
            # Randomly select 15 points to swap out
            swap_out_indices = rng.choice(K, size=min(15, K), replace=False)
            for idx_out in swap_out_indices:
                if time.time() - start_time > timeout - safety_margin:
                    break
                point_out = supp[idx_out]
                available = np.setdiff1d(all_indices, supp)
                if len(available) == 0: continue
                point_in = rng.choice(available)
                
                new_supp = supp.copy()
                new_supp[idx_out] = point_in
                
                w_new, sc_new = optimize_weights_for_support(new_supp)
                if sc_new > sc:
                    candidates.append((sc_new, new_supp.copy(), w_new.copy()))
                    new_candidates.append((sc_new, new_supp.copy(), w_new.copy()))
                    if sc_new > best_score:
                        P_new = np.zeros(100)
                        P_new[new_supp] = w_new
                        best_P = P_new.reshape(10,10)
                        best_score = sc_new
        
        # Update top candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:80]
        
    # Phase 4: High Precision Polish on the absolute best support
    if best_score > 0:
        non_zero = np.where(best_P.ravel() > 1e-9)[0]
        if len(non_zero) > 0:
            w_best = best_P.ravel()[non_zero]
            w_best /= w_best.sum()
            
            supp_D = D_inv[non_zero]
            supp_H0 = H0_inv[non_zero]
            supp_H1 = H1_inv[non_zero]
            supp_H2 = H2_inv[non_zero]
            
            def obj_and_grad_fine(w):
                D, gD = calc_grad_entropy(supp_D, w, D_bins)
                H0, gH0 = calc_grad_entropy(supp_H0, w, H0_bins)
                H1, gH1 = calc_grad_entropy(supp_H1, w, H1_bins)
                H2, gH2 = calc_grad_entropy(supp_H2, w, H2_bins)
                Hs = np.array([H0, H1, H2])
                beta = 50.0
                scaled = beta * Hs
                max_scaled = np.max(scaled)
                exp_scaled = np.exp(scaled - max_scaled)
                sum_exp = np.sum(exp_scaled)
                M = (max_scaled + np.log(sum_exp)) / beta
                if M < 1e-9: return -1e9, np.zeros_like(w)
                weights_max = exp_scaled / sum_exp
                gM = weights_max[0]*gH0 + weights_max[1]*gH1 + weights_max[2]*gH2
                ratio = D / M
                grad_ratio = (M * gD - D * gM) / (M * M)
                return -ratio, -grad_ratio

            try:
                res = opt.minimize(
                    lambda x: obj_and_grad_fine(x)[0],
                    w_best,
                    jac=lambda x: obj_and_grad_fine(x)[1],
                    method='SLSQP',
                    bounds=[(0, 1)] * len(non_zero),
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                                  'jac': lambda x: np.ones(len(non_zero))}],
                    options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
                )
                w_polished = res.x
                w_polished = np.clip(w_polished, 0, None)
                s = w_polished.sum()
                if s > 0:
                    w_polished /= s
                    sc_polished = score_weights(w_polished, supp_D, supp_H0, supp_H1, supp_H2)
                    if sc_polished > best_score:
                        P_polished = np.zeros(100)
                        P_polished[non_zero] = w_polished
                        best_P = P_polished.reshape(10, 10)
                        best_score = sc_polished
            except Exception:
                pass

    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()
