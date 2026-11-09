# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    V_m1 = X - Y
    V_0 = X
    V_1 = X + Y
    V_2 = X + 2*Y

    bin_idx_m1 = (V_m1 + 9).flatten()
    bin_idx_0 = V_0.flatten()
    bin_idx_1 = V_1.flatten()
    bin_idx_2 = V_2.flatten()

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    def compute_ratio_from_logP(logP_flat, soft_max_beta=100.0):
        z = logP_flat - np.max(logP_flat)
        P_flat = np.exp(z)
        sum_P = np.sum(P_flat)
        if sum_P == 0:
            return 0.0
        P_flat = P_flat / sum_P
        
        counts_m1 = np.bincount(bin_idx_m1, weights=P_flat, minlength=num_bins_m1)
        counts_0 = np.bincount(bin_idx_0, weights=P_flat, minlength=num_bins_0)
        counts_1 = np.bincount(bin_idx_1, weights=P_flat, minlength=num_bins_1)
        counts_2 = np.bincount(bin_idx_2, weights=P_flat, minlength=num_bins_2)
        
        def entropy(counts):
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))
            
        H_m1 = entropy(counts_m1)
        H_0 = entropy(counts_0)
        H_1 = entropy(counts_1)
        H_2 = entropy(counts_2)
        
        # Soft max for smooth optimization
        H_vals = np.array([H_0, H_1, H_2])
        if soft_max_beta > 0:
            M = (1.0 / soft_max_beta) * np.log(np.sum(np.exp(soft_max_beta * H_vals)))
        else:
            M = np.max(H_vals)
            
        if M < 1e-9:
            return 0.0
            
        return H_m1 / M

    def objective(logP_flat, soft_max_beta=100.0):
        r = compute_ratio_from_logP(logP_flat, soft_max_beta)
        return -r

    start_time = time.time()
    time_limit = 610.0
    
    best_score = 0.0
    best_P = np.ones((10, 10)) / 100.0
    best_logP = np.zeros(100)
    
    rng = np.random.default_rng(42)
    
    init_points = []
    init_points.append(np.zeros(100))
    
    for _ in range(30):
        init_points.append(rng.normal(0, 1, 100))
        
    # Structured patterns
    for y in range(10):
        for x in range(10):
            if (x+y) % 2 == 0:
                p = np.zeros(100)
                p[y*10+x] = 8.0
                init_points.append(p)
                
    for _ in range(30):
        p = np.zeros(100)
        idx = rng.choice(100, size=15, replace=False)
        p[idx] = 8.0
        init_points.append(p)
    
    # Diagonal patterns
    for d in range(-9, 10):
        p = np.zeros(100)
        for y in range(10):
            x = y + d
            if 0 <= x < 10:
                p[y*10+x] = 8.0
        if np.sum(p) > 0:
            init_points.append(p)
    
    # Anti-diagonal patterns
    for s in range(0, 19):
        p = np.zeros(100)
        for y in range(10):
            x = s - y
            if 0 <= x < 10:
                p[y*10+x] = 8.0
        if np.sum(p) > 0:
            init_points.append(p)

    # Phase 1: Broad search with soft max
    for i, init in enumerate(init_points):
        if time.time() - start_time > time_limit - 60:
            break
            
        try:
            res = minimize(lambda lp: objective(lp, 100.0), init, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-9})
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_logP = res.x.copy()
                z = res.x - np.max(res.x)
                P_flat = np.exp(z)
                P_flat = P_flat / np.sum(P_flat)
                best_P = P_flat.reshape(10, 10)
        except Exception:
            pass

    # Phase 2: Polish with exact max (soft_max_beta=0)
    if time.time() - start_time < time_limit - 30:
        try:
            res = minimize(lambda lp: objective(lp, 0.0), best_logP, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-11})
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_logP = res.x.copy()
                z = res.x - np.max(res.x)
                P_flat = np.exp(z)
                P_flat = P_flat / np.sum(P_flat)
                best_P = P_flat.reshape(10, 10)
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if time.time() - start_time < time_limit - 10:
        try:
            res = minimize(lambda lp: objective(lp, 0.0), best_logP, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-13})
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_logP = res.x.copy()
                z = res.x - np.max(res.x)
                P_flat = np.exp(z)
                P_flat = P_flat / np.sum(P_flat)
                best_P = P_flat.reshape(10, 10)
        except Exception:
            pass

    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s
        
    return best_P

# EVOLVE-BLOCK-END
