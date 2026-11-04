import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    def score(h):
        s = np.sum(h)
        if s == 0:
            return 0
        h_norm = h / s
        q = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        m = np.max(conv)
        if m == 0:
            return 0
        return q / m

    def soft_max(conv, t=50.0):
        """Smooth approximation of max via log-sum-exp"""
        return np.log(np.sum(np.exp(t * conv)) / len(conv)) / t

    def smooth_score(h):
        s = np.sum(h)
        if s < 1e-15:
            return 0
        h_norm = h / s
        q = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        m = soft_max(conv, t=100.0)
        if m <= 0:
            return 0
        return q / m

    def smooth_score_neg(h):
        return -smooth_score(h)

    best_score = 1.0
    best_h = np.ones(N) / N

    # Try structured initial candidates
    # 1. Sidon-like sets (quadratic residues mod p, or known small Sidon sets)
    sidon_sets = [
        [0, 1, 4, 10, 18, 23, 28, 37, 44],  # Known Sidon set
        [0, 1, 4, 10, 18, 23, 28, 37, 44, 49],
        [0, 2, 5, 11, 17, 23, 29, 35, 41, 47],
        [0, 1, 4, 10, 18, 23, 28, 37, 44],
        [0, 3, 7, 12, 18, 25, 33, 42],
        [0, 1, 5, 11, 19, 29, 41],
    ]
    for sidon in sidon_sets:
        if len(sidon) > N:
            continue
        h = np.zeros(N)
        k = len(sidon)
        h[sidon] = 1.0 / k
        sc = score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # 2. Geometric progression indices
    for ratio in [2, 3]:
        idx = []
        i = 0
        while i < N:
            idx.append(i)
            i = i * ratio + 1 if i > 0 else 1
        if len(idx) >= 3:
            h = np.zeros(N)
            k = len(idx)
            h[idx] = 1.0 / k
            sc = score(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()

    # 3. Random sparse search with diverse parameters
    np.random.seed(42)
    num_trials = 3000
    for i in range(num_trials):
        if time.time() - start_time > deadline - 10:
            break
        k = np.random.randint(3, 18)
        idx = np.random.choice(N, k, replace=False)
        
        # Equal weights
        h = np.zeros(N)
        h[idx] = 1.0 / k
        sc = score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        
        # Random weights
        if time.time() - start_time > deadline - 10:
            break
        w = np.random.rand(k)
        w = w / np.sum(w)
        h = np.zeros(N)
        h[idx] = w
        sc = score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        
        # Power-law weights
        if time.time() - start_time > deadline - 10:
            break
        alpha = np.random.uniform(0.5, 3.0)
        w = np.power(np.arange(1, k+1), -alpha)
        w = w / np.sum(w)
        h = np.zeros(N)
        h[idx] = w
        sc = score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # 4. Local optimization polish on best candidate
    if time.time() - start_time < deadline - 5:
        # Work on the support of best_h
        support = np.where(best_h > 1e-10)[0]
        if len(support) >= 2:
            k = len(support)
            # Optimize weights on this support
            def objective(w):
                if time.time() - start_time > deadline - 3:
                    return 0
                h = np.zeros(N)
                w = np.maximum(w, 0)
                s = np.sum(w)
                if s < 1e-15:
                    return 0
                w = w / s
                h[support] = w
                return -smooth_score(h)
            
            w0 = best_h[support].copy()
            w0 = w0 / np.sum(w0)
            
            try:
                res = minimize(objective, w0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                     'adaptive': True})
                if res.success or res.fun < -best_score:
                    w_opt = np.maximum(res.x, 0)
                    s = np.sum(w_opt)
                    if s > 1e-15:
                        h_opt = np.zeros(N)
                        h_opt[support] = w_opt / s
                        sc_opt = score(h_opt)
                        if sc_opt > best_score:
                            best_score = sc_opt
                            best_h = h_opt
            except Exception:
                pass

        # Try a few more local optimizations from random perturbations
        if time.time() - start_time < deadline - 5:
            for _ in range(3):
                if time.time() - start_time > deadline - 5:
                    break
                # Perturb best_h
                h_pert = best_h.copy()
                noise = np.random.normal(0, 0.05, N)
                h_pert = np.maximum(h_pert + noise, 0)
                s = np.sum(h_pert)
                if s < 1e-15:
                    continue
                h_pert = h_pert / s
                
                support = np.where(h_pert > 1e-10)[0]
                if len(support) < 2:
                    continue
                k = len(support)
                
                def objective2(w):
                    if time.time() - start_time > deadline - 3:
                        return 0
                    h = np.zeros(N)
                    w = np.maximum(w, 0)
                    s2 = np.sum(w)
                    if s2 < 1e-15:
                        return 0
                    w = w / s2
                    h[support] = w
                    return -smooth_score(h)
                
                w0 = h_pert[support].copy()
                w0 = w0 / np.sum(w0)
                
                try:
                    res = minimize(objective2, w0, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                         'adaptive': True})
                    if res.success or res.fun < -best_score:
                        w_opt = np.maximum(res.x, 0)
                        s_opt = np.sum(w_opt)
                        if s_opt > 1e-15:
                            h_opt = np.zeros(N)
                            h_opt[support] = w_opt / s_opt
                            sc_opt = score(h_opt)
                            if sc_opt > best_score:
                                best_score = sc_opt
                                best_h = h_opt
                except Exception:
                    pass

    return best_h.tolist()
# EVOLVE-BLOCK-END
