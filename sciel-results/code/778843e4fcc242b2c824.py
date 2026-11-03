import numpy as np
import time
from scipy.optimize import differential_evolution
import warnings

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def optimize_amplitudes_de(support, max_iter=1000, popsize=20):
        """Optimize amplitudes for a fixed support using Differential Evolution."""
        k = len(support)
        if k == 0:
            return np.zeros(N), 0.0
        
        def objective(ampl):
            h = np.zeros(N)
            h[support] = ampl
            return -score(h)
        
        bounds = [(0, 1)] * k
        
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=max_iter, 
                popsize=popsize,
                tol=1e-9,
                seed=42,
                polish=True
            )
            best_ampl = result.x
            h = np.zeros(N)
            h[support] = best_ampl
            s_val = score(h)
            return h, s_val
        except Exception:
            h = np.zeros(N)
            h[support] = 1.0 / k
            return h, score(h)

    def find_best_support(k, iterations=50):
        """Find a good k-element Sidon set using greedy + local search."""
        best_support = None
        best_mc = np.inf
        
        for _ in range(20):
            S = set()
            for _ in range(k):
                best_cand = None
                best_mc_cand = np.inf
                for j in range(N):
                    if j in S:
                        continue
                    S_test = list(S) + [j]
                    h_tmp = np.zeros(N)
                    h_tmp[S_test] = 1.0
                    conv = np.convolve(h_tmp, h_tmp, mode='full')
                    mc = np.max(conv)
                    if mc < best_mc_cand:
                        best_mc_cand = mc
                        best_cand = j
                if best_cand is not None:
                    S.add(best_cand)
            
            S = sorted(S)
            h_tmp = np.zeros(N)
            h_tmp[S] = 1.0
            conv = np.convolve(h_tmp, h_tmp, mode='full')
            mc = np.max(conv)
            
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                for i in range(len(S)):
                    for j in range(N):
                        if j in S:
                            continue
                        S_new = sorted(S[:i] + [j] + S[i+1:])
                        h_tmp = np.zeros(N)
                        h_tmp[S_new] = 1.0
                        conv = np.convolve(h_tmp, h_tmp, mode='full')
                        mc_new = np.max(conv)
                        if mc_new < mc:
                            mc = mc_new
                            S = S_new
                            improved = True
                            break
                    if improved:
                        break
            
            if mc < best_mc:
                best_mc = mc
                best_support = S

        return best_support

    best_h = np.zeros(N)
    best_score = 0.0

    for k in range(3, 15):
        if time.time() > deadline - 20.0:
            break
            
        S = find_best_support(k)
        if S is None:
            continue
            
        h_opt, s = optimize_amplitudes_de(S)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()

    rng = np.random.RandomState(42)
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        k = rng.randint(3, 12)
        S = sorted(rng.choice(N, k, replace=False).tolist())
        h_opt, s = optimize_amplitudes_de(S, max_iter=200, popsize=15)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()

    return list(best_h)
