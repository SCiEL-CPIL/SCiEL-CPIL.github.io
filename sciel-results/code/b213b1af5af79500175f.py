import numpy as np
import scipy.optimize as opt
import time

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Generate a good Sidon set (B2 set) of size K
    # Greedy construction for B2 set
    def generate_sidon_set(N, K):
        S = [0]
        diffs = set()
        for i in range(1, N):
            valid = True
            new_diffs = set()
            for j in S:
                d = abs(i - j)
                if d in diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                S.append(i)
                diffs.update(new_diffs)
                if len(S) == K:
                    break
        return S

    best_h = None
    best_score = -1.0

    # Try different Sidon set sizes
    for K in [5, 6, 7, 8]:
        S = generate_sidon_set(N, K)
        if len(S) < K:
            continue
        
        h_init = np.zeros(N)
        for idx in S:
            h_init[idx] = 1.0
        h_init /= np.sum(h_init)
        
        # Optimize weights on the support
        # We only optimize the non-zero elements to save time and maintain sparsity
        support_idx = np.where(h_init > 0)[0]
        x0 = h_init[support_idx]
        
        def objective(x):
            h = np.zeros(N)
            h[support_idx] = x
            # Ensure non-negative and sum=1 is handled by constraints/bounds
            l2_sq = np.sum(x**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv <= 1e-15:
                return 0.0
            return -l2_sq / max_conv

        def constraints_sum(x):
            return np.sum(x) - 1.0

        bounds = [(1e-6, 1.0)] * K
        cons = [{'type': 'eq', 'fun': constraints_sum}]

        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = np.zeros(N)
                h_candidate[support_idx] = res.x
                h_candidate /= np.sum(h_candidate)
                s = _score_raw(h_candidate)
                if s > best_score:
                    best_score = s
                    best_h = h_candidate
        except Exception:
            pass

        if time.time() > deadline - 10:
            break

    # 2. Try a "smooth" optimization from a random start to see if dense solutions are better
    # (Unlikely, but checks the hypothesis)
    if time.time() < deadline - 20:
        h_rand = np.random.rand(N)
        h_rand /= np.sum(h_rand)
        
        def objective_full(h):
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv <= 1e-15:
                return 0.0
            return -l2_sq / max_conv

        bounds_full = [(1e-6, 1.0)] * N
        cons_full = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]

        try:
            res = opt.minimize(objective_full, h_rand, method='SLSQP', bounds=bounds_full, constraints=cons_full, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success:
                h_candidate = res.x
                h_candidate /= np.sum(h_candidate)
                s = _score_raw(h_candidate)
                if s > best_score:
                    best_score = s
                    best_h = h_candidate
        except Exception:
            pass

    if best_h is None:
        # Fallback
        best_h = np.ones(N) / N

    return list(best_h)
