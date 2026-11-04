"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def autoconv(h):
        return np.convolve(h, h, mode='full')

    def score_fn(h):
        """Return the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf). Higher is better."""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h * h)
        if norm2_sq < 1e-30:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-30:
            return 0.0
        ac = autoconv(h)
        ac_max = np.max(np.abs(ac))
        if ac_max < 1e-30:
            return 0.0
        return norm2_sq / (norm1 * ac_max)

    def _smooth_max(x, beta=50.0):
        """Smooth approximation of max using log-sum-exp."""
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

    def _smooth_ratio(h, beta=50.0):
        """Smooth surrogate for the ratio using smooth max for the denominator."""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h * h)
        if norm2_sq < 1e-30:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-30:
            return 0.0
        ac = autoconv(h)
        ac_abs = np.abs(ac)
        ac_max_smooth = _smooth_max(ac_abs, beta)
        if ac_max_smooth < 1e-30:
            return 0.0
        return norm2_sq / (norm1 * ac_max_smooth)

    def _optimize_complex_amplitudes(support, max_iter=300):
        """Optimize complex amplitudes on a given support using smooth objective."""
        m = len(support)
        if m < 2:
            return None, 0.0
        
        # Initialize with random phases
        rng = np.random.RandomState(42)
        phases = rng.uniform(0, 2 * np.pi, m)
        amps = np.ones(m) / np.sqrt(m)
        
        h = np.zeros(N)
        h[support] = amps * np.cos(phases)
        h = np.maximum(h, 0.0)
        best_h = h.copy()
        best_score = score_fn(h)
        
        try:
            from scipy.optimize import minimize
        except ImportError:
            return best_h, best_score
        
        def _params_to_h(params):
            """Convert parameter vector to height sequence."""
            a = np.maximum(params[:m], 0.0)
            p = params[m:]
            h_tmp = np.zeros(N)
            h_tmp[support] = a * np.cos(p)
            return np.maximum(h_tmp, 0.0)
        
        def _neg_smooth_obj(params):
            h_tmp = _params_to_h(params)
            return -_smooth_ratio(h_tmp, beta=50.0)
        
        x0 = np.concatenate([amps, phases])
        bounds = [(0.0, 1.0)] * m + [(0.0, 2 * np.pi)] * m
        
        try:
            res = minimize(_neg_smooth_obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success:
                h_opt = _params_to_h(res.x)
                s_opt = score_fn(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
        
        # Also try pure real amplitudes (no phases)
        def _neg_real_obj(params):
            a = np.maximum(params, 0.0)
            h_tmp = np.zeros(N)
            h_tmp[support] = a
            return -_smooth_ratio(h_tmp, beta=50.0)
        
        x0_real = amps.copy()
        bounds_real = [(0.0, 1.0)] * m
        
        try:
            res = minimize(_neg_real_obj, x0_real, method='L-BFGS-B', bounds=bounds_real,
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[support] = np.maximum(res.x, 0.0)
                s_opt = score_fn(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
        
        return best_h, best_score

    def _get_sidon_set(n, k, seed=42):
        """Find a subset of {0,...,n-1} of size k with distinct pairwise sums (Sidon set)."""
        rng = np.random.RandomState(seed)
        best_set = None
        best_size = 0
        
        for attempt in range(300):
            if time.time() > deadline - 15:
                break
            s = [0]
            sums = {0}
            for _ in range(k - 1):
                found = False
                for cand in range(1, n):
                    if cand in s:
                        continue
                    new_sums = set(sums)
                    ok = True
                    for idx in s:
                        ns = cand + idx
                        if ns in new_sums:
                            ok = False
                            break
                        new_sums.add(ns)
                    if ok:
                        ns = cand + cand
                        if ns in new_sums:
                            ok = False
                        else:
                            new_sums.add(ns)
                    if ok:
                        s.append(cand)
                        sums = new_sums
                        found = True
                        break
                if not found:
                    break
            if len(s) > best_size:
                best_size = len(s)
                best_set = s[:]
                if best_size >= k:
                    break
        
        if best_set is None:
            best_set = list(range(k))
        return best_set[:k]

    best_h = None
    best_score = 0.0

    # Phase 1: Try Sidon sets with complex amplitude optimization
    for k in range(3, 12):
        if time.time() > deadline - 40:
            break
        
        sidon = _get_sidon_set(N, k)
        if len(sidon) < 3:
            continue
        
        h_opt, s_opt = _optimize_complex_amplitudes(sidon)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # Try perturbations
        for trial in range(5):
            if time.time() > deadline - 40:
                break
            support_pert = sidon[:]
            idx_rep = np.random.randint(0, len(support_pert))
            old_val = support_pert[idx_rep]
            new_val = None
            for offset in [-3, -2, -1, 1, 2, 3, -4, 4]:
                cand = old_val + offset
                if 0 <= cand < N and cand not in support_pert:
                    new_val = cand
                    break
            if new_val is not None:
                support_pert[idx_rep] = new_val
            h_opt, s_opt = _optimize_complex_amplitudes(support_pert)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 2: Try dense Gaussian signals as fallback
    if time.time() < deadline - 20:
        x = np.linspace(-0.25, 0.25, N)
        for alpha in [10, 20, 50, 100, 200, 500, 1000, 2000]:
            if time.time() > deadline - 15:
                break
            h = np.exp(-alpha * x * x)
            h = h / np.sqrt(np.sum(h * h))
            s = score_fn(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()

    # Phase 3: Coordinate descent polish with smooth objective
    if best_h is not None and time.time() < deadline - 5:
        h_curr = best_h.copy()
        curr_score = score_fn(h_curr)

        for step in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 3:
                break
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for i in range(N):
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-14:
                        h_curr = h_trial
                        curr_score = s
                        improved = True
                        continue
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-14:
                        h_curr = h_trial
                        curr_score = s
                        improved = True

        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()

    if best_h is None:
        return [1.0] * N

    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
