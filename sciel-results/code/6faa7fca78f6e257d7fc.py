import random
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def score_raw(h):
        """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
        h = np.asarray(h, dtype=float)
        if len(h) == 0:
            return 0.0
        h1 = np.sum(h)
        if h1 <= 0:
            return 0.0
        h2 = np.sum(h * h)
        c = np.convolve(h, h)
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        return h2 / (h1 * cmax)

    def optimize_epigraph(h_init, max_iter=500, ftol=1e-8):
        """
        Optimize h using SLSQP with epigraph formulation.
        Variables: x = [h_0, ..., h_{N-1}, M]
        Minimize M subject to sum(h)=1, M >= c_k for all k, h >= 0
        """
        def obj(x):
            return x[N]

        def obj_grad(x):
            grad = np.zeros(N + 1)
            grad[N] = 1.0
            return grad

        def constraint_sum(x):
            return np.sum(x[:N]) - 1.0

        def constraint_sum_grad(x):
            grad = np.zeros(N + 1)
            grad[:N] = 1.0
            return grad

        def constraints_autoconv(x):
            h = x[:N]
            M = x[N]
            c = np.convolve(h, h)
            return M - c

        def constraints_autoconv_grad(x):
            h = x[:N]
            grad = np.zeros((2 * N - 1, N + 1))
            for k in range(2 * N - 1):
                derivs = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        derivs[i] = 2.0 * h[j]
                grad[k, :N] = -derivs
                grad[k, N] = 1.0
            return grad

        h_init = np.asarray(h_init, dtype=float)
        c_init = np.convolve(h_init, h_init)
        M_init = np.max(c_init)
        x0 = np.concatenate([h_init, [M_init]])

        bounds = [(0.0, None)] * N + [(0.0, None)]

        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
            {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
        ]

        try:
            res = minimize(
                obj,
                x0,
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )

            if res.success or res.fun < np.max(c_init):
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0
                h_res = np.clip(h_res, 0, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0

                ratio_res = score_raw(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        return None, 0.0

    def generate_gaussian_mixtures(max_time):
        """
        Generate diverse Gaussian mixture initializations.
        Each mixture has K components with parameters (center, width, weight).
        This reduces the search to a low-dimensional continuous space.
        """
        candidates = []
        current_time = time.time()
        
        # Single Gaussian: 2 params (center, width)
        for center in np.linspace(5, N - 5, 15):
            if time.time() - current_time > max_time:
                break
            for width in np.linspace(1.5, N / 3, 10):
                if time.time() - current_time > max_time:
                    break
                h = np.exp(-0.5 * ((np.arange(N) - center) / width) ** 2)
                if np.sum(h) > 1e-10:
                    h = h / np.sum(h)
                    candidates.append(h)
        
        # Two Gaussians: 4 params (c1, w1, c2, w2) with weights determined by relative amplitudes
        if time.time() - current_time < max_time * 0.5:
            for c1 in np.linspace(5, N // 2 - 3, 8):
                if time.time() - current_time > max_time * 0.7:
                    break
                for c2 in np.linspace(N // 2 + 3, N - 5, 8):
                    if time.time() - current_time > max_time * 0.8:
                        break
                    for w1 in np.linspace(1.5, N / 4, 5):
                        if time.time() - current_time > max_time * 0.9:
                            break
                        for w2 in np.linspace(1.5, N / 4, 5):
                            if time.time() - current_time > max_time:
                                break
                            h = np.exp(-0.5 * ((np.arange(N) - c1) / w1) ** 2) + \
                                np.exp(-0.5 * ((np.arange(N) - c2) / w2) ** 2)
                            if np.sum(h) > 1e-10:
                                h = h / np.sum(h)
                                candidates.append(h)
        
        # Three Gaussians: 6 params, more diverse
        if time.time() - current_time < max_time * 0.3:
            for c1 in np.linspace(4, N // 3, 5):
                if time.time() - current_time > max_time * 0.7:
                    break
                for c2 in np.linspace(N // 3 + 2, 2 * N // 3 - 2, 5):
                    if time.time() - current_time > max_time * 0.8:
                        break
                    for c3 in np.linspace(2 * N // 3 + 2, N - 4, 5):
                        if time.time() - current_time > max_time * 0.9:
                            break
                        for w in np.linspace(2, N / 4, 4):
                            if time.time() - current_time > max_time:
                                break
                            h = np.exp(-0.5 * ((np.arange(N) - c1) / w) ** 2) + \
                                np.exp(-0.5 * ((np.arange(N) - c2) / w) ** 2) + \
                                np.exp(-0.5 * ((np.arange(N) - c3) / w) ** 2)
                            if np.sum(h) > 1e-10:
                                h = h / np.sum(h)
                                candidates.append(h)
        
        # Random smooth functions (low-pass filtered)
        if time.time() - current_time < max_time * 0.2:
            for _ in range(20):
                if time.time() - current_time > max_time:
                    break
                h = np.random.rand(N)
                # Apply low-pass filter
                kernel = np.ones(7) / 7
                h = np.convolve(h, kernel, mode='same')
                h = np.maximum(h, 0)
                if np.sum(h) > 1e-10:
                    h = h / np.sum(h)
                    candidates.append(h)
        
        return candidates

    def find_best_sidon_set(max_time):
        """
        Finds a large subset of {0, ..., N-1} that is as close to a Sidon set as possible.
        """
        best_set = []
        best_size = 0
        best_collisions = float('inf')
        best_score = -1
        
        num_trials = 5000
        current_time = time.time()
        
        for _ in range(num_trials):
            if time.time() - current_time > max_time:
                break
            
            support = []
            used_sums = set()
            
            first = random.randint(0, N - 1)
            support.append(first)
            used_sums.add(2 * first)
            
            candidates = [i for i in range(N) if i != first]
            random.shuffle(candidates)
            
            for cand in candidates:
                if time.time() - current_time > max_time:
                    break
                    
                new_collisions = 0
                new_sums_to_add = []
                
                for s in support:
                    pair_sum = cand + s
                    if pair_sum in used_sums:
                        new_collisions += 1
                    else:
                        new_sums_to_add.append(pair_sum)
                
                if new_collisions == 0:
                    support.append(cand)
                    for ns in new_sums_to_add:
                        used_sums.add(ns)
                elif new_collisions <= 1 and len(support) < 6:
                    support.append(cand)
                    for ns in new_sums_to_add:
                        used_sums.add(ns)
            
            if len(support) > best_size or (len(support) == best_size):
                total_collisions = 0
                sums_count = {}
                for i in range(len(support)):
                    for j in range(i, len(support)):
                        s_val = support[i] + support[j]
                        sums_count[s_val] = sums_count.get(s_val, 0) + 1
                for count in sums_count.values():
                    if count > 1:
                        total_collisions += count - 1
                        
                current_score = len(support) * 1000 - total_collisions
                if current_score > best_score:
                    best_score = current_score
                    best_size = len(support)
                    best_collisions = total_collisions
                    best_set = support[:]

        return best_set, best_size

    best_h = None
    best_score_val = 0.0

    # Phase 1: Generate diverse initializations from both families
    # Allocate time: 40% Gaussian mixtures, 30% Sidon search, 30% optimization
    gaussian_time = 50.0
    sidon_time = 40.0
    opt_time = 45.0

    # Generate Gaussian mixture candidates
    g_start = time.time()
    gaussian_candidates = generate_gaussian_mixtures(gaussian_time)
    g_elapsed = time.time() - g_start
    
    # Evaluate and keep top Gaussian candidates
    scored_gaussians = []
    for h in gaussian_candidates:
        s = score_raw(h)
        scored_gaussians.append((s, h))
    scored_gaussians.sort(key=lambda x: -x[0])
    
    # Keep top 20 for refinement
    top_gaussians = [h for _, h in scored_gaussians[:20]]
    
    for h in top_gaussians:
        s = score_raw(h)
        if s > best_score_val:
            best_score_val = s
            best_h = h.copy()

    # Phase 2: Sidon-like discrete support search
    if time.time() < deadline - opt_time - 5:
        s_start = time.time()
        sidon_set, sidon_size = find_best_sidon_set(sidon_time)
        s_elapsed = time.time() - s_start
        
        if sidon_size > 0:
            h_sidon = np.zeros(N)
            h_sidon[sidon_set] = 1.0 / len(sidon_set)
            s_val = score_raw(h_sidon)
            
            if s_val > best_score_val:
                best_score_val = s_val
                best_h = h_sidon.copy()

    # Phase 3: Refinement using SLSQP on best candidates
    remaining_time = deadline - time.time()
    
    # Create a pool of candidates to refine
    refine_pool = []
    
    # Add top Gaussian candidates
    for h in top_gaussians[:10]:
        refine_pool.append(h)
    
    # Add Sidon candidate if found
    if best_h is not None and len(np.nonzero(best_h)[0]) < 15:
        refine_pool.append(best_h.copy())
    
    # Add some random sparse candidates
    for _ in range(5):
        K = random.randint(4, 8)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        refine_pool.append(h_rand)
    
    # Refine each candidate with SLSQP
    for h_init in refine_pool:
        if time.time() > deadline - 3:
            break
        
        elapsed = time.time() - start_time
        time_left = deadline - time.time()
        max_iter = min(500, int(time_left * 20))
        
        if max_iter < 50:
            break
            
        h_opt, s_opt = optimize_epigraph(h_init, max_iter=max_iter, ftol=1e-9)
        if h_opt is not None and s_opt > best_score_val:
            best_score_val = s_opt
            best_h = h_opt.copy()

    # Final Polish if time allows
    if best_h is not None and time.time() < deadline - 2:
        h_polished, s_polished = optimize_epigraph(best_h, max_iter=300, ftol=1e-10)
        if h_polished is not None and s_polished > best_score_val:
            best_score_val = s_polished
            best_h = h_polished.copy()

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N
        best_score_val = score_raw(best_h)

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
# EVOLVE-BLOCK-END
