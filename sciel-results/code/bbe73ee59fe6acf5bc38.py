import numpy as np
from scipy.optimize import minimize
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def score_raw(h):
        """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
        h = np.asarray(h, dtype=float)
        if len(h) == 0:
            return 0.0
        h1 = np.sum(h)
        if h1 <= 0:
            return 0.0
        h2 = np.sum(h * h)
        # Linear convolution
        c = np.convolve(h, h)
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        return h2 / (h1 * cmax)

    def sidon_score(support):
        """Score a Sidon set: for uniform weights on support, R = K if all sums distinct."""
        K = len(support)
        if K == 0:
            return 0.0
        # Count collisions in pairwise sums
        sums = {}
        for i in range(K):
            for j in range(K):
                s = support[i] + support[j]
                sums[s] = sums.get(s, 0) + 1
        # Max collision count determines the peak
        max_count = max(sums.values()) if sums else 1
        # For uniform weights 1/K on K points:
        # ||h||_2^2 = K*(1/K)^2 = 1/K
        # ||h||_1 = 1
        # ||h*h||_inf = max_count / K^2
        # R = (1/K) / (1 * max_count/K^2) = K / max_count
        return K / max_count

    def find_sidon_set_greedy(target_size, max_iter=1000):
        """Greedy construction of a Sidon set."""
        support = []
        used_sums = set()
        remaining = list(range(N))
        random.shuffle(remaining)
        
        for _ in range(max_iter):
            if len(support) >= target_size:
                break
            best_pos = None
            best_conflicts = float('inf')
            for pos in remaining:
                # Check how many new sum collisions this would cause
                new_conflicts = 0
                for s in support:
                    if (pos + s) in used_sums:
                        new_conflicts += 1
                    if (s + pos) in used_sums:
                        new_conflicts += 1
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_pos = pos
                if new_conflicts == 0:
                    break
            
            if best_pos is not None:
                support.append(best_pos)
                remaining.remove(best_pos)
                # Add new sums
                for s in support:
                    used_sums.add(support[-1] + s)
                    used_sums.add(s + support[-1])
        
        return sorted(support)

    def optimize_full_h(h_init, max_iter=500, ftol=1e-8):
        """
        Optimize h directly using SLSQP.
        Objective: minimize M
        Constraints:
        1. sum(h) == 1
        2. M - c_k >= 0 for all k
        """
        # Variables: h (length N) and M (scalar)
        # x = [h_0, ..., h_{N-1}, M]
        
        def neg_ratio_obj(x):
            return x[N]

        def neg_ratio_obj_grad(x):
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
            # Gradient of (M - c_k) wrt x
            # d/dh_i (M - c_k) = - d/dh_i c_k
            # c_k = sum_j h_j h_{k-j}
            # d c_k / d h_i = 2 h_{k-i} (with h_out_of_bounds=0)
            
            grad = np.zeros((2*N - 1, N + 1))
            for k in range(2*N - 1):
                derivs = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        derivs[i] = 2.0 * h[j]
                grad[k, :N] = -derivs
                grad[k, N] = 1.0
            return grad

        # Initial guess
        h_init = np.asarray(h_init, dtype=float)
        c_init = np.convolve(h_init, h_init)
        M_init = np.max(c_init)
        x0 = np.concatenate([h_init, [M_init]])
        
        # Bounds
        bounds = [(0.0, None)] * N + [(0.0, None)]
        
        # Constraints dict
        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
            {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
        ]
        
        try:
            res = minimize(
                neg_ratio_obj, 
                x0, 
                method='SLSQP', 
                jac=neg_ratio_obj_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )
            
            if res.success:
                h_res = res.x[:N]
                # Normalize to ensure sum is exactly 1
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0
                # Clip negative values due to numerical errors
                h_res = np.clip(h_res, 0, None)
                h_res = h_res / np.sum(h_res)
                
                ratio_res = score_raw(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        return None, 0.0

    best_h = None
    best_score = 0.0

    # 1. Systematically search for large Sidon-like sets
    # Try different target sizes and multiple random restarts
    for target_k in [12, 11, 10, 9, 8]:
        if time.time() > deadline - 15:
            break
        
        best_sidon_support = None
        best_sidon_score = 0.0
        
        for attempt in range(30):
            if time.time() > deadline - 15:
                break
            
            support = find_sidon_set_greedy(target_k, max_iter=200)
            s = sidon_score(support)
            if s > best_sidon_score:
                best_sidon_score = s
                best_sidon_support = support
        
        if best_sidon_support is not None:
            # Create uniform weight vector on this support
            h_sidon = np.zeros(N)
            h_sidon[best_sidon_support] = 1.0 / len(best_sidon_support)
            
            # Score the uniform Sidon set
            s_uniform = score_raw(h_sidon)
            if s_uniform > best_score:
                best_score = s_uniform
                best_h = h_sidon.copy()
            
            # Polish with optimizer
            h_opt, s_opt = optimize_full_h(h_sidon, max_iter=300)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 2. Try structured constructions
    # Powers of 2 style
    for base in [2, 3, 4]:
        if time.time() > deadline - 10:
            break
        support = []
        val = 0
        while val < N:
            support.append(val)
            val = val * base + 1
        if len(support) > 2:
            h_geom = np.zeros(N)
            h_geom[support] = 1.0 / len(support)
            s = score_raw(h_geom)
            if s > best_score:
                best_score = s
                best_h = h_geom.copy()
            
            h_opt, s_opt = optimize_full_h(h_geom, max_iter=200)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 3. Try random supports with varying sizes
    for K in [7, 8, 9, 10, 11, 12]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            support = np.random.choice(N, K, replace=False)
            h_rand = np.zeros(N)
            h_rand[support] = 1.0 / K
            
            s = score_raw(h_rand)
            if s > best_score:
                best_score = s
                best_h = h_rand.copy()
            
            # Only optimize if the raw score is promising
            if s > best_score * 0.9:
                h_opt, s_opt = optimize_full_h(h_rand, max_iter=200)
                if h_opt is not None and s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()

    # 4. Final polish on the best solution found
    if best_h is not None and time.time() < deadline - 5:
        h_opt, s_opt = optimize_full_h(best_h, max_iter=500, ftol=1e-10)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N
        best_score = score_raw(best_h)

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
