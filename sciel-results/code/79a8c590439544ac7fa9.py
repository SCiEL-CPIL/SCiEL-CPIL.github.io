import numpy as np
from scipy.optimize import minimize
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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

    def optimize_full_h(h_init, max_iter=500, ftol=1e-8):
        """
        Optimize h directly using SLSQP with epigraph formulation.
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

        h_init = np.asarray(h_init, dtype=float)
        s = np.sum(h_init)
        if s <= 0:
            h_init = np.ones(N) / N
            s = 1.0
        h_init = h_init / s
        
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
            
            if res.success or res.fun < M_init:
                h_res = res.x[:N]
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

    def coordinate_descent_polish(h_init, max_iters=500, tol=1e-10):
        """
        Refine h using coordinate descent to maximize the ratio.
        We minimize the peak autoconvolution M while keeping sum(h)=1.
        For each coordinate i, we adjust h_i to minimize the new peak,
        while scaling the other coordinates to maintain sum=1.
        """
        h = np.array(h_init, dtype=float)
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s == 0:
            return h, score_raw(h)
        h /= s
        
        best_ratio = score_raw(h)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            for i in range(N):
                # Try to adjust h_i
                # Let h_others = h with h_i set to 0
                # Sum_others = s - h_i
                # If we set h_i to v, we scale others by (1-v)/Sum_others
                # New h: h_new[i] = v, h_new[j] = h[j] * (1-v)/Sum_others for j!=i
                
                h_i_old = h[i]
                sum_others = 1.0 - h_i_old
                
                if sum_others < 1e-12:
                    continue
                
                # Binary search or line search for optimal v in [0, 1]
                # We want to minimize peak autoconvolution of h_new.
                # Let's try a few candidate values for v
                # v = 0, 0.5*h_i_old, h_i_old, 1.5*h_i_old (clipped to [0,1])
                
                candidates = [0.0, h_i_old * 0.5, h_i_old, h_i_old * 1.5, 1.0]
                # Add some random perturbations around h_i_old
                for _ in range(5):
                    candidates.append(max(0, min(1, h_i_old + random.uniform(-0.05, 0.05))))
                
                best_v = h_i_old
                best_m = np.inf
                
                for v in candidates:
                    if v < 0 or v > 1:
                        continue
                    if v == h_i_old:
                        continue
                    
                    h_new = np.array(h)
                    h_new[i] = v
                    # Scale others
                    if v < 1.0:
                        scale = (1.0 - v) / sum_others
                        for j in range(N):
                            if j != i:
                                h_new[j] = h[j] * scale
                    else:
                        # All mass on i
                        h_new[:] = 0
                        h_new[i] = 1.0
                    
                    c = np.convolve(h_new, h_new)
                    m = np.max(c)
                    
                    if m < best_m:
                        best_m = m
                        best_v = v
                
                if best_v != h_i_old:
                    h[i] = best_v
                    sum_others = 1.0 - best_v
                    if sum_others > 1e-12:
                        scale = (1.0 - best_v) / (1.0 - h_i_old) # Wait, the scaling logic above was for a specific v.
                        # Let's re-normalize properly to be safe.
                        # Actually, the candidate evaluation computed h_new explicitly.
                        # Let's just re-assign h to the best h_new found.
                        # This requires storing the best h_new.
                        pass # We will refactor this loop to store best_h_new
                    
            # Refactored CD step for correctness
            improved = False
            for i in range(N):
                h_i_old = h[i]
                sum_others = 1.0 - h_i_old
                
                if sum_others < 1e-12:
                    continue
                
                best_v = h_i_old
                best_h_candidate = h.copy()
                best_m = np.convolve(h, h).max()
                
                candidates = [0.0, h_i_old * 0.5, h_i_old * 1.5, 1.0]
                for _ in range(5):
                    candidates.append(max(0, min(1, h_i_old + random.uniform(-0.1, 0.1))))
                
                for v in candidates:
                    if v < 0 or v > 1:
                        continue
                    if abs(v - h_i_old) < 1e-9:
                        continue
                    
                    h_new = h.copy()
                    h_new[i] = v
                    if v < 1.0 and sum_others > 1e-12:
                        scale = (1.0 - v) / sum_others
                        h_new[i] = v
                        h_new_other = h.copy()
                        h_new_other[i] = 0
                        h_new = h_new_other * scale
                        h_new[i] = v
                    elif v == 1.0:
                        h_new = np.zeros(N)
                        h_new[i] = 1.0
                    else:
                        continue
                    
                    c = np.convolve(h_new, h_new)
                    m = np.max(c)
                    
                    if m < best_m - 1e-12:
                        best_m = m
                        best_v = v
                        best_h_candidate = h_new
                
                if best_v != h_i_old:
                    h = best_h_candidate
                    improved = True
            
            if not improved:
                break
            
            # Update best ratio
            current_ratio = score_raw(h)
            if current_ratio > best_ratio:
                best_ratio = current_ratio

        return h, best_ratio

    def find_good_sidon_support(target_size):
        """Greedy construction of a near-Sidon set to minimize pairwise sum collisions."""
        if target_size <= 0:
            return []
        
        support = [0]
        used_sums = {0: 1}
        
        for _ in range(target_size - 1):
            best_candidate = None
            best_score = -1
            
            for candidate in range(N):
                if candidate in support:
                    continue
                
                # Count new collisions
                new_collisions = 0
                for s in support:
                    pair_sum = candidate + s
                    if pair_sum in used_sums:
                        new_collisions += 1
                
                # We want to minimize collisions, so maximize (target - collisions)
                score = target_size - new_collisions
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate is not None:
                support.append(best_candidate)
                for s in support:
                    pair_sum = best_candidate + s
                    used_sums[pair_sum] = used_sums.get(pair_sum, 0) + 1
        
        return support

    best_h = None
    best_score = 0.0

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = optimize_full_h(h_uniform, max_iter=300)
    if h_opt is not None and s_opt > best_score:
        best_score = s_opt
        best_h = h_opt.copy()

    # 2. Try Sidon-like structured supports
    for target_k in [5, 6, 7, 8, 10, 12]:
        if time.time() > deadline - 30:
            break
        
        support = find_good_sidon_support(target_k)
        if len(support) >= 3:
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0 / len(support)
            h_opt, s_opt = optimize_full_h(h_sidon, max_iter=300)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 3. Try random sparse starts with varying support sizes
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        K = random.randint(3, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = optimize_full_h(h_rand, max_iter=200)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 4. Polish the best solution found using SLSQP with zeroed-out small values
    if best_h is not None and time.time() < deadline - 15:
        h_polish_init = best_h.copy()
        # Zero out very small values to simplify the problem for the final SLSQP pass
        h_polish_init[h_polish_init < 1e-5] = 0.0
        s = np.sum(h_polish_init)
        if s > 0:
            h_polish_init /= s
        
        h_polished, s_polished = optimize_full_h(h_polish_init, max_iter=1000, ftol=1e-10)
        if h_polished is not None and s_polished > best_score:
            best_score = s_polished
            best_h = h_polished.copy()

    # 5. Final Coordinate Descent Polish
    if best_h is not None and time.time() < deadline - 5:
        h_cd, s_cd = coordinate_descent_polish(best_h, max_iters=200)
        if s_cd > best_score:
            best_score = s_cd
            best_h = h_cd.copy()

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
