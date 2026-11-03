import numpy as np
from scipy.optimize import minimize
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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

    def soft_max(c, tau=0.01):
        """Smooth approximation of max(c) using log-sum-exp."""
        c_max = np.max(c)
        if c_max <= 0:
            return 0.0
        # Shift for numerical stability
        c_shifted = c / c_max
        log_sum = np.log(np.sum(np.exp(c_shifted / tau)))
        return c_max * (1.0 + tau * log_sum)

    def optimize_with_soft_max(h_init, max_iter=500, ftol=1e-10, tau=0.01):
        """
        Optimize h using SLSQP with smooth max surrogate.
        Minimize M subject to M >= soft_max(c_k) and sum(h)=1, h>=0.
        """
        h_init = np.asarray(h_init, dtype=float)
        
        def objective(x):
            h = x[:N]
            M = x[N]
            return M

        def objective_grad(x):
            grad = np.zeros(N + 1)
            grad[N] = 1.0
            return grad

        def constraint_sum(x):
            return np.sum(x[:N]) - 1.0

        def constraint_sum_grad(x):
            grad = np.zeros(N + 1)
            grad[:N] = 1.0
            return grad

        def constraint_soft_max(x):
            h = x[:N]
            M = x[N]
            c = np.convolve(h, h)
            sm = soft_max(c, tau)
            return M - sm

        def constraint_soft_max_grad(x):
            h = x[:N]
            c = np.convolve(h, h)
            c_max = np.max(c)
            if c_max <= 0:
                return np.zeros(N + 1)
            
            # Gradient of soft_max wrt c
            c_shifted = c / c_max
            exp_vals = np.exp(c_shifted / tau)
            sum_exp = np.sum(exp_vals)
            d_sm_dc = (exp_vals / sum_exp) / tau  # d(soft_max)/dc_k
            
            # Gradient of c_k wrt h_i
            grad = np.zeros(N + 1)
            for k in range(2 * N - 1):
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        grad[i] += d_sm_dc[k] * 2.0 * h[j]
            grad[N] = 1.0
            return grad

        c_init = np.convolve(h_init, h_init)
        M_init = soft_max(c_init, tau) + 1e-6
        x0 = np.concatenate([h_init, [M_init]])
        
        bounds = [(0.0, None)] * N + [(0.0, None)]
        
        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
            {'type': 'ineq', 'fun': constraint_soft_max, 'jac': constraint_soft_max_grad}
        ]
        
        try:
            res = minimize(
                objective, x0, method='SLSQP', jac=objective_grad,
                bounds=bounds, constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )
            
            if res.success:
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

    def optimize_epigraph(h_init, max_iter=500, ftol=1e-10):
        """
        Optimize h using SLSQP with exact epigraph formulation.
        Minimize M subject to M >= c_k for all k, sum(h)=1, h>=0.
        """
        h_init = np.asarray(h_init, dtype=float)
        
        def objective(x):
            return x[N]

        def objective_grad(x):
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

        c_init = np.convolve(h_init, h_init)
        M_init = np.max(c_init) + 1e-6
        x0 = np.concatenate([h_init, [M_init]])
        
        bounds = [(0.0, None)] * N + [(0.0, None)]
        
        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
            {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
        ]
        
        try:
            res = minimize(
                objective, x0, method='SLSQP', jac=objective_grad,
                bounds=bounds, constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )
            
            if res.success:
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

    def block_coordinate_refine(h, num_rounds=20, tol=1e-8):
        """
        Refine h by adjusting symmetric pairs around the dominant peak.
        """
        h = h.copy()
        c = np.convolve(h, h)
        peak_idx = np.argmax(c)
        
        best_score = score_raw(h)
        
        for round_num in range(num_rounds):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adjusting pairs symmetric around peak
            for offset in range(1, N):
                if time.time() > deadline - 5:
                    break
                
                i = peak_idx - offset
                j = peak_idx + offset
                
                if i < 0 or j >= 2 * N - 1:
                    continue
                
                # These correspond to convolution indices, not h indices
                # We need to map back to h indices
                # c[k] = sum_i h[i]*h[k-i]
                # For symmetric adjustment, we adjust h[a] and h[b] where a+b = k
                
                # Find h indices that contribute to c[i] and c[j]
                # This is complex, so let's use a simpler approach:
                # Adjust small perturbations on h directly
                
                # Try moving mass between two h positions
                # Find the two largest h values and try redistributing
                h_sorted_idx = np.argsort(h)[::-1]
                if len(h_sorted_idx) < 2:
                    continue
                
                a = h_sorted_idx[0]
                b = h_sorted_idx[1]
                
                # Try different transfer amounts
                for frac in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
                    h_test = h.copy()
                    transfer = frac * h[a]
                    if transfer > h[a]:
                        continue
                    h_test[a] -= transfer
                    h_test[b] += transfer
                    h_test = np.clip(h_test, 0, None)
                    s = np.sum(h_test)
                    if s > 0:
                        h_test = h_test / s
                    
                    new_score = score_raw(h_test)
                    if new_score > best_score + tol:
                        h = h_test
                        best_score = new_score
                        improved = True
                        break
            
            if not improved:
                # Try random small perturbations
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    h_test = h.copy()
                    # Random perturbation
                    idx1 = random.randint(0, N - 1)
                    idx2 = random.randint(0, N - 1)
                    if idx1 == idx2:
                        continue
                    transfer = 0.01 * h[idx1]
                    if transfer > h[idx1]:
                        continue
                    h_test[idx1] -= transfer
                    h_test[idx2] += transfer
                    h_test = np.clip(h_test, 0, None)
                    s = np.sum(h_test)
                    if s > 0:
                        h_test = h_test / s
                    
                    new_score = score_raw(h_test)
                    if new_score > best_score + tol:
                        h = h_test
                        best_score = new_score
                        improved = True
                        break
            
            if not improved:
                break
        
        return h, best_score

    def polish(h, max_iter=1000, ftol=1e-12):
        """Final high-precision polish pass."""
        if time.time() > deadline - 10:
            return h, score_raw(h)
        
        # First try epigraph with tight tolerance
        h_opt, s_opt = optimize_epigraph(h, max_iter=max_iter, ftol=ftol)
        if h_opt is not None and s_opt > score_raw(h):
            h = h_opt
        
        # Then try soft-max with small tau
        for tau in [0.005, 0.001]:
            if time.time() > deadline - 5:
                break
            h_opt, s_opt = optimize_with_soft_max(h, max_iter=500, ftol=ftol, tau=tau)
            if h_opt is not None and s_opt > score_raw(h):
                h = h_opt
        
        # Block coordinate refinement
        h, s = block_coordinate_refine(h, num_rounds=30, tol=1e-10)
        
        return h, score_raw(h)

    best_h = None
    best_score = 0.0

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = optimize_epigraph(h_uniform)
    if h_opt is not None and s_opt > best_score:
        best_score = s_opt
        best_h = h_opt.copy()

    # 2. Try structured sparse starts - Sidon-like sets
    # Generate various structured supports
    structured_supports = []
    
    # Powers of 2
    support = []
    val = 1
    while val < N:
        support.append(val)
        val *= 2
    if len(support) >= 3:
        structured_supports.append(support)
    
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    structured_supports.append(primes[:10])
    
    # Fibonacci-like
    fib = [1, 2]
    while fib[-1] < N:
        fib.append(fib[-1] + fib[-2])
    structured_supports.append(fib[:10])
    
    # Arithmetic progressions with different steps
    for step in [3, 4, 5, 7]:
        support = list(range(0, N, step))
        if len(support) >= 3:
            structured_supports.append(support)
    
    # Random Sidon-like: try to find sets with few collisions
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        K = random.randint(5, 12)
        support = set()
        attempts = 0
        while len(support) < K and attempts < 1000:
            idx = random.randint(0, N - 1)
            # Check if adding idx creates too many collisions
            new_sums = set()
            for i in support:
                new_sums.add(i + idx)
                new_sums.add(idx + i)
            # Count collisions with existing sums
            existing_sums = set()
            for i in support:
                for j in support:
                    if i != j:
                        existing_sums.add(i + j)
            collisions = len(new_sums & existing_sums)
            if collisions <= 1:
                support.add(idx)
            attempts += 1
        if len(support) >= 3:
            structured_supports.append(list(support))
    
    for support in structured_supports:
        if time.time() > deadline - 15:
            break
        support = [s for s in support if 0 <= s < N]
        if len(support) < 2:
            continue
        h_struct = np.zeros(N)
        h_struct[support] = 1.0 / len(support)
        
        # Try both epigraph and soft-max
        h_opt, s_opt = optimize_epigraph(h_struct)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        if time.time() > deadline - 10:
            break
        h_opt, s_opt = optimize_with_soft_max(h_struct, tau=0.01)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 3. Random sparse starts
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        K = random.randint(4, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        # Randomly choose optimizer
        if random.random() < 0.5:
            h_opt, s_opt = optimize_epigraph(h_rand)
        else:
            h_opt, s_opt = optimize_with_soft_max(h_rand, tau=random.choice([0.005, 0.01, 0.02]))
        
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 4. If we still have time, do more random restarts
    while time.time() < deadline - 20:
        K = random.randint(3, 18)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        # Add some noise to weights
        if random.random() < 0.3:
            noise = np.random.dirichlet(np.ones(K) * 0.5)
            h_rand[support] = noise
        
        if random.random() < 0.5:
            h_opt, s_opt = optimize_epigraph(h_rand, max_iter=300)
        else:
            h_opt, s_opt = optimize_with_soft_max(h_rand, tau=0.01, max_iter=300)
        
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 5. Final polish on best candidate
    if best_h is not None:
        best_h, best_score = polish(best_h)

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
