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

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = optimize_full_h(h_uniform)
    if h_opt is not None and s_opt > best_score:
        best_score = s_opt
        best_h = h_opt.copy()

    # 2. Try sparse random starts
    # The idea is that sparse vectors might have lower convolution peaks relative to their L2 norm
    # if the support is chosen well (e.g., Sidon-like).
    # We will generate random sparse vectors and optimize them.
    
    # Also try some structured sparse starts
    # e.g. powers of 2, or random subsets
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random support size
        # Smaller supports tend to have higher L2 norm for fixed L1, but also higher peak?
        # R = ||h||_2^2 / (||h||_1 ||h*h||_inf)
        # If h is uniform on K points:
        # ||h||_2^2 = K * (1/K)^2 = 1/K
        # ||h||_1 = 1
        # ||h*h||_inf = 1/K (if no collisions in sums)
        # R = (1/K) / (1 * 1/K) = 1
        # Wait, if no collisions, c[k] = h[i]h[j] = 1/K^2. Max is 1/K^2.
        # Then R = (1/K) / (1/K^2) = K.
        # So if we can find a Sidon set of size K, the score should be K.
        # The max Sidon set size in [0, 49] is roughly sqrt(49) = 7.
        # So max score from pure Sidon sets is around 7-10.
        
        # Let's try random supports of size 5 to 15.
        K = random.randint(3, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        # Optimize this random start
        h_opt, s_opt = optimize_full_h(h_rand)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 3. Try a "geometric" progression or other structured sets
    # Sometimes structured sets avoid collisions better than random.
    for base in [2, 3]:
        if time.time() > deadline - 10:
            break
        support = []
        val = 1
        while val < N:
            support.append(val)
            val *= base
        if len(support) > 0:
            h_geom = np.zeros(N)
            h_geom[support] = 1.0 / len(support)
            h_opt, s_opt = optimize_full_h(h_geom)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 4. If we still have time, do more random restarts with different K
    while time.time() < deadline - 5:
        K = random.randint(2, 20)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        # Maybe add some noise to the weights?
        # No, uniform on support is a good start for SLSQP to find local optimum.
        
        h_opt, s_opt = optimize_full_h(h_rand, max_iter=200)
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
