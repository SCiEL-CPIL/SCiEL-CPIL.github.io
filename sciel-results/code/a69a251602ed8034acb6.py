import numpy as np
import time
import random
from scipy.optimize import minimize

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

def find_sidon_set(N, K, attempts=500):
    """
    Attempts to find a Sidon set of size K in [0, N).
    Returns the set if found, None otherwise.
    """
    for _ in range(attempts):
        candidates = list(range(N))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for cand in candidates:
            if len(S) == K:
                break
            
            conflict = False
            new_sums = set()
            
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
                
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums:
                        conflict = True
                        break
                    if sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == K:
            return S
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def autoconv(h):
        return np.convolve(h, h, mode='full')

    # Objective: Minimize M
    # Variables: x = [h_0, ..., h_{N-1}, M]
    def objective(x):
        return x[N]

    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad

    # Constraints
    def constraint_sum(x):
        return np.sum(x[:N]) - 1.0

    def constraint_sum_grad(x):
        grad = np.zeros(N + 1)
        grad[:N] = 1.0
        return grad

    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        c = autoconv(h)
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

    bounds = [(0.0, None)] * N + [(1e-10, None)]
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]

    # Initial guesses
    initial_guesses = []

    # 1. Uniform
    h_uni = np.ones(N) / N
    c_uni = autoconv(h_uni)
    M_uni = np.max(c_uni)
    initial_guesses.append(np.concatenate([h_uni, [M_uni]]))

    # 2. Sidon sets
    # Try to find a good Sidon set. Larger K means sparser, potentially higher ratio.
    # For Sidon set of size K with equal weights 1/K:
    # ||h||_2^2 = 1/K
    # ||h*h||_inf = 2/K^2 (if no sum collisions)
    # Ratio = (1/K) / (2/K^2) = K/2
    # So larger K is better. Max K for N=50 is around sqrt(N) ~ 7-8? 
    # Actually Sidon sets can be larger. Let's try K from 15 down to 5.
    best_sidon_h = None
    for K in range(15, 4, -1):
        if time.time() > deadline - 30.0:
            break
        S = find_sidon_set(N, K, attempts=200)
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0 / K
            c = autoconv(h)
            M = np.max(c)
            initial_guesses.append(np.concatenate([h, [M]]))
            if best_sidon_h is None:
                best_sidon_h = h.copy()
            break

    # 3. Random sparse guesses
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 10.0:
            break
        K_rand = np.random.randint(5, 15)
        support = np.random.choice(N, K_rand, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K_rand
        c_rand = autoconv(h_rand)
        M_rand = np.max(c_rand)
        initial_guesses.append(np.concatenate([h_rand, [M_rand]]))

    # Run SLSQP on each guess
    best_h = None
    best_ratio = 0.0

    for x0 in initial_guesses:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                jac=objective_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-12, 'iprint': 0}
            )
            
            if res.success:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    continue
                h_res = np.clip(h_res, 0, None)
                if np.sum(h_res) > 0:
                    h_res = h_res / np.sum(h_res)
                
                ratio_res = _score_raw(h_res)
                if ratio_res > best_ratio:
                    best_ratio = ratio_res
                    best_h = h_res.copy()
                    
        except Exception:
            continue

    # If no SLSQP succeeded, fall back to the best initial guess
    if best_h is None:
        # Use the best initial guess by ratio
        best_ratio = 0.0
        for x0 in initial_guesses:
            h = x0[:N]
            s = np.sum(h)
            if s > 0:
                h = h / s
            ratio = _score_raw(h)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h.copy()
        
        if best_h is None:
            best_h = np.ones(N) / N
            best_ratio = _score_raw(best_h)

    # --- Block-Coordinate Descent Refinement (from current program) ---
    def block_coordinate_refine(h_start, max_iter=50):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 10:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            
            # For each position i, try to optimize h[i] and h[k_star - i] (if valid)
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                    
                hi = h[i]
                hj = h[j]
                
                t_min = -hi
                t_max = hj
                
                if t_max - t_min < 1e-15:
                    continue
                    
                best_t = 0.0
                best_local_score = _score_raw(h)
                
                for t in np.linspace(t_min * 0.8, t_max * 0.8, 10):
                    h_test = h.copy()
                    h_test[i] += t
                    h_test[j] -= t
                    if np.any(h_test < -1e-15):
                        continue
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    s_test = _score_raw(h_test)
                    if s_test > best_local_score:
                        best_local_score = s_test
                        best_t = t
                        
                if abs(best_t) > 1e-10:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    if np.sum(h) > 1e-15:
                        h /= np.sum(h)
                    improved = True
                    
        return h

    # Apply block-coordinate refinement to best solution
    if time.time() < deadline - 15:
        h_refined = block_coordinate_refine(best_h, max_iter=30)
        s_refined = _score_raw(h_refined)
        if s_refined > best_ratio:
            best_ratio = s_refined
            best_h = h_refined.copy()

    # Final polish: Run SLSQP again on the best_h with tighter tolerance
    if time.time() < deadline - 2.0:
        c_best = autoconv(best_h)
        M_best = np.max(c_best)
        x_best = np.concatenate([best_h, [M_best]])
        try:
            res = minimize(
                objective, 
                x_best, 
                method='SLSQP', 
                jac=objective_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-14, 'iprint': 0}
            )
            if res.success:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    h_res = best_h
                h_res = np.clip(h_res, 0, None)
                if np.sum(h_res) > 0:
                    h_res = h_res / np.sum(h_res)
                
                ratio_res = _score_raw(h_res)
                if ratio_res > best_ratio:
                    best_ratio = ratio_res
                    best_h = h_res.copy()
        except Exception:
            pass

    # Final cleanup
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
