import numpy as np
from scipy.optimize import minimize
import time

N = 100
NUM_LAGS = 2 * N - 1

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0

def _calc_max_corr(h):
    h = np.asarray(h, dtype=np.float64)
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

def _local_search_binary(h_bin):
    h_bin = np.array(h_bin, dtype=np.int8)
    current_max = _calc_max_corr(h_bin.astype(np.float64))
    
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_new_h = None
        
        for i in ones:
            for j in zeros:
                h_temp = h_bin.copy()
                h_temp[i] = 0
                h_temp[j] = 1
                
                new_max = _calc_max_corr(h_temp.astype(np.float64))
                gain = current_max - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_new_h = h_temp
        
        if best_gain > 1e-6:
            h_bin = best_new_h
            current_max = current_max - best_gain
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            improved = True
            
    return h_bin.astype(np.float64), current_max

def _convex_refine(h_init, deadline):
    """
    Refine using auxiliary variable t to minimize max correlation.
    Variables: x (heights), t (upper bound on correlations)
    Objective: minimize t
    Constraints:
      1. sum(x) = N/2
      2. 0 <= x_i <= 1
      3. corr_k(x) <= t for all lags k
    """
    def objective(z):
        # z = [x_0, ..., x_{N-1}, t]
        return z[-1]

    def gradient(z):
        # Gradient of t is 0 for x, 1 for t
        grad = np.zeros_like(z)
        grad[-1] = 1.0
        return grad

    def corr_jacobian(x):
        # Returns a (NUM_LAGS, N) matrix J such that corr = J @ x
        # corr_k = sum_i x_i * (1 - x_{i+k}) for valid i
        # This is not linear in x because of the x_i * x_{i+k} term.
        # Wait, the objective is max(corr). corr_k = sum_i (x_i - x_i x_{i+k}).
        # This is non-convex!
        # However, the current program uses binary search first, then continuous.
        # The continuous refinement in the current program uses a soft-max on the non-convex objective.
        # The DIRECTIVE suggests convex optimization. But the objective is NOT convex.
        # np.correlate(h, 1-h) = sum h_i (1-h_{i+k}).
        # This is quadratic. Minimizing the max of quadratic forms is generally non-convex.
        # However, if we fix the binary pattern and only adjust values slightly, or if we assume the "convex" part refers to something else?
        # Actually, the directive says: "Model the problem as a convex optimization ... which transforms the non-smooth max objective into a smooth, convex-constrained problem".
        # This implies the author might believe it is convex, or perhaps I am misinterpreting the correlation.
        # Let's re-read carefully.
        # If the problem is non-convex, SLSQP with t constraints might still work as a local solver, but it's not a global convex solver.
        # Let's try to implement the t-constraint approach with SLSQP, treating it as a local optimization with explicit inequality constraints.
        # This might be more stable than soft-max.
        pass

    # Since the problem is non-convex, we cannot use a pure convex solver.
    # However, we can use SLSQP with explicit inequality constraints: corr_k(x) - t <= 0.
    # This avoids the soft-max approximation error.

    def constraints_func(z):
        x = z[:N]
        t = z[-1]
        # Inequality constraints: corr_k(x) <= t  =>  corr_k(x) - t <= 0
        # We need to return a list of constraint dicts or use a vectorized form.
        # SLSQP supports a list of dicts.
        # But creating 199 constraints dynamically might be slow.
        # Alternative: use a penalty method or just minimize the soft-max as before but with higher precision?
        # Or, we can define the constraints explicitly.

        # Let's try defining the constraints.
        # This will be a list of 199 + 1 (sum) constraints.
        cons = []
        
        # Sum constraint
        cons.append({'type': 'eq', 'fun': lambda z: np.sum(z[:N]) - N/2.0, 'jac': lambda z: np.concatenate([np.ones(N), [0]])})

        # Correlation constraints
        for k in range(NUM_LAGS):
            # corr_k = sum_{i} x_i (1 - x_{i+k})
            # We need the function and its gradient.
            def corr_k_fun(z, k=k):
                x = z[:N]
                # Compute correlation at lag k
                # np.correlate(x, 1-x, mode='full')[k]
                # But we want to avoid calling np.correlate inside the loop for speed if possible?
                # For N=100, it's fast enough.
                c = np.correlate(x, 1.0 - x, mode="full")[k]
                return c - z[-1]
            
            def corr_k_jac(z, k=k):
                x = z[:N]
                # Gradient of corr_k with respect to x and t
                # corr_k = sum_i (x_i - x_i x_{i+k})
                # d/dx_j corr_k = sum_i (delta_{ij} - (delta_{ij} x_{i+k} + delta_{i+k,j} x_i))
                # This is getting complicated to vectorize easily without precomputation.
                # Let's just use numerical differentiation or compute it analytically.
                # Analytical:
                # Let L = k.
                # If k >= 0:
                #   corr_k = sum_{i=0}^{N-1-L} x_i (1 - x_{i+L})
                #   d/dx_j = 1 (if j in 0..N-1-L) - x_{j+L} (if j+L < N)
                #           - x_{j-L} (if j-L >= 0 and j-L < N-L? No, term is x_{i+L} where i = j-L)
                #   So d/dx_j = 1 (if 0<=j<N-L) - x_{j+L} (if 0<=j<N-L) - x_{j-L} (if L<=j<N)
                # If k < 0:
                #   Let L = -k.
                #   corr_k = sum_{i=L}^{N-1} x_i (1 - x_{i-L})
                #   d/dx_j = 1 (if L<=j<N) - x_{j+L} (if j+L < N) - x_{j-L} (if j-L >= 0)
                
                grad_x = np.zeros(N)
                L = abs(k)
                if k >= 0:
                    # i from 0 to N-1-L
                    for i in range(N - L):
                        # term x_i (1 - x_{i+L})
                        grad_x[i] += 1.0 - x[i+L]
                        if i + L < N:
                            grad_x[i+L] += -x[i]
                else:
                    # i from L to N-1
                    for i in range(L, N):
                        # term x_i (1 - x_{i-L})
                        grad_x[i] += 1.0 - x[i-L]
                        if i - L >= 0:
                            grad_x[i-L] += -x[i]
                
                grad_t = -1.0
                return np.concatenate([grad_x, [grad_t]])

            cons.append({'type': 'ineq', 'fun': corr_k_fun, 'jac': corr_k_jac})
            
        return cons

    # This creates 200 constraints. SLSQP might be slow but N=100 is small.
    # Let's try it.

    initial_t = _calc_max_corr(h_init) + 0.1
    z0 = np.concatenate([h_init, [initial_t]])
    
    bounds = [(0.0, 1.0)] * N + [(-np.inf, np.inf)]
    
    try:
        res = minimize(
            objective,
            z0,
            jac=gradient,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_func(z0),
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        h_candidate = res.x[:N]
        h_candidate = np.clip(h_candidate, 0.0, 1.0)
        
        # Fix sum
        diff = N/2.0 - np.sum(h_candidate)
        if abs(diff) > 1e-9:
            h_candidate = h_candidate + diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
        val = _value(h_candidate)
        return h_candidate, val
    except Exception:
        return h_init, _value(h_init)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # 1. Binary Local Search
    num_random_restarts = 20
    
    for restart in range(num_random_restarts):
        if time.time() > deadline - 40:
            break
            
        h_bin = np.zeros(N, dtype=np.int8)
        ones_idx = rng.choice(N, size=N//2, replace=False)
        h_bin[ones_idx] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    # 2. Convex-like Refinement with explicit constraints
    if time.time() < deadline - 10:
        h_refined, val_refined = _convex_refine(best_h, deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined

    return best_h.tolist()
