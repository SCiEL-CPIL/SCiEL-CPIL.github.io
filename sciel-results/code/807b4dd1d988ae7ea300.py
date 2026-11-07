import numpy as np
from scipy.optimize import minimize
import time

N = 100

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
    """
    Local search on binary vectors with exactly N/2 ones.
    Tries swaps (one 1 with one 0) to preserve the count of ones.
    """
    h_bin = np.array(h_bin, dtype=np.int8)
    current_max = _calc_max_corr(h_bin.astype(np.float64))
    
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_new_h = None
        
        # Try swapping a 1 at i with a 0 at j
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
        
        if best_gain > 1e-6 and best_new_h is not None:
            h_bin = best_new_h
            current_max = current_max - best_gain
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            improved = True
            
    return h_bin.astype(np.float64), current_max

def _convex_refine(h_start, deadline):
    """
    Refine using convex optimization: minimize t subject to C_k <= t for all lags,
    0 <= h_i <= 1, sum(h) = N/2.
    """
    best_h = h_start.copy()
    best_val = _value(best_h)
    
    # Variables: h[0:N], t
    # Objective: t
    # Constraints:
    #   h_i >= 0, h_i <= 1 (bounds)
    #   sum(h) - N/2 = 0 (equality)
    #   C_k - t <= 0 for each lag k (inequality)
    
    # Correlation C_k = sum_i h_i * (1 - h_{i+k}) for valid i
    # np.correlate(h, 1-h, mode="full") gives 2N-1 values
    # C[lag] where lag ranges from -(N-1) to (N-1)
    # In 'full' mode, index k corresponds to lag k - (N-1)
    # So C_full[j] = sum_i h[i] * (1-h[i + j - (N-1)]) for valid i
    # Actually, np.correlate(a, v, mode='full') computes:
    # c[k] = sum_i a[i] * v[i + k - (len(v)-1)]
    # For a=h, v=1-h:
    # c[k] = sum_i h[i] * (1 - h[i + k - (N-1)])
    # The max over all k is what we want to minimize.
    
    def objective(x):
        return x[N]  # t
    
    # Gradient of objective
    def jac(x):
        g = np.zeros(N + 1)
        g[N] = 1.0
        return g
    
    bounds = [(0.0, 1.0)] * N + [(-np.inf, np.inf)]
    
    # Equality constraint: sum(h) - N/2 = 0
    def eq_constraint(x):
        return np.sum(x[:N]) - N / 2.0
    
    def eq_jac(x):
        g = np.zeros(N + 1)
        g[:N] = 1.0
        return g
    
    # Inequality constraints: C_k - t <= 0
    # We need to compute C_k for all k and enforce C_k - t <= 0
    
    def corr_values(h):
        corr = np.correlate(h, 1.0 - h, mode="full")
        return corr
    
    # Number of correlation values
    num_corr = 2 * N - 1
    
    # We'll use trust-constr method which handles many constraints well
    # But it might be slow. Let's try SLSQP with a subset or all constraints.
    
    # For SLSQP, we need to provide all inequality constraints.
    # With 2N-1 = 199 constraints, SLSQP might struggle.
    
    # Alternative: Use the fact that only the max matters.
    # We can use a smooth approximation or just constrain the top few.
    
    # Let's try trust-constr first.
    from scipy.optimize import LinearConstraint, NonlinearConstraint
    
    # Define the nonlinear constraint function
    def constraint_func(x):
        h = np.clip(x[:N], 0.0, 1.0)
        t = x[N]
        corr = np.correlate(h, 1.0 - h, mode="full")
        # We need corr[k] - t <= 0, i.e., t - corr[k] >= 0
        return t - corr
    
    def constraint_jac(x):
        h = np.clip(x[:N], 0.0, 1.0)
        t = x[N]
        # Jacobian of (t - corr[k]) with respect to x
        # d/dh_i (t - corr[k]) = -d/dh_i corr[k]
        # corr[k] = sum_j h[j] * (1 - h[j + k - (N-1)])
        # This is complex. Let's use finite differences or just provide the function.
        
        # For trust-constr, we can provide the Jacobian.
        # Let's compute it analytically.
        # corr[k] = sum_j h[j] * (1 - h[j + k - (N-1)])
        # Let shift = k - (N-1)
        # corr[k] = sum_j h[j] - sum_j h[j] * h[j + shift]
        # d/dh_i corr[k] = 1 - (h[i + shift] + h[i - shift]) if valid, else 1 or 0
        
        J = np.zeros((num_corr, N + 1))
        for k in range(num_corr):
            shift = k - (N - 1)
            # d/dh_i corr[k]
            for i in range(N):
                dc = 1.0
                if 0 <= i + shift < N:
                    dc -= h[i + shift]
                if 0 <= i - shift < N:
                    dc -= h[i - shift]
                J[k, i] = -dc  # derivative of (t - corr[k]) w.r.t h_i
            J[k, N] = 1.0  # derivative of (t - corr[k]) w.r.t t
        
        return J
    
    # This Jacobian computation is O(N * num_corr) = O(N^2) per iteration, which is fine.
    
    # Lower bound for constraint is 0, upper is inf
    lower = np.zeros(num_corr)
    upper = np.full(num_corr, np.inf)
    
    # Initial t estimate
    h_init = h_start.copy()
    corr_init = np.correlate(h_init, 1.0 - h_init, mode="full")
    t_init = np.max(corr_init) + 0.1
    
    x0 = np.concatenate([h_init, [t_init]])
    
    try:
        res = minimize(
            objective, x0,
            method='trust-constr',
            jac=jac,
            bounds=bounds,
            constraints=[
                NonlinearConstraint(constraint_func, 0.0, np.inf, jac=constraint_jac),
                {'type': 'eq', 'fun': eq_constraint, 'jac': eq_jac}
            ],
            options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        
        h_candidate = np.clip(res.x[:N], 0.0, 1.0)
        # Fix sum
        diff = N / 2.0 - np.sum(h_candidate)
        if abs(diff) > 1e-9:
            h_candidate = h_candidate + diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
        
        val = _value(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()
    except Exception as e:
        pass
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin for 150s limit
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Binary local search with diverse initializations
    num_random_restarts = 15
    num_sin_restarts = 10
    
    # Random restarts
    for restart in range(num_random_restarts):
        if time.time() > deadline - 30:
            break
            
        h_bin = np.zeros(N, dtype=np.int8)
        ones_idx = rng.choice(N, size=N//2, replace=False)
        h_bin[ones_idx] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    # Sinusoidal restarts for diverse starting points
    for k in range(1, num_sin_restarts + 1):
        if time.time() > deadline - 30:
            break
        
        phase = rng.uniform(0, 2*np.pi)
        h_cont = 0.5 + 0.5 * np.sin(2 * np.pi * k * np.arange(N) / N + phase)
        h_bin = (h_cont >= 0.5).astype(np.int8)
        
        # Ensure exactly N/2 ones
        num_ones = np.sum(h_bin)
        if num_ones != N // 2:
            if num_ones > N // 2:
                excess = num_ones - N // 2
                ones_idx = np.where(h_bin == 1)[0]
                to_flip = rng.choice(ones_idx, size=excess, replace=False)
                h_bin[to_flip] = 0
            else:
                missing = N // 2 - num_ones
                zeros_idx = np.where(h_bin == 0)[0]
                to_flip = rng.choice(zeros_idx, size=missing, replace=False)
                h_bin[to_flip] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()

    # Phase 2: Convex refinement from the best binary solution
    if time.time() < deadline - 10:
        h_refined, val_refined = _convex_refine(best_h, deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined
    
    # Phase 3: If time remains, try convex refinement from perturbed solutions
    if time.time() < deadline - 15:
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution
            perturbation = rng.normal(0, 0.1, size=N)
            h_perturbed = np.clip(best_h + perturbation, 0.0, 1.0)
            # Fix sum
            diff = N/2.0 - np.sum(h_perturbed)
            h_perturbed = h_perturbed + diff / N
            h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
            
            h_refined, val_refined = _convex_refine(h_perturbed, deadline)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined
    
    return best_h.tolist()
