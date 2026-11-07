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
    Tries single flips and double flips (swap one 1 with one 0).
    """
    h_bin = np.array(h_bin, dtype=np.int8)
    current_max = _calc_max_corr(h_bin.astype(np.float64))
    
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_i = -1
        best_j = -1
        best_new_h = None
        
        # Try swapping a 1 at i with a 0 at j
        # This preserves the count of ones.
        for i in ones:
            for j in zeros:
                h_temp = h_bin.copy()
                h_temp[i] = 0
                h_temp[j] = 1
                
                new_max = _calc_max_corr(h_temp.astype(np.float64))
                gain = current_max - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
                    best_new_h = h_temp
        
        if best_gain > 1e-6:
            h_bin = best_new_h
            current_max = current_max - best_gain
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            improved = True
            
    return h_bin.astype(np.float64), current_max

def _continuous_refine(h0, deadline):
    """
    Refine a continuous vector h0 using SLSQP with an auxiliary variable t
    to minimize max(corr) subject to sum(h) = N/2 and 0 <= h <= 1.
    """
    N_ = len(h0)
    num_lags = 2 * N_ - 1

    def _objective(x):
        # x = [h (N), t (1)]
        h = np.clip(x[:N_], 0.0, 1.0)
        t = x[N_]
        corr = np.correlate(h, 1.0 - h, mode="full")
        # Soft constraint: penalize violations of corr <= t
        # We use a smooth max approximation here for the objective
        # but the constraints will enforce the hard limit.
        # Actually, for SLSQP, it's better to just minimize t.
        # The constraints will push t up if corr > t.
        return t

    def _constraints(x):
        h = np.clip(x[:N_], 0.0, 1.0)
        t = x[N_]
        corr = np.correlate(h, 1.0 - h, mode="full")
        
        # Constraint: sum(h) = N/2
        # This is handled by the equality constraint list below.
        
        # Constraints: corr[k] <= t for all k
        # We can't easily list 199 constraints efficiently in a generic way
        # without a loop in the function, but SLSQP can handle it.
        # However, passing a list of constraint dicts is cleaner.
        pass

    # Better approach: Define constraints dynamically
    # But defining 199 constraints every time might be slow.
    # Let's try a different formulation:
    # Minimize t
    # Subject to:
    # 1. sum(h) - N/2 == 0
    # 2. h[i] - 1 <= 0
    # 3. -h[i] <= 0
    # 4. corr[k] - t <= 0 for all k

    def obj(x):
        return x[N_]  # t

    # We need to return a list of constraints.
    # To avoid recreating the list every time (which is slow),
    # we can define a single constraint function that returns an array
    # of inequalities, but SLSQP expects a list of dicts.
    # Alternatively, we can use a "soft" constraint in the objective
    # and a hard constraint for the sum.
    # Given the difficulty of dynamic constraints in SLSQP interface,
    # let's stick to the soft-max objective but refine it with the t-variable
    # logic if we use trust-constr or similar.
    
    # Let's try SLSQP with a penalty for max(corr) - t
    # If we minimize t + penalty * max(0, max(corr) - t), it's not smooth.
    
    # Let's go back to the soft-max objective but use a high lambda.
    # The directive suggests auxiliary t.
    # Let's implement the auxiliary t approach with SLSQP by
    # using a list of constraints.
    
    # To be efficient, we will pre-calculate the constraint functions?
    # No, they depend on x.
    
    # Let's try a simpler approach:
    # Minimize max(corr) using SLSQP directly? No, max is not differentiable.
    
    # Let's use the auxiliary variable t and define the constraints
    # in a loop inside the construct function, but that's not possible
    # because the constraints must be defined before minimize.
    
    # Workaround: Define a single inequality constraint that returns
    # the maximum violation: max(corr) - t <= 0.
    # This is not differentiable at max(corr) = t, but SLSQP might handle it
    # or we can smooth it.
    
    def _ineq_constraint(x):
        h = np.clip(x[:N_], 0.0, 1.0)
        t = x[N_]
        corr = np.correlate(h, 1.0 - h, mode="full")
        # Return t - max(corr). We want this to be >= 0.
        # max(corr) is not differentiable.
        # Let's use a smooth approximation: log-sum-exp.
        lam = 50.0
        shifted = corr - np.max(corr)
        soft_max = np.log(np.sum(np.exp(lam * shifted))) / lam + np.max(corr)
        return t - soft_max

    def _eq_constraint(x):
        h = np.clip(x[:N_], 0.0, 1.0)
        return np.sum(h) - N_ / 2.0

    # Bounds:
    # h[i] in [0, 1]
    # t in [0, 2] (max possible corr is N, so max corr/N * 2 is 2? No, corr max is N. t is max corr. So t in [0, N])
    # Wait, the objective in the task is max(corr)/N * 2.
    # If we minimize t = max(corr), then value = t/N * 2.
    # So minimizing t is equivalent to minimizing value.
    
    bounds_t = [(0.0, 1.0)] * N_ + [(0.0, float(N_))]
    
    cons = [
        {'type': 'ineq', 'fun': _ineq_constraint},
        {'type': 'eq', 'fun': _eq_constraint}
    ]

    h_init = h0.copy()
    # Initial t: current max corr
    corr_init = np.correlate(h_init, 1.0 - h_init, mode="full")
    t_init = np.max(corr_init)
    
    x0 = np.concatenate([h_init, [t_init]])
    
    try:
        res = minimize(
            obj, x0,
            method='SLSQP',
            bounds=bounds_t,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if res.success:
            h_new = np.clip(res.x[:N_], 0.0, 1.0)
            # Fix sum
            diff = N_ / 2.0 - np.sum(h_new)
            if abs(diff) > 1e-9:
                h_new = h_new + diff / N_
                h_new = np.clip(h_new, 0.0, 1.0)
            
            val = _value(h_new)
            return h_new, val
        else:
            return h0, _value(h0)
            
    except Exception:
        return h0, _value(h0)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin for 150s limit

    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # 1. Binary Local Search with diverse initializations
    # We use random starts AND structured sinusoidal starts to find diverse local optima.
    num_random_restarts = 20
    num_sin_restarts = 15
    
    # Random restarts
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
            
    # Sinusoidal restarts
    # Structured patterns might lead to different local minima.
    for k in range(1, num_sin_restarts + 1):
        if time.time() > deadline - 40:
            break
        
        # Frequency k
        # A simple sine wave sin(2*pi*k*i/N) will have equal positive and negative parts (approx).
        # We add a small random phase shift to diversify.
        phase = rng.uniform(0, 2*np.pi)
        h_cont = 0.5 + 0.5 * np.sin(2 * np.pi * k * np.arange(N) / N + phase)
        
        # Threshold to binary
        h_bin = (h_cont >= 0.5).astype(np.int8)
        
        # Ensure exactly N/2 ones? The task requires sum(h) = N/2.
        # The binary local search preserves the number of ones.
        # If the number of ones is not N/2, we need to adjust.
        num_ones = np.sum(h_bin)
        if num_ones != N // 2:
            # Swap some to get exactly N/2 ones
            if num_ones > N // 2:
                # Remove excess ones
                excess = num_ones - N // 2
                ones_idx = np.where(h_bin == 1)[0]
                zeros_idx = np.where(h_bin == 0)[0]
                # Randomly pick excess ones to flip to 0
                to_flip = rng.choice(ones_idx, size=excess, replace=False)
                h_bin[to_flip] = 0
            else:
                # Add missing ones
                missing = N // 2 - num_ones
                ones_idx = np.where(h_bin == 1)[0]
                zeros_idx = np.where(h_bin == 0)[0]
                # Randomly pick missing zeros to flip to 1
                to_flip = rng.choice(zeros_idx, size=missing, replace=False)
                h_bin[to_flip] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()

    # 2. Continuous Refinement
    # Use the best binary solution as a starting point for continuous optimization.
    
    if time.time() < deadline - 10:
        h_refined, val_refined = _continuous_refine(best_h, deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined

    # If time remains, try a few more refinements or restarts
    if time.time() < deadline - 2:
        # Try a few random perturbations of the best solution and refine
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            h_perturbed = best_h + rng.normal(0, 0.05, N)
            h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
            # Fix sum
            diff = N/2.0 - np.sum(h_perturbed)
            h_perturbed = h_perturbed + diff / N
            h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
            
            h_refined, val_refined = _continuous_refine(h_perturbed, deadline)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined

    return best_h.tolist()
