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

def _calc_corr_vec(h):
    """Return the full correlation vector for fast local search."""
    h = np.asarray(h, dtype=np.float64)
    return np.correlate(h, 1.0 - h, mode="full")

def _local_search_binary(h_bin):
    """
    Local search on binary vectors with exactly N/2 ones.
    Tries swaps (one 1 with one 0) to preserve the count of ones.
    """
    h_bin = np.array(h_bin, dtype=np.int8)
    h_float = h_bin.astype(np.float64)
    corr = _calc_corr_vec(h_float)
    current_max = float(np.max(corr))
    
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
                
                new_corr = _calc_corr_vec(h_temp.astype(np.float64))
                new_max = float(np.max(new_corr))
                gain = current_max - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_new_h = h_temp
        
        if best_gain > 1e-6 and best_new_h is not None:
            h_bin = best_new_h
            h_float = h_bin.astype(np.float64)
            corr = _calc_corr_vec(h_float)
            current_max = float(np.max(corr))
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            improved = True
            
    return h_bin.astype(np.float64), current_max

def _continuous_refine(h_start, deadline):
    """
    Multi-stage continuous refinement using epigraph formulation.
    Minimize t subject to corr_k <= t for all k, sum(h) = N/2, 0 <= h <= 1.
    Returns the best h found and its value.
    """
    best_h = h_start.copy()
    best_val = _value(best_h)
    
    # Use epigraph: variables are [h_0, ..., h_{N-1}, t]
    # Objective: t
    # Constraints: corr_k - t <= 0 for all k, sum(h) - N/2 = 0, 0 <= h_i <= 1
    
    def _epigraph_objective(x):
        h = x[:N]
        t = x[N]
        return t
    
    def _corr_constraint(k):
        def c(x):
            h = x[:N]
            t = x[N]
            corr = np.correlate(h, 1.0 - h, mode="full")
            return corr[k] - t
        return c
    
    # Number of correlation lags
    num_lags = 2 * N - 1
    
    bounds = [(0.0, 1.0)] * N + [(0.0, 10.0)]  # t upper bound
    
    # Build constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N/2.0}]
    for k in range(num_lags):
        constraints.append({'type': 'ineq', 'fun': _corr_constraint(k)})
    
    # Initial t estimate
    h_init = best_h.copy()
    t_init = _calc_max_corr(h_init) + 0.1
    
    x0 = np.concatenate([h_init, [t_init]])
    
    # Try multiple precision levels
    for maxiter in [200, 500, 1000]:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _epigraph_objective, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14}
            )
            
            if res.success or res.fun < best_val:
                h_candidate = np.clip(res.x[:N], 0.0, 1.0)
                # Fix sum to satisfy constraint
                diff = N/2.0 - np.sum(h_candidate)
                if abs(diff) > 1e-9:
                    h_candidate = h_candidate + diff / N
                    h_candidate = np.clip(h_candidate, 0.0, 1.0)
                
                val = _value(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
                    t_init = _calc_max_corr(h_candidate) + 0.01
                    x0 = np.concatenate([h_candidate, [t_init]])
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin for 150s limit
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Binary local search with diverse initializations
    # We keep track of the top few binary solutions to use as seeds for continuous refinement
    
    binary_solutions = []
    
    # Random restarts
    num_random_restarts = 25
    for restart in range(num_random_restarts):
        if time.time() > deadline - 40:
            break
            
        h_bin = np.zeros(N, dtype=np.int8)
        ones_idx = rng.choice(N, size=N//2, replace=False)
        h_bin[ones_idx] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        # Keep top 5 binary solutions
        binary_solutions.append((val, h_opt.copy()))
        binary_solutions.sort(key=lambda x: x[0])
        if len(binary_solutions) > 5:
            binary_solutions.pop()
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    # Sinusoidal restarts for diverse starting points
    num_sin_restarts = 15
    for k in range(1, num_sin_restarts + 1):
        if time.time() > deadline - 40:
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
        
        binary_solutions.append((val, h_opt.copy()))
        binary_solutions.sort(key=lambda x: x[0])
        if len(binary_solutions) > 5:
            binary_solutions.pop()
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()

    # Phase 2: Continuous refinement from the best binary solution
    if time.time() < deadline - 20:
        h_refined, val_refined = _continuous_refine(best_h, deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined
    
    # Phase 3: Continuous refinement from other good binary solutions
    # This helps escape local minima that the best binary solution might lead into
    if time.time() < deadline - 10:
        for val_bin, h_bin in binary_solutions:
            if time.time() > deadline - 5:
                break
            if val_bin >= best_val:
                continue
                
            h_refined, val_refined = _continuous_refine(h_bin, deadline)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined

    # Phase 4: Perturbation and re-refinement
    if time.time() < deadline - 5:
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
            
            # Perturb the best solution
            perturbation = rng.normal(0, 0.05, size=N)
            h_perturbed = np.clip(best_h + perturbation, 0.0, 1.0)
            # Fix sum
            diff = N/2.0 - np.sum(h_perturbed)
            h_perturbed = h_perturbed + diff / N
            h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
            
            h_refined, val_refined = _continuous_refine(h_perturbed, deadline)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined
    
    # Phase 5: High-precision polish with exact local search
    if time.time() < deadline - 3:
        # Do a few rounds of coordinate descent with tight tolerance
        for polish_round in range(3):
            if time.time() > deadline - 2:
                break
            
            h_polish = best_h.copy()
            improved_polish = True
            while improved_polish and time.time() < deadline - 2:
                improved_polish = False
                best_gain_polish = 0
                best_change = None
                
                for i in range(N):
                    for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]:
                        h_temp = h_polish.copy()
                        h_temp[i] = np.clip(h_temp[i] + delta, 0.0, 1.0)
                        # Fix sum
                        diff_sum = N/2.0 - np.sum(h_temp)
                        if abs(diff_sum) > 1e-9:
                            # Distribute correction to other elements
                            others = [j for j in range(N) if j != i]
                            h_temp[others] += diff_sum / len(others)
                            h_temp = np.clip(h_temp, 0.0, 1.0)
                        
                        val_temp = _value(h_temp)
                        gain = best_val - val_temp
                        if gain > best_gain_polish:
                            best_gain_polish = gain
                            best_change = (i, delta)
                
                if best_gain_polish > 1e-10 and best_change is not None:
                    i, delta = best_change
                    h_polish[i] = np.clip(h_polish[i] + delta, 0.0, 1.0)
                    diff_sum = N/2.0 - np.sum(h_polish)
                    if abs(diff_sum) > 1e-9:
                        others = [j for j in range(N) if j != i]
                        h_polish[others] += diff_sum / len(others)
                        h_polish = np.clip(h_polish, 0.0, 1.0)
                    val_polish = _value(h_polish)
                    if val_polish < best_val:
                        best_val = val_polish
                        best_h = h_polish.copy()
                    improved_polish = True
    
    # Final validation
    final_h = np.clip(best_h, 0.0, 1.0)
    diff = N/2.0 - np.sum(final_h)
    if abs(diff) > 1e-9:
        final_h = final_h + diff / N
        final_h = np.clip(final_h, 0.0, 1.0)
    
    return final_h.tolist()
