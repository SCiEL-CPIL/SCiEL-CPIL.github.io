"""Combined program for the Erdos minimum overlap task."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _objective(h):
    """
    Compute the objective value: max(correlate(h, 1-h, mode='full')) / N * 2
    Lower is better.
    """
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _objective_smooth(x, lam=50.0):
    """Smooth approximation of max using log-sum-exp."""
    h = x
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    if m > 1e15:
        return m
    # log-sum-exp trick for stability
    shifted = (corr - m) * lam
    # clip to avoid overflow in exp
    shifted = np.clip(shifted, -100, 100)
    return m + np.log(np.sum(np.exp(shifted)) / len(corr)) / lam

def _sum_constraint(h):
    """
    Return the sum constraint violation: sum(h) - N/2
    """
    return np.sum(h) - N / 2

def _bounds():
    """
    Return bounds for each variable: [0, 1]
    """
    return [(0.0, 1.0)] * N

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    if abs(diff) < 1e-12:
        return h
    # Distribute the correction
    h = h - diff / N
    h = np.clip(h, 0.0, 1.0)
    # Second pass
    diff = np.sum(h) - N / 2.0
    h = h - diff / N
    h = np.clip(h, 0.0, 1.0)
    return h

def _binary_local_search(h_bin, max_iters=50):
    """
    Local search on binary vectors with exactly N/2 ones.
    Tries swaps (one 1 with one 0) to preserve the count of ones.
    Returns the best binary vector found.
    """
    h_bin = np.array(h_bin, dtype=np.int8)
    h_float = h_bin.astype(np.float64)
    current_corr = np.correlate(h_float, 1.0 - h_float, mode="full")
    current_max = float(np.max(current_corr))
    
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        best_gain = 0
        best_i = -1
        best_j = -1
        best_new_corr = None
        
        # Try all swaps
        for i in ones:
            for j in zeros:
                # Compute new correlation efficiently
                # Only lags affected by changing h[i] and h[j]
                # But for simplicity and correctness, recompute if N is small enough
                # N=100, so 200 lags. 50*50=2500 swaps. 2500*200 = 500k ops. Fast.
                new_h = h_float.copy()
                new_h[i] = 0.0
                new_h[j] = 1.0
                new_corr = np.correlate(new_h, 1.0 - new_h, mode="full")
                new_max = float(np.max(new_corr))
                
                if new_max < current_max - 1e-12:
                    gain = current_max - new_max
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                        best_j = j
                        best_new_corr = new_corr
                        best_new_max = new_max
        
        if best_i != -1:
            # Apply the best swap
            h_float[best_i] = 0.0
            h_float[best_j] = 1.0
            h_bin[best_i] = 0
            h_bin[best_j] = 1
            
            # Update sets
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            
            current_max = best_new_max
            current_corr = best_new_corr
            improved = True
    
    return h_bin, h_float, current_max

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Start with the seed: all 0.5
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _objective(h0)
    
    # Strategy 1: Binary local search on rounded version
    # Round h0 to binary (all 0.5 -> 0 or 1 arbitrarily, but sum must be 50)
    # Since h0 is all 0.5, we can pick any 50 indices to be 1.
    # Let's start with a structured binary pattern: 0,1,0,1,...
    h_bin = np.zeros(N, dtype=np.int8)
    h_bin[::2] = 1  # 50 ones
    h_bin_float = h_bin.astype(np.float64)
    
    # Run binary local search
    h_bin, h_bin_float, bin_max = _binary_local_search(h_bin, max_iters=100)
    bin_val = bin_max / N * 2
    if bin_val < best_val:
        best_val = bin_val
        best_h = h_bin_float.copy()
    
    # Strategy 2: Smooth optimization from the binary solution
    # Use the binary solution as a starting point for continuous optimization
    h_start = h_bin_float.copy()
    
    # Try multiple optimization strategies with smooth surrogate
    lambdas = [10.0, 50.0, 200.0, 1000.0]
    current_h = h_start.copy()
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 15:
            break
        try:
            res = minimize(
                lambda x: _objective_smooth(x, lam=lam),
                current_h,
                method='SLSQP',
                bounds=_bounds(),
                constraints=[{'type': 'eq', 'fun': _sum_constraint}],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success and np.isfinite(res.fun):
                h_candidate = res.x
                h_candidate = _project_to_feasible(h_candidate)
                val = _objective(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
                current_h = h_candidate
        except Exception:
            pass
    
    # Strategy 3: Exact objective optimization from best continuous solution
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=_bounds(),
                constraints=[{'type': 'eq', 'fun': _sum_constraint}],
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            if res.success and np.isfinite(res.fun):
                h_candidate = res.x
                h_candidate = _project_to_feasible(h_candidate)
                val = _objective(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Strategy 4: Try different binary starting points
    rng = np.random.default_rng(42)
    for init_seed in range(3):
        if time.time() - start_time > deadline - 10:
            break
        
        # Random binary with 50 ones
        h_rand_bin = np.zeros(N, dtype=np.int8)
        idx = rng.choice(N, 50, replace=False)
        h_rand_bin[idx] = 1
        h_rand_float = h_rand_bin.astype(np.float64)
        
        # Run binary local search
        h_rand_bin, h_rand_float, rand_bin_max = _binary_local_search(h_rand_bin, max_iters=50)
        rand_bin_val = rand_bin_max / N * 2
        
        if rand_bin_val < best_val:
            best_val = rand_bin_val
            best_h = h_rand_float.copy()
        
        # Smooth optimization from this binary solution
        current_h = h_rand_float.copy()
        for lam in [50.0, 200.0]:
            if time.time() - start_time > deadline - 10:
                break
            try:
                res = minimize(
                    lambda x: _objective_smooth(x, lam=lam),
                    current_h,
                    method='SLSQP',
                    bounds=_bounds(),
                    constraints=[{'type': 'eq', 'fun': _sum_constraint}],
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if res.success and np.isfinite(res.fun):
                    h_candidate = res.x
                    h_candidate = _project_to_feasible(h_candidate)
                    val = _objective(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()
                    current_h = h_candidate
            except Exception:
                pass
    
    # Final polish: high-precision run
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=_bounds(),
                constraints=[{'type': 'eq', 'fun': _sum_constraint}],
                options={'maxiter': 5000, 'ftol': 1e-16}
            )
            if res.success and np.isfinite(res.fun):
                h_candidate = res.x
                h_candidate = _project_to_feasible(h_candidate)
                val = _objective(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final projection to ensure feasibility
    best_h = _project_to_feasible(best_h)
    
    # Verify
    final_val = _objective(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
