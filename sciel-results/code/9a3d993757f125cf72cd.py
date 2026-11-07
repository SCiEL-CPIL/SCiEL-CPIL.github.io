import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _raw_max_corr(h):
    """Compute just the max correlation value (for optimization)."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def _fast_max_corr(h):
    """Faster max correlation using FFT."""
    # np.correlate(h, 1-h, 'full') computes sum_i h[i] * (1-h)[i-lag]
    # = sum_i h[i] - sum_i h[i]*h[i-lag]
    # Since sum h[i] is fixed, minimizing max corr = minimizing max (sum h - autocorr)
    # = maximizing min autocorr... wait no.
    # Actually: corr_full[lag] = sum_i h[i] * (1-h)[i-lag] where (1-h)[i-lag] = 1 - h[i-lag]
    # So corr_full[lag] = sum_i h[i] - sum_i h[i]*h[i-lag] = N/2 - P_lag
    # where P_lag = sum_i h[i]*h[i-lag] is the autocorrelation
    # To minimize max corr_full, we want to maximize min P_lag
    # But we can just compute it directly with FFT
    n = len(h)
    f1 = np.fft.rfft(h, n=2*n)
    f2 = np.fft.rfft(1-h, n=2*n)
    corr = np.fft.irfft(f1 * np.conj(f2), n=2*n)
    # The 'full' correlation has length 2*n-1
    # np.correlate(h, 1-h, 'full') gives indices 0 to 2*n-2
    # FFT gives circular correlation of length 2*n, which wraps around
    # We need to extract the linear part
    # Actually, let's just use the direct method for correctness and speed up the local search differently
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(50):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _smooth_max_corr(h, lam=50.0):
    """Smooth approximation of max(corr) using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    # Soft max: m + (1/lam) * log(sum(exp((corr - m) * lam)))
    # This is numerically stable
    return m + np.log(np.sum(np.exp((corr - m) * lam))) / lam

def _local_search_targeted(h, max_iters=3000, time_limit=None):
    """
    Targeted local search:
    1. Compute R_k = sum h_i (1-h_{i+k}) for all k (these are the correlation values).
    2. Find k_max = argmax R_k.
    3. Try to decrease R_k_max by adjusting pairs.
    4. Specifically, R_k = sum_i h_i (1 - h_{i+k}) = sum_i h_i - sum_i h_i h_{i+k}.
       Since sum h_i is fixed, minimizing R_k is equivalent to maximizing P_k = sum_i h_i h_{i+k}.
       So we want to increase P_k for the peak lag k.
       P_k = sum_{i} h_i h_{i+k}.
       If we increase h_i and decrease h_j, the change in P_k is:
       Delta P_k = (h_i + d)h_{i+k} + h_{i+k+d}... wait, indices are tricky.
       
       Let's just use a greedy approach:
       Find the lag k with max correlation.
       Try to move mass from a position where it contributes little to P_k to a position where it contributes a lot.
       Or simpler: try small random moves and accept if max_corr decreases.
    """
    best_h = h.copy()
    best_val = _raw_max_corr(best_h)
    
    start_time = time.time()
    if time_limit is None:
        time_limit = start_time + 30.0
    
    # Precompute indices for lags
    # corr index c corresponds to lag l = c - (N-1)
    # R_l = sum_{i} h_i h_{i-l} (with 0 outside [0, N))
    # Actually np.correlate(h, 1-h, 'full') computes sum h[i] (1-h[i-lag])
    # So we want to minimize max_l sum h[i] (1-h[i-l])
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        current_max = corr[max_idx]
        
        # Try a few random pair swaps
        improved = False
        for _ in range(30):
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            # Determine direction for i
            # We want to reduce the peak.
            # Heuristic: if h[i] is high and h[j] is low, maybe swap?
            # Or just try small perturbations.
            
            # Let's try moving d from i to j
            # d can be positive or negative
            # Constraint: 0 <= h_i + d <= 1, 0 <= h_j - d <= 1
            
            # Try to push h[i] towards 0 or 1 (binarization tends to help in these problems)
            if best_h[i] < 0.5:
                d = -best_h[i] # set to 0
            else:
                d = 1.0 - best_h[i] # set to 1
            
            # Check if j can absorb -d
            new_h_j = best_h[j] - d
            if new_h_j < 0 or new_h_j > 1:
                # Scale d
                if d > 0:
                    # h_i increases, h_j decreases
                    # h_j - d >= 0 => d <= h_j
                    # h_j - d <= 1 => d >= h_j - 1 (usually true if d small)
                    # h_i + d <= 1 => d <= 1 - h_i
                    d_max = min(1.0 - best_h[i], best_h[j])
                    if d > d_max:
                        d = d_max
                else:
                    # h_i decreases, h_j increases
                    # h_i + d >= 0 => d >= -h_i
                    # h_j - d <= 1 => -d <= 1 - h_j => d >= h_j - 1
                    # h_j - d >= 0 => d <= h_j
                    d_min = max(-best_h[i], best_h[j] - 1.0)
                    if d < d_min:
                        d = d_min
                
                if abs(d) < 1e-6:
                    continue
            
            new_h = best_h.copy()
            new_h[i] += d
            new_h[j] -= d
            
            val = _raw_max_corr(new_h)
            if val < best_val - 1e-12:
                best_val = val
                best_h = new_h
                improved = True
                break
        
        if not improved:
            # If no improvement with large steps, try smaller random steps
            for _ in range(10):
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j:
                    continue
                d = np.random.uniform(-0.05, 0.05)
                
                new_h_i = best_h[i] + d
                new_h_j = best_h[j] - d
                if 0 <= new_h_i <= 1 and 0 <= new_h_j <= 1:
                    new_h = best_h.copy()
                    new_h[i] = new_h_i
                    new_h[j] = new_h_j
                    val = _raw_max_corr(new_h)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
            
            if not improved and it > 50:
                # Stuck, break early? Or continue?
                # Let's break if stuck for a while to save time for other phases
                # But maybe we should try a different strategy?
                # For now, break.
                if it > 100:
                    break
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)

    # Phase 1: Differential Evolution on Fourier Coefficients
    # h_i = 0.5 + sum_k (a_k cos(...) + b_k sin(...))
    n_harm = 12
    n_params = 2 * n_harm
    i = np.arange(N)
    
    # Build basis matrix
    cols = []
    for k in range(1, n_harm + 1):
        cols.append(np.cos(2 * np.pi * k * i / N))
        cols.append(np.sin(2 * np.pi * k * i / N))
    basis = np.column_stack(cols)

    def fourier_objective(params):
        h = np.full(N, 0.5)
        h += params @ basis
        h = _project_to_feasible(h)
        return _raw_max_corr(h)

    bounds = [(-0.5, 0.5)] * n_params

    de_deadline = start_time + 50.0

    def timed_fourier_obj(params):
        if time.time() > de_deadline:
            raise TimeoutError()
        return fourier_objective(params)

    try:
        result = differential_evolution(
            timed_fourier_obj,
            bounds=bounds,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            init='sobol'
        )
        if result.x is not None:
            h_de = 0.5 + result.x @ basis
            h_de = _project_to_feasible(h_de)
            val_de = _raw_max_corr(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass

    # Phase 2: Local Search Polish
    ls_deadline = deadline - 25.0
    if time.time() < ls_deadline:
        best_h = _local_search_targeted(best_h, time_limit=ls_deadline)
        best_val = _raw_max_corr(best_h)

    # Phase 3: SLSQP with smooth objective - try multiple lambda values
    if time.time() < deadline - 10.0:
        try:
            # Use a smooth approximation of max for gradient
            # Try progressively tighter lambda for better convergence
            for lam in [50.0, 100.0, 200.0]:
                if time.time() > deadline - 5.0:
                    break
                
                def smooth_obj(x, lam=lam):
                    return _smooth_max_corr(x, lam=lam)

                def constraint(x):
                    return np.sum(x) - N / 2.0

                res = minimize(
                    smooth_obj,
                    best_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': constraint}],
                    options={'maxiter': 150, 'ftol': 1e-14}
                )
                if res.success or res.fun < best_val:
                    h_opt = _project_to_feasible(res.x)
                    val_opt = _raw_max_corr(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Final Polish with Local Search
    if time.time() < deadline - 2.0:
        best_h = _local_search_targeted(best_h, max_iters=500, time_limit=deadline - 1.0)
        best_val = _raw_max_corr(best_h)
        
    best_h = _project_to_feasible(best_h)
    
    return best_h
