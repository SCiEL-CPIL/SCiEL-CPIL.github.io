"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import random

def _max_overlap_fast(h):
    """Compute max overlap using FFT-based circular cross-correlation."""
    N = len(h)
    one_minus_h = 1.0 - h
    # overlap[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # Using FFT: ifft(fft(h) * conj(fft(one_minus_h))) gives circular cross-correlation
    # Specifically, ifft(fft(a)*conj(fft(b)))[k] = sum_n a[(n-k)%N] * b[n]
    # We want sum_i h[i] * one_minus_h[(i+k)%N]
    # Let's verify with a small example or just use the property:
    # Actually, the circular cross-correlation (h ⊛ g)[k] = sum_n h[(n-k)%N] * g[n]
    # = ifft(fft(h) * conj(fft(g)))[k]
    # We want sum_i h[i] * g[(i+k)%N] where g = one_minus_h
    # Let m = i+k, i = m-k: sum_m h[(m-k)%N] * g[m] = (h ⊛ g)[k]
    # So overlap[k] = ifft(fft(h) * conj(fft(g)))[k]
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    overlaps = np.real(np.fft.ifft(H * np.conj(G)))
    return np.max(overlaps)

def _score(h):
    """Score is the upper bound on C5, which is the max overlap."""
    return _max_overlap_fast(h)

def _fix_sum(h, N):
    """Ensure h sums to exactly N/2 and is in [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            # Add to elements with room, spread out
            order = np.argsort(1.0 - h)  # most room first
            for idx in order:
                if diff <= 1e-14:
                    break
                add = min(diff, 1.0 - h[idx])
                h[idx] += add
                diff -= add
        else:
            # Subtract from elements with value
            order = np.argsort(-h)  # most value first
            for idx in order:
                if -diff <= 1e-14:
                    break
                sub = min(-diff, h[idx])
                h[idx] -= sub
                diff += sub
    return h

def _try_alternating(N):
    """Try alternating high/low values with optimal a."""
    # h[i] = a if i even, 1-a if i odd
    # Sum = N/2 * a + N/2 * (1-a) = N/2 ✓
    # Overlap at even k: N*a*(1-a)  [wait, let me recompute]
    # overlap[k] = sum_i h[i]*(1-h[(i+k)%N])
    # For k even: h[(i+k)%N] has same parity as h[i]
    #   = sum_i h[i]*(1-h[i]) = sum_i h[i] - sum_i h[i]^2
    #   = N/2 - (N/2)(a^2 + (1-a)^2)
    # For k odd: h[(i+k)%N] has opposite parity
    #   = sum_i h[i]*(1-(1-h[i])) = sum_i h[i]*h[i] = sum_i h[i]^2
    #   = (N/2)(a^2 + (1-a)^2)
    # Max overlap = max(N/2 - (N/2)(a^2+(1-a)^2), (N/2)(a^2+(1-a)^2))
    # Let f(a) = a^2+(1-a)^2 = 2a^2-2a+1
    # Even: N/2 - (N/2)f(a) = (N/2)(1-f(a))
    # Odd: (N/2)f(a)
    # Max = (N/2)*max(1-f(a), f(a))
    # Minimize max(1-f, f) => minimize max(f, 1-f) => f = 1/2
    # 2a^2-2a+1 = 1/2 => 2a^2-2a+1/2=0 => 4a^2-4a+1=0 => (2a-1)^2=0 => a=1/2
    # At a=1/2: f=1/2, max overlap = N/2 * 1/2 = N/4
    # So a=0.5 is optimal for period-2
    a = 0.5
    h = np.zeros(N)
    h[0::2] = a
    h[1::2] = 1 - a
    return _fix_sum(h, N)

def _try_periodic_optimize(N, period, time_budget=5.0):
    """Optimize a periodic pattern with given period using scipy."""
    if period <= 0 or period > N or N % period != 0:
        return None
    repeats = N // period
    target_sum = period / 2.0  # sum of one period
    
    def _objective(x):
        """x is the pattern of length 'period', sum must be target_sum."""
        pattern = np.clip(x, 0, 1)
        # Scale to correct sum
        s = np.sum(pattern)
        if s < 1e-14:
            pattern = np.full(period, target_sum / period)
        else:
            pattern = pattern * (target_sum / s)
            pattern = np.clip(pattern, 0, 1)
        # Re-fix sum after clipping
        diff = target_sum - np.sum(pattern)
        if abs(diff) > 1e-14:
            pattern[0] += diff
            pattern[0] = np.clip(pattern[0], 0, 1)
        h = np.tile(pattern, repeats)
        return _score(h)
    
    # Start from a few initial points
    best_x = None
    best_val = float('inf')
    
    # Initial: uniform
    x0 = np.full(period, target_sum / period)
    for trial in range(3):
        if trial == 0:
            x_start = x0.copy()
        elif trial == 1:
            x_start = np.random.uniform(0, 1, period)
        else:
            # Alternating within period
            x_start = np.zeros(period)
            x_start[0::2] = target_sum / (period/2 + 0.5)
            x_start[1::2] = 0
            x_start = np.clip(x_start, 0, 1)
        
        res = minimize(_objective, x_start, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        val = _objective(res.x)
        if val < best_val:
            best_val = val
            best_x = res.x
    
    if best_x is None:
        return None
    
    pattern = np.clip(best_x, 0, 1)
    s = np.sum(pattern)
    if s > 1e-14:
        pattern = pattern * (target_sum / s)
    else:
        pattern = np.full(period, target_sum / period)
    pattern = np.clip(pattern, 0, 1)
    diff = target_sum - np.sum(pattern)
    if abs(diff) > 1e-14:
        pattern[0] += diff
        pattern[0] = np.clip(pattern[0], 0, 1)
    
    h = np.tile(pattern, repeats)
    return _fix_sum(h, N)

def _local_search_fast(h, max_iter=2000, step_size=0.03, time_limit=10.0):
    """Fast local search using FFT-based scoring."""
    N = len(h)
    current = h.copy()
    current_score = _score(current)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        # Pick two random indices
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        delta = random.uniform(-step_size, step_size)
        new_i = current[i] + delta
        new_j = current[j] - delta
        if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
            continue
        
        new_h = current.copy()
        new_h[i] = new_i
        new_h[j] = new_j
        
        new_score = _score(new_h)
        if new_score < current_score - 1e-12:
            current = new_h
            current_score = new_score
    
    return current, current_score

def _differential_evolution_search(N, time_budget=30.0):
    """Use differential evolution to search for good h."""
    start = time.time()
    
    def _objective(x):
        if time.time() - start > time_budget - 5:
            return 1e10
        h = np.clip(x, 0, 1)
        # Fix sum
        diff = N/2.0 - np.sum(h)
        if abs(diff) > 1e-14:
            # Distribute the difference
            idx = np.random.randint(0, N)
            h[idx] += diff
            h[idx] = np.clip(h[idx], 0, 1)
            # If still off, adjust another
            diff2 = N/2.0 - np.sum(h)
            if abs(diff2) > 1e-14:
                idx2 = np.random.randint(0, N)
                if idx2 != idx:
                    h[idx2] += diff2
                    h[idx2] = np.clip(h[idx2], 0, 1)
        return _score(h)
    
    bounds = [(0, 1)] * N
    try:
        res = differential_evolution(_objective, bounds, maxiter=50, 
                                     popsize=15, tol=1e-6, seed=42,
                                     polish=True, init='sobol')
        h = np.clip(res.x, 0, 1)
        h = _fix_sum(h, N)
        return h, _score(h)
    except Exception:
        return None, float('inf')

def _refine_with_scipy(h, time_budget=10.0):
    """Refine h using scipy minimize with penalty for constraints."""
    start = time.time()
    N = len(h)
    
    def _objective(x):
        if time.time() - start > time_budget:
            return 1e10
        h = np.clip(x, 0, 1)
        # Penalty for sum constraint
        sum_penalty = (np.sum(h) - N/2.0) ** 2 * 100
        return _score(h) + sum_penalty
    
    res = minimize(_objective, h, method='Nelder-Mead',
                   options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
    h_new = np.clip(res.x, 0, 1)
    h_new = _fix_sum(h_new, N)
    return h_new, _score(h_new)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin from 150s limit
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with alternating pattern (known good: N/4)
    best_h = _try_alternating(N)
    best_score = _score(best_h)
    
    # Try periodic patterns with optimization
    for period in [2, 4, 5, 10, 20, 25, 50]:
        if time.time() > deadline - 40:
            break
        if period > N or N % period != 0:
            continue
        pat = _try_periodic_optimize(N, period, time_budget=5.0)
        if pat is not None:
            s = _score(pat)
            if s < best_score:
                best_score = s
                best_h = pat
    
    # Local search from best
    if time.time() < deadline - 30:
        h1, s1 = _local_search_fast(best_h, max_iter=2000, step_size=0.03, time_limit=10.0)
        if s1 < best_score:
            best_score = s1
            best_h = h1
    
    # Differential evolution
    if time.time() < deadline - 20:
        h_de, s_de = _differential_evolution_search(N, time_budget=25.0)
        if h_de is not None and s_de < best_score:
            best_score = s_de
            best_h = h_de
    
    # Refine with scipy
    if time.time() < deadline - 10:
        h_ref, s_ref = _refine_with_scipy(best_h, time_budget=8.0)
        if s_ref < best_score:
            best_score = s_ref
            best_h = h_ref
    
    # Final local search
    if time.time() < deadline - 3:
        h2, s2 = _local_search_fast(best_h, max_iter=1000, step_size=0.01, time_limit=5.0)
        if s2 < best_score:
            best_score = s2
            best_h = h2
    
    # Ensure exact sum and bounds
    best_h = _fix_sum(best_h, N)
    
    return best_h
# EVOLVE-BLOCK-END
