"""
Combined PSL minimization: sparse combinatorial design + Gordon's gradient method.

Strategy:
- The sparse approach decouples the search: first choose WHICH positions are active,
  then optimize signs. This is structurally different from pure ±1 search.
- The gradient method provides fast local optimization of signs.
- We combine: initialize with sparse patterns, use gradient to optimize signs, then use
  a swap-based move to adjust sparsity/positions, and simulated annealing for global escape.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _score(h):
    """Compute the upper bound on C3: max absolute autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf


def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _gradient_step(h, acf):
    """
    Gordon's fast PSL minimization step.
    For each position i, compute the change in PSL if we flip h[i].
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    S = np.zeros((n, n - 1), dtype=np.float64)
    for k in range(1, n):
        idx = k - 1
        S[k:, idx] += h[:n - k]
        S[:n - k, idx] += h[k:]
    
    delta = -2.0 * h[:, np.newaxis] * S
    
    acf_lags = acf[1:n]
    
    new_acf = delta + acf_lags[np.newaxis, :]
    new_psl = np.max(np.abs(new_acf), axis=1)
    
    best_i = int(np.argmin(new_psl))
    best_new_psl = new_psl[best_i]
    
    return best_i, float(best_new_psl), current_psl


def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if new_psl >= cur_psl:
            break
        
        h[i] = -h[i]
        passes += 1
        
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl


def _simulated_annealing_psl(h, time_budget=30.0, start_time=None):
    """
    Simulated annealing on PSL using the fast gradient to evaluate moves.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 10.0
    T_min = 0.1
    cooling = 0.95
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = np.random.randint(n)
        
        delta = np.zeros(n - 1)
        for k in range(1, n):
            contrib = 0.0
            if i - k >= 0:
                contrib += h[i - k]
            if i + k < n:
                contrib += h[i + k]
            delta[k - 1] = -2.0 * h[i] * contrib
        
        new_acf_lags = acf[1:n] + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_psl


def _sparse_init(k, rng):
    """Create a sparse initialization with k active positions."""
    h = np.zeros(N, dtype=np.float64)
    indices = rng.choice(N, size=k, replace=False)
    h[indices] = rng.choice([-1.0, 1.0], size=k)
    return h


def _swap_move(h, acf, rng, time_budget, start_time):
    """
    Try a swap-based move: pick two positions, swap their values (or flip one).
    This allows adjusting sparsity/positions.
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    # Pick two random positions
    i = rng.randint(n)
    j = rng.randint(n)
    if i == j:
        return h, acf, current_psl
    
    # Try swapping h[i] and h[j]
    trial = h.copy()
    trial[i], trial[j] = h[j], h[i]
    
    trial_acf = _compute_acf(trial)
    trial_psl = _psl_value(trial_acf, n)
    
    if trial_psl < current_psl:
        return trial, trial_acf, trial_psl
    
    # Try flipping h[i]
    trial = h.copy()
    trial[i] = -h[i]
    
    trial_acf = _compute_acf(trial)
    trial_psl = _psl_value(trial_acf, n)
    
    if trial_psl < current_psl:
        return trial, trial_acf, trial_psl
    
    return h, acf, current_psl


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the peak sidelobe level (upper bound on C3).
    
    Strategy:
    1. Try multiple sparse initializations with varying densities.
    2. For each, run fast gradient-based PSL optimization.
    3. Use swap-based moves to adjust sparsity/positions.
    4. Use simulated annealing to escape local minima.
    5. Return the best sequence found.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Sparse initializations with varying densities
    sparse_densities = [20, 40, 60, 80, 100, 120, 150, 200]
    
    for k in sparse_densities:
        if time.time() > deadline - 30:
            break
        
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            
            rng = np.random.default_rng(k * 1000 + trial * 137 + 42)
            h_init = _sparse_init(k, rng)
            
            # Also try some structured starts
            if trial == 0 and k == 80:
                h_init = np.array([1.0 if (i * 7) % 11 < 6 else -1.0 for i in range(N)])
            elif trial == 0 and k == 100:
                h_init = np.array([1.0 if (i * 13) % 17 < 9 else -1.0 for i in range(N)])
            
            time_budget = min(5.0, deadline - 30 - (time.time() - start_time))
            if time_budget < 1.0:
                break
            
            h_opt, score = _fast_psl_optimize(h_init, max_passes=500, time_budget=time_budget, start_time=start_time)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Phase 2: Swap-based moves to adjust sparsity/positions
    if best_h is not None and time.time() < deadline - 10:
        swap_time = min(20.0, deadline - 10 - (time.time() - start_time))
        if swap_time > 2.0:
            rng = np.random.default_rng(999)
            h_swap = best_h.copy()
            acf = _compute_acf(h_swap)
            current_psl = _psl_value(acf, N)
            
            swaps = 0
            while time.time() < start_time + swap_time and swaps < 200:
                h_swap, acf, current_psl = _swap_move(h_swap, acf, rng, swap_time, start_time)
                swaps += 1
                
                if current_psl < best_score:
                    best_score = current_psl
                    best_h = h_swap.copy()
    
    # Phase 3: Simulated annealing from the best found
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(30.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            for sa_run in range(3):
                if time.time() > deadline - 5:
                    break
                
                rng = np.random.default_rng(1000 + sa_run * 7)
                h_perturbed = best_h.copy()
                n_flips = max(5, N // 20)
                flip_positions = rng.choice(N, size=n_flips, replace=False)
                h_perturbed[flip_positions] *= -1
                
                sa_budget = min(sa_time / 3, 10.0)
                if sa_budget < 1.0:
                    break
                
                h_sa, score_sa = _simulated_annealing_psl(h_perturbed, time_budget=sa_budget, start_time=start_time)
                
                if score_sa < best_score:
                    best_score = score_sa
                    best_h = h_sa.copy()
    
    # Phase 4: Final gradient polish
    if best_h is not None and time.time() < deadline - 2:
        final_budget = deadline - 2 - (time.time() - start_time)
        if final_budget > 0.5:
            h_final, score_final = _fast_psl_optimize(best_h, max_passes=1000, time_budget=final_budget, start_time=start_time)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
    
    if best_h is None:
        best_h = np.random.choice([-1.0, 1.0], size=N)
    
    return best_h

# EVOLVE-BLOCK-END
