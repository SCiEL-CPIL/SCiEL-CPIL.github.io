"""
Combined PSL minimization: sparse combinatorial design + Gordon's gradient method.

Strategy:
- The sparse approach (inspiration 1) decouples the search: first choose WHICH positions are active,
  then optimize signs. This is structurally different from pure ±1 search.
- The gradient method (current program) provides fast local optimization of signs.
- We combine: initialize with sparse patterns, use gradient to optimize signs, then use
  a swap-based move to adjust sparsity/positions, and simulated annealing for global escape.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_acf(h):
    """Compute full autocorrelation via FFT (circular, padded to avoid wrap)."""
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


def _score(h):
    """Compute the upper bound on C3: max absolute autocorrelation at non-zero lags."""
    acf = _compute_acf(h)
    return _psl_value(acf, len(h))


def _gradient_best_flip(h, acf):
    """
    Find the single best sign flip that reduces PSL most.
    Returns (best_i, new_psl, cur_psl).
    """
    n = len(h)
    cur_psl = _psl_value(acf, n)
    
    # Build S matrix: S[i,k] = h[i-k] + h[i+k] (with 0 for out of bounds)
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
    return best_i, float(new_psl[best_i]), cur_psl


def _fast_psl_optimize(h, max_passes=300, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    Works on any real-valued vector (not just ±1).
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
        
        i, new_psl, cur_psl = _gradient_best_flip(h, acf)
        
        if new_psl >= cur_psl - 1e-12:
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


def _swap_move(h, acf, rng):
    """
    Try a swap: remove one active position and add another.
    Returns (new_h, new_psl, improved) where improved indicates if score decreased.
    """
    n = len(h)
    cur_psl = _psl_value(acf, n)
    
    # Find active positions (non-zero)
    active = np.nonzero(np.abs(h) > 1e-10)[0]
    if len(active) < 2:
        return h, cur_psl, False
    
    # Pick a random active position to remove
    i_remove = active[rng.integers(len(active))]
    
    # Pick a random inactive position to add
    inactive = np.setdiff1d(np.arange(n), active)
    if len(inactive) == 0:
        return h, cur_psl, False
    j_add = inactive[rng.integers(len(inactive))]
    
    # Try both signs for the new position
    best_val = None
    best_psl = cur_psl
    best_sign = 1.0
    
    for sign in [1.0, -1.0]:
        h_trial = h.copy()
        h_trial[i_remove] = 0.0
        h_trial[j_add] = sign
        
        acf_trial = _compute_acf(h_trial)
        psl_trial = _psl_value(acf_trial, n)
        
        if psl_trial < best_psl:
            best_psl = psl_trial
            best_sign = sign
            best_val = h_trial
    
    if best_val is not None:
        return best_val, best_psl, True
    return h, cur_psl, False


def _sparse_optimize(h, time_budget=15.0, start_time=None, rng=None):
    """
    Combined optimization: gradient sign optimization + swap moves for position adjustment.
    This is the structural innovation: alternating between sign optimization (fast gradient)
    and position swaps (sparse combinatorial moves).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    best_h = h.copy()
    
    # Phase 1: Gradient sign optimization
    h, psl = _fast_psl_optimize(h, max_passes=200, time_budget=time_budget * 0.4, start_time=start_time)
    acf = _compute_acf(h)
    psl = _psl_value(acf, n)
    if psl < best_psl:
        best_psl = psl
        best_h = h.copy()
    
    # Phase 2: Alternating swaps and gradient optimization
    swap_count = 0
    max_swaps = 50
    while time.time() - start_time < time_budget * 0.8 and swap_count < max_swaps:
        # Try a swap
        h_new, psl_new, improved = _swap_move(h, acf, rng)
        
        if improved:
            h = h_new
            acf = _compute_acf(h)
            psl = psl_new
            swap_count += 1
            
            # Quick gradient polish after swap
            h, psl = _fast_psl_optimize(h, max_passes=50, time_budget=1.0, start_time=time.time())
            acf = _compute_acf(h)
            psl = _psl_value(acf, n)
            
            if psl < best_psl:
                best_psl = psl
                best_h = h.copy()
        else:
            # No improvement from swap; try a few more random swaps before giving up
            if swap_count > 10:
                break
    
    # Final gradient polish
    if time.time() - start_time < time_budget:
        h_final, psl_final = _fast_psl_optimize(best_h, max_passes=100, time_budget=time_budget * 0.1, start_time=start_time)
        if psl_final < best_psl:
            best_psl = psl_final
            best_h = h_final
    
    return best_h, best_psl


def _simulated_annealing(h, time_budget=10.0, start_time=None, rng=None):
    """
    Simulated annealing with both sign flips and swap moves.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 5.0
    T_min = 0.1
    cooling = 0.97
    
    while T > T_min and time.time() - start_time < time_budget:
        # Randomly choose move type: flip (70%) or swap (30%)
        if rng.random() < 0.7:
            # Sign flip
            i = rng.integers(n)
            h[i] = -h[i]
            acf = _compute_acf(h)
            new_psl = _psl_value(acf, n)
        else:
            # Swap move
            h, new_psl, _ = _swap_move(h, acf, rng)
            acf = _compute_acf(h)
        
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or rng.random() < np.exp(-delta_psl / max(T, 1e-10)):
            current_psl = new_psl
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_psl


def _make_sparse_init(rng, K):
    """Create a sparse initialization with K active positions."""
    h = np.zeros(N)
    active = rng.choice(N, size=K, replace=False)
    signs = rng.choice([-1.0, 1.0], size=K)
    h[active] = signs
    return h


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the peak sidelobe level (upper bound on C3).
    
    Strategy:
    1. Try multiple sparse initializations with varying K (sparsity levels).
    2. For each, run combined sparse+gradient optimization.
    3. Use simulated annealing to escape local minima.
    4. Return the best sequence found.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Sparsity levels to try: from sparse to dense
    sparsity_levels = [50, 80, 100, 120, 150, 200, 250, 300, 350, 400]
    
    # Try multiple seeds for each sparsity level
    n_seeds = 3
    
    for K in sparsity_levels:
        if time.time() > deadline - 20:
            break
        
        for seed in range(n_seeds):
            if time.time() > deadline - 20:
                break
            
            rng = np.random.default_rng(seed * 1000 + K * 7 + 42)
            
            if K == N:
                # Dense: all positions active
                h_init = rng.choice([-1.0, 1.0], size=N)
            else:
                h_init = _make_sparse_init(rng, K)
            
            # Run combined sparse+gradient optimization
            time_budget = min(8.0, deadline - 20 - (time.time() - start_time))
            if time_budget < 1.0:
                break
            
            h_opt, score = _sparse_optimize(h_init, time_budget=time_budget, start_time=start_time, rng=rng)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Phase 2: Simulated annealing from the best found
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(30.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            for sa_run in range(4):
                if time.time() > deadline - 5:
                    break
                
                rng = np.random.default_rng(5000 + sa_run * 13)
                
                # Start SA from best_h with some perturbation
                h_perturbed = best_h.copy()
                n_flips = max(5, N // 25)
                flip_positions = rng.choice(N, size=n_flips, replace=False)
                h_perturbed[flip_positions] *= -1
                
                sa_budget = min(sa_time / (4 - sa_run) if sa_run < 3 else sa_time, 8.0)
                if sa_budget < 1.0:
                    break
                
                h_sa, score_sa = _simulated_annealing(h_perturbed, time_budget=sa_budget, start_time=start_time, rng=rng)
                
                if score_sa < best_score:
                    best_score = score_sa
                    best_h = h_sa.copy()
                
                # Also try SA directly from best_h
                if time.time() < deadline - 5:
                    h_sa2, score_sa2 = _simulated_annealing(best_h, time_budget=min(5.0, deadline - 5 - (time.time() - start_time)), start_time=start_time, rng=rng)
                    if score_sa2 < best_score:
                        best_score = score_sa2
                        best_h = h_sa2.copy()
    
    # Phase 3: Final gradient polish
    if best_h is not None and time.time() < deadline - 2:
        final_budget = deadline - 2 - (time.time() - start_time)
        if final_budget > 0.5:
            h_final, score_final = _fast_psl_optimize(best_h, max_passes=500, time_budget=final_budget, start_time=start_time)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
    
    if best_h is None:
        # Fallback: return a random sequence
        best_h = np.random.choice([-1.0, 1.0], size=N)
    
    return best_h

# EVOLVE-BLOCK-END
