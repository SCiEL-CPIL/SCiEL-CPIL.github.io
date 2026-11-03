"""Search for a low-autocorrelation signed step function (400 equal intervals).

Uses Iterated Local Search (ILS) with fast greedy (first-improvement) local
search and a final best-improvement polish. This approach explores more of the
solution space by perturbing the current best solution and re-optimizing,
escaping local minima more effectively than pure simulated annealing.
"""
import numpy as np
import time

N = 400


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    tm = np.zeros(n, dtype=np.int8)
    tm[0] = 0
    for i in range(1, n):
        tm[i] = tm[i // 2] if (i % 2 == 0) else 1 - tm[i // 2]
    # Map 0 -> -1, 1 -> 1
    return 1.0 - 2.0 * tm.astype(np.float64)


def _calc_ac(h):
    """Calculate aperiodic autocorrelation using FFT."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.fft(h, fft_size)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe_from_ac(ac):
    """Calculate the peak aperiodic autocorrelation sidelobe."""
    if len(ac) <= 1:
        return 0.0
    return float(np.max(np.abs(ac[1:])))


def _flip_delta(h, i, ac_buf):
    """
    Compute the delta to add to ac[1:] when flipping h[i].
    Uses a pre-allocated buffer to reduce allocation overhead.
    """
    ac_buf[:] = 0.0
    
    # Contribution from h[i+k] for k=1 to N-1-i
    if i + 1 < N:
        ac_buf[1 : N - i] = -2.0 * h[i] * h[i + 1 : N]
    
    # Contribution from h[i-k] for k=1 to i
    if i > 0:
        ac_buf[1 : i + 1] += -2.0 * h[i] * h[i - 1 : -1 : -1]
    
    return ac_buf


def _fast_greedy_polish(h, ac, time_budget, start_time):
    """
    Fast greedy polish using 'first improvement' strategy.
    Scans indices 0..N-1. If a flip improves the peak, apply it and restart scan.
    This is much faster than 'best improvement' as it avoids full scans if
    improvements are found early.
    """
    h = h.copy()
    ac = ac.copy()
    cur_peak = _peak_sidelobe_from_ac(ac)
    best_h = h.copy()
    best_peak = cur_peak
    
    delta_buf = np.zeros(N, dtype=np.float64)
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            old_h_i = h[i]
            h[i] = -old_h_i
            
            # Compute delta
            _flip_delta(h, i, delta_buf)
            # Note: _flip_delta uses h[i] which is now flipped.
            # The formula in _flip_delta is -2 * h[i] * ...
            # If h[i] is flipped, the delta calculated is for the flip from -old to old?
            # Let's re-verify the delta logic.
            # Original R[k] = sum h_j h_{j+k}.
            # Flip h_i. New R[k] = R[k] - 2 h_i (h_{i+k} + h_{i-k}).
            # So delta = -2 h_i (h_{i+k} + h_{i-k}).
            # In _flip_delta, we pass h (with h_i flipped).
            # So h[i] in _flip_delta is -old_h_i.
            # So delta_calc = -2 * (-old_h_i) * (...) = 2 * old_h_i * (...).
            # But we want -2 * old_h_i * (...).
            # So we need to NEGATE the result of _flip_delta if we pass the flipped h.
            # OR, we pass the original h? No, h is modified in place.
            # Let's adjust _flip_delta to take old_h_i or just negate here.
            
            # Actually, looking at the previous code:
            # "delta = -2.0 * old_h_i * h[i + 1 : N]"
            # My _flip_delta uses h[i] (current).
            # So if I pass current h (flipped), _flip_delta gives 2*old_h_i*...
            # I need -2*old_h_i*...
            # So I must negate the delta_buf.
            
            delta_buf *= -1.0
            
            ac[1:] += delta_buf
            new_peak = _peak_sidelobe_from_ac(ac)
            
            if new_peak < cur_peak:
                cur_peak = new_peak
                if cur_peak < best_peak:
                    best_peak = cur_peak
                    best_h = h.copy()
                improved = True
                break  # Restart scan from 0
            else:
                # Revert
                h[i] = old_h_i
                ac[1:] -= delta_buf
        
        if not improved:
            break
            
    return best_h, best_peak


def _best_greedy_polish(h, ac, time_budget, start_time):
    """
    Best improvement greedy polish.
    Finds the single flip that most reduces the peak, applies it, repeats.
    Slower but higher quality.
    """
    h = h.copy()
    ac = ac.copy()
    cur_peak = _peak_sidelobe_from_ac(ac)
    best_h = h.copy()
    best_peak = cur_peak
    
    delta_buf = np.zeros(N, dtype=np.float64)
    
    while time.time() - start_time < time_budget:
        best_i = -1
        best_new_peak = cur_peak
        
        for i in range(N):
            old_h_i = h[i]
            h[i] = -old_h_i
            
            _flip_delta(h, i, delta_buf)
            delta_buf *= -1.0  # Correct sign as discussed
            
            ac[1:] += delta_buf
            new_peak = _peak_sidelobe_from_ac(ac)
            
            if new_peak < best_new_peak:
                best_new_peak = new_peak
                best_i = i
            
            # Revert
            h[i] = old_h_i
            ac[1:] -= delta_buf
        
        if best_i < 0:
            break  # Local minimum
        
        # Apply best flip
        i = best_i
        old_h_i = h[i]
        h[i] = -old_h_i
        _flip_delta(h, i, delta_buf)
        delta_buf *= -1.0
        ac[1:] += delta_buf
        cur_peak = best_new_peak
        
        if cur_peak < best_peak:
            best_peak = cur_peak
            best_h = h.copy()
            
    return best_h, best_peak


def construct():
    try:
        # Safety margin: 145s out of 150s
        hard_deadline = time.time() + 145.0
        start_time = time.time()
        
        # 1. Initialize with Thue-Morse
        h_init = _thue_morse(N)
        ac_init = _calc_ac(h_init)
        
        # 2. Initial Fast Polish
        # Give 10s for initial polish
        budget_initial = min(10.0, hard_deadline - (time.time() - start_time) - 5.0)
        if budget_initial > 1.0:
            h_best, peak_best = _fast_greedy_polish(h_init, ac_init, budget_initial, time.time())
        else:
            h_best = h_init
            peak_best = _peak_sidelobe_from_ac(ac_init)
        
        # 3. Iterated Local Search (ILS)
        # Loop: Perturb -> Fast Polish -> Update Best
        # Continue until time is up
        rng = np.random.default_rng(42)
        perturbation_size = 5
        
        while time.time() < hard_deadline - 10.0:  # Leave 10s for final polish
            # Perturb
            h_perturb = h_best.copy()
            indices = rng.choice(N, size=perturbation_size, replace=False)
            h_perturb[indices] *= -1.0
            
            ac_perturb = _calc_ac(h_perturb)
            
            # Fast Polish
            # Allocate time based on remaining budget, e.g., 5s per iteration
            budget_iter = min(5.0, hard_deadline - (time.time() - start_time) - 10.0)
            if budget_iter < 1.0:
                break
                
            h_new, peak_new = _fast_greedy_polish(h_perturb, ac_perturb, budget_iter, time.time())
            
            if peak_new < peak_best:
                peak_best = peak_new
                h_best = h_new
                # Optionally reduce perturbation size if we found a good one?
                # Or keep it constant.
                perturbation_size = 5
            else:
                # If no improvement, slightly increase perturbation to escape
                # but not too much to avoid destroying the structure
                if perturbation_size < 15:
                    perturbation_size += 1
                else:
                    perturbation_size = 5 # Reset to small if stuck at max? Or keep large?
                    # Let's try randomizing it between 5 and 15
                    perturbation_size = int(rng.integers(5, 15))
        
        # 4. Final Best Polish
        # Use the remaining time for a high-quality polish
        remaining_time = hard_deadline - (time.time() - start_time) - 1.0
        if remaining_time > 1.0:
            ac_best = _calc_ac(h_best)
            h_final, peak_final = _best_greedy_polish(h_best, ac_best, remaining_time, time.time())
            if peak_final < peak_best:
                h_best = h_final
                peak_best = peak_final
        
        return np.ascontiguousarray(h_best, dtype=np.float64)
        
    except Exception:
        return np.ones(N)
