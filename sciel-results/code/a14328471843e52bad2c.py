"""Spectral Barker-like initialization with greedy peak-shaving for
signed step heights minimizing max absolute autoconvolution / (sum)^2."""

import time
import numpy as np

N = 400
DEADLINE = 140.0  # wall-clock safety margin before 150s kill


def _fft_size(n):
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    return size


def _offmax_from_fft(F, n, size):
    """Compute max absolute off-zero autocorrelation from FFT."""
    power = F * np.conj(F)
    ac = np.fft.irfft(power, size)[:2 * n - 1]
    off = np.concatenate([ac[:n - 1], ac[n:]])
    return float(np.max(np.abs(off)))


def _objective(f, size):
    """max absolute off-zero autoconvolution divided by (sum f)^2."""
    s = f.sum()
    s2 = s * s
    if abs(s2) < 1e-9:
        return float('inf')
    F = np.fft.rfft(f, size)
    offmax = _offmax_from_fft(F, len(f), size)
    return offmax / s2


def _spectral_init(rng, n, size, num_candidates=200):
    """Generate candidates via inverse FFT of phase-only masks with flat magnitude."""
    candidates = []
    t = np.arange(n)
    
    # Barker-like: flat magnitude spectrum, random phase
    for _ in range(num_candidates):
        if time.time() > DEADLINE:
            break
        # Random phase in frequency domain
        freq_idx = np.arange(size // 2 + 1)
        # Keep DC component real and positive
        phase = rng.uniform(0, 2 * np.pi, size // 2 + 1)
        phase[0] = 0.0
        # Flat magnitude (all 1s)
        F_mask = np.exp(1j * phase)
        # Ensure Hermitian symmetry for real signal
        # rfft gives half spectrum; irfft reconstructs
        signal = np.fft.irfft(F_mask, size)[:n]
        # Sign quantize
        cand = np.sign(signal)
        cand[cand == 0] = 1.0
        s = cand.sum()
        if abs(s) > 1.0:
            candidates.append(cand.astype(np.float64))
    
    # Also try some structured low-frequency patterns
    for freq in [1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 19, 23, 29, 31, 37, 41]:
        if time.time() > DEADLINE:
            break
        for phase_offset in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]:
            cand = np.sign(np.cos(2 * np.pi * freq * t / n + phase_offset))
            cand[cand == 0] = 1.0
            s = cand.sum()
            if abs(s) > 1.0:
                candidates.append(cand.astype(np.float64))
    
    # Random binary
    for _ in range(100):
        if time.time() > DEADLINE:
            break
        cand = rng.choice([-1.0, 1.0], size=n)
        s = cand.sum()
        if abs(s) > 1.0:
            candidates.append(cand.astype(np.float64))
    
    return candidates


def _greedy_peak_shaving(cand, size, max_iters=500, time_budget=20.0):
    """
    Greedy algorithm: iteratively flip the element most correlated with
    the current maximum autoconvolution peak to reduce the peak.
    """
    start_time = time.time()
    cur = cand.copy().astype(np.float64)
    cur_sum = cur.sum()
    cur_F = np.fft.rfft(cur, size)
    cur_offmax = _offmax_from_fft(cur_F, N, size)
    cur_obj = cur_offmax / (cur_sum * cur_sum)
    
    best_cur = cur.copy()
    best_obj = cur_obj
    
    # Precompute phase factors for FFT update
    k_vals = np.arange(size // 2 + 1)
    # phase_factors[i, k] = exp(-2*pi*i*k*i/size) for position i, frequency k
    # We need: when flipping f[i] from v to -v, delta F[k] = -2*v*exp(-2*pi*i*k*i/size)
    positions = np.arange(N)
    # For each position i, the phase vector over frequencies k
    phase_matrix = np.exp(-2j * np.pi * np.outer(positions, k_vals) / size)
    
    # Identify the lag of the maximum absolute off-zero autocorrelation
    def _get_peak_lag(F):
        power = F * np.conj(F)
        ac = np.fft.irfft(power, size)[:2 * N - 1]
        off = np.concatenate([ac[:N - 1], ac[N:]])
        abs_off = np.abs(off)
        peak_idx = int(np.argmax(abs_off))
        # Map back to lag: indices 0..N-2 are lags 1..N-1, indices N-1..2N-3 are lags -(N-1)..-1
        if peak_idx < N - 1:
            lag = peak_idx + 1
        else:
            lag = peak_idx - (N - 1)
            lag = -lag
        return lag, abs_off[peak_idx]
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        # Find the peak lag
        peak_lag, peak_val = _get_peak_lag(cur_F)
        
        if peak_val < 1e-10:
            break
        
        # For a given lag L, the autocorrelation at lag L is:
        # ac[L] = sum_j f[j] * f[j+L] (for L > 0)
        # Flipping f[i] changes ac[L] by:
        #   if i and i+L are both in range: -2*f[i]*f[i+L] (if L>0)
        #   if i and i-L are both in range: -2*f[i]*f[i-L] (if L>0)
        # So the change in ac[L] from flipping f[i] is:
        #   delta = -2*f[i]*(f[i+L] if i+L<N else 0 + f[i-L] if i-L>=0 else 0)
        # We want to flip the i that maximizes the reduction in |ac[L]|.
        # If ac[L] > 0, we want delta < 0, i.e., f[i]*(neighbors) > 0
        # If ac[L] < 0, we want delta > 0, i.e., f[i]*(neighbors) < 0
        
        # Compute the contribution for each i
        if peak_lag > 0:
            # ac[peak_lag] involves pairs (j, j+peak_lag)
            # Flipping f[i] affects terms where i is j or j+peak_lag
            # Term where i = j: f[i]*f[i+peak_lag]
            # Term where i = j+peak_lag: f[i-peak_lag]*f[i]
            contrib = np.zeros(N)
            # For i from 0 to N-1-peak_lag: f[i]*f[i+peak_lag]
            valid = slice(0, N - peak_lag)
            contrib[valid] += cur[valid] * cur[peak_lag:]
            # For i from peak_lag to N-1: f[i-peak_lag]*f[i]
            valid2 = slice(peak_lag, N)
            contrib[valid2] += cur[:-peak_lag] * cur[valid2]
        else:
            # peak_lag < 0, let L = -peak_lag > 0
            # ac[peak_lag] = ac[-L] = sum_j f[j]*f[j-L]
            # Same as ac[L] by symmetry for real signals? Actually ac[-L] = ac[L] for real
            # So we can just use |peak_lag|
            L = -peak_lag
            contrib = np.zeros(N)
            valid = slice(0, N - L)
            contrib[valid] += cur[valid] * cur[L:]
            valid2 = slice(L, N)
            contrib[valid2] += cur[:-L] * cur[valid2]
        
        # The change in ac[peak_lag] when flipping f[i] is -2*contrib[i]
        # We want to minimize |ac[peak_lag] - 2*contrib[i]|
        # i.e., choose i to make ac[peak_lag] - 2*contrib[i] as close to 0 as possible
        # Or more precisely, we want to reduce the peak value
        
        # Current peak value (signed)
        power = cur_F * np.conj(cur_F)
        ac = np.fft.irfft(power, size)[:2 * N - 1]
        if peak_lag > 0:
            current_peak_signed = ac[peak_lag]
        else:
            # ac index for lag -L is 2N-1-L
            current_peak_signed = ac[2 * N - 1 + peak_lag]
        
        # After flipping i, new ac[peak_lag] = current_peak_signed - 2*contrib[i]
        # We want to minimize |current_peak_signed - 2*contrib[i]|
        new_vals = current_peak_signed - 2 * contrib
        # Choose i that minimizes |new_vals[i]|
        best_i = int(np.argmin(np.abs(new_vals)))
        
        # But we also need to check that the flip doesn't create a larger peak elsewhere
        # For efficiency, let's just try the best candidate and verify
        old_val = cur[best_i]
        new_sum = cur_sum - 2 * old_val
        if abs(new_sum) < 1e-9:
            # Try next best
            abs_new = np.abs(new_vals)
            abs_new[best_i] = np.inf
            best_i = int(np.argmin(abs_new))
            old_val = cur[best_i]
            new_sum = cur_sum - 2 * old_val
            if abs(new_sum) < 1e-9:
                break
        
        # Apply flip
        cur[best_i] = -old_val
        cur_sum = new_sum
        
        # Update FFT
        delta = -2 * old_val * phase_matrix[best_i, :]
        cur_F = cur_F + delta
        
        # Compute new objective
        new_offmax = _offmax_from_fft(cur_F, N, size)
        new_obj = new_offmax / (cur_sum * cur_sum)
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_cur = cur.copy()
        else:
            # Revert if it made things worse
            cur[best_i] = old_val
            cur_sum += 2 * old_val
            cur_F = cur_F - delta
            # Try a few more candidates
            abs_new = np.abs(new_vals)
            abs_new[best_i] = np.inf
            # Get top 5 candidates
            top5 = np.argsort(abs_new)[:5]
            found_improvement = False
            for trial_i in top5:
                trial_val = cur[trial_i]
                trial_sum = cur_sum - 2 * trial_val
                if abs(trial_sum) < 1e-9:
                    continue
                trial_F = cur_F - 2 * trial_val * phase_matrix[trial_i, :]
                trial_offmax = _offmax_from_fft(trial_F, N, size)
                trial_obj = trial_offmax / (trial_sum * trial_sum)
                if trial_obj < best_obj:
                    cur[trial_i] = -trial_val
                    cur_sum = trial_sum
                    cur_F = trial_F
                    best_obj = trial_obj
                    best_cur = cur.copy()
                    found_improvement = True
                    break
            if not found_improvement:
                break
    
    return best_cur, best_obj


def _local_search_flip(cand, size, time_budget=15.0):
    """Local search: try single flips that reduce the objective."""
    start_time = time.time()
    cur = cand.copy().astype(np.float64)
    cur_sum = cur.sum()
    cur_F = np.fft.rfft(cur, size)
    cur_offmax = _offmax_from_fft(cur_F, N, size)
    cur_obj = cur_offmax / (cur_sum * cur_sum)
    
    best_cur = cur.copy()
    best_obj = cur_obj
    
    k_vals = np.arange(size // 2 + 1)
    positions = np.arange(N)
    phase_matrix = np.exp(-2j * np.pi * np.outer(positions, k_vals) / size)
    
    improved = True
    while improved and (time.time() - start_time < time_budget) and (time.time() < DEADLINE):
        improved = False
        best_gain = 0.0
        best_idx = -1
        best_new_obj = cur_obj
        best_new_sum = cur_sum
        best_new_F = cur_F
        
        # Sample positions
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        indices = rng.choice(N, size=min(N, 80), replace=False)
        
        for i in indices:
            if time.time() > DEADLINE:
                break
            old_val = cur[i]
            new_sum = cur_sum - 2 * old_val
            if abs(new_sum) < 1e-9:
                continue
            delta = -2 * old_val * phase_matrix[i, :]
            new_F = cur_F + delta
            new_offmax = _offmax_from_fft(new_F, N, size)
            new_obj = new_offmax / (new_sum * new_sum)
            gain = cur_obj - new_obj
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_new_obj = new_obj
                best_new_sum = new_sum
                best_new_F = new_F
        
        if best_idx >= 0:
            cur[best_idx] = -cur[best_idx]
            cur_sum = best_new_sum
            cur_F = best_new_F
            cur_obj = best_new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_cur = cur.copy()
            improved = True
    
    return best_cur, best_obj


def construct():
    start = time.time()
    size = _fft_size(N)
    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral initialization
    candidates = _spectral_init(rng, N, size, num_candidates=150)
    
    if not candidates:
        return np.ones(N, dtype=np.float64)
    
    # Score all candidates
    scored = []
    for cand in candidates:
        s = cand.sum()
        if abs(s) < 1.0:
            continue
        obj = _objective(cand, size)
        scored.append((obj, cand, s))
    
    scored.sort(key=lambda x: x[0])
    
    if not scored:
        return np.ones(N, dtype=np.float64)
    
    best_obj = scored[0][0]
    best_cand = scored[0][1]
    
    # Phase 2: Greedy peak shaving on top candidates
    time_per_candidate = min(8.0, (DEADLINE - 5 - (time.time() - start)) / max(1, len(scored[:15])))
    
    for obj_val, cand, s in scored[:15]:
        if time.time() > DEADLINE - 5:
            break
        remaining = DEADLINE - 5 - (time.time() - start)
        if remaining < 2.0:
            break
        tb = min(time_per_candidate, remaining * 0.6)
        if tb < 1.0:
            break
        
        shaved, shaved_obj = _greedy_peak_shaving(cand, size, max_iters=300, time_budget=tb)
        if shaved_obj < best_obj:
            best_obj = shaved_obj
            best_cand = shaved
    
    # Phase 3: Local search on best candidate
    if time.time() < DEADLINE - 10:
        ls_time = min(25.0, DEADLINE - 10 - (time.time() - start))
        if ls_time > 2.0:
            refined, refined_obj = _local_search_flip(best_cand, size, time_budget=ls_time)
            if refined_obj < best_obj:
                best_obj = refined_obj
                best_cand = refined
    
    # Phase 4: More greedy shaving if time remains
    if time.time() < DEADLINE - 5:
        remaining = DEADLINE - 5 - (time.time() - start)
        if remaining > 3.0:
            shaved2, shaved2_obj = _greedy_peak_shaving(best_cand, size, max_iters=200, time_budget=remaining * 0.8)
            if shaved2_obj < best_obj:
                best_obj = shaved2_obj
                best_cand = shaved2
    
    return best_cand
