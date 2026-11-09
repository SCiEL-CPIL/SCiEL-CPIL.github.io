"""Direct construction: signed step heights minimizing max|autoconv| / (sum)^2."""
import numpy as np
import time

N = 400


def _compute_autoconv(f):
    """Compute full autoconvolution via FFT"""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, size)[:2 * n - 1]
    return auto


def _update_autoconv(auto, f, i, delta):
    """
    Update autoconvolution when f[i] changes by delta.
    (f + delta*delta_i) * (f + delta*delta_i) = f*f + delta*(delta_i*f + f*delta_i) + delta^2*delta_i*delta_i
    delta_i * f is f shifted right by i (positions i..i+n-1 get f[0..n-1])
    f * delta_i is f shifted left by i (positions 0..n-1-i get f[i..n-1])
    """
    n = len(f)
    # delta_i * f: f shifted right by i
    # Position j in result gets f[j-i] if i <= j < i+n
    left = np.zeros(2*n-1)
    left[i:i+n] = f
    # f * delta_i: f shifted left by i  
    # Position j in result gets f[j+i] if 0 <= j < n-i
    right = np.zeros(2*n-1)
    right[:n-i] = f[i:]
    # delta^2 * delta_i * delta_i: delta at position 2i
    diag = np.zeros(2*n-1)
    if 0 <= 2*i < 2*n-1:
        diag[2*i] = 1.0
    
    return auto + delta * (left + right) + delta * delta * diag


def _score_from_auto(auto, s):
    """Compute score from autoconvolution and sum"""
    if abs(s) < 1e-12:
        return float('inf')
    max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)


def _try_periodic_patterns(deadline, best_f, best_score, best_auto, best_s):
    """Try periodic patterns and return updated best values."""
    for period in range(2, 40):
        if time.time() > deadline:
            break
        for n_flip in range(1, min(period // 2 + 1, 10)):
            if time.time() > deadline:
                break
            for trial in range(8):
                if time.time() > deadline:
                    break
                pattern = np.ones(period)
                rng_p = np.random.default_rng(trial * 1000 + period * 10 + n_flip)
                flip_idx = rng_p.choice(period, size=n_flip, replace=False)
                pattern[flip_idx] = -1
                f = np.tile(pattern, N // period + 1)[:N]
                auto = _compute_autoconv(f)
                s = np.sum(f)
                score = _score_from_auto(auto, s)
                if score < best_score:
                    best_score = score
                    best_f = f.copy()
                    best_auto = auto.copy()
                    best_s = s
    return best_f, best_score, best_auto, best_s


def _local_search(f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng):
    """Greedy local search with incremental autoconv update."""
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 10:
                break
            old_val = f[i]
            delta = -2 * old_val  # flip from +1 to -1 or vice versa
            new_auto = _update_autoconv(auto, f, i, delta)
            new_s = s + delta
            new_score = _score_from_auto(new_auto, new_s)
            if new_score < current_score - 1e-12:
                f[i] = old_val + delta
                auto = new_auto
                s = new_s
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    best_auto = auto.copy()
                    best_s = s
                improved = True
    return f, auto, s, current_score, best_score, best_f, best_auto, best_s


def _simulated_annealing(f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng):
    """Simulated annealing with incremental autoconv update."""
    start_temp = 0.05
    end_temp = 0.0005
    start_time = time.time()
    max_time = deadline - 3 - start_time
    
    if max_time <= 0:
        return f, auto, s, current_score, best_score, best_f, best_auto, best_s
    
    while time.time() < deadline - 3:
        progress = (time.time() - start_time) / max(max_time, 1e-9)
        temp = start_temp * (end_temp / start_temp) ** min(progress, 1.0)
        i = rng.integers(N)
        old_val = f[i]
        delta = -2 * old_val
        new_auto = _update_autoconv(auto, f, i, delta)
        new_s = s + delta
        new_score = _score_from_auto(new_auto, new_s)
        
        if new_score < current_score:
            f[i] = old_val + delta
            auto = new_auto
            s = new_s
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
                best_auto = auto.copy()
                best_s = s
        else:
            delta_score = new_score - current_score
            if delta_score > 0 and temp > 0:
                if rng.random() < np.exp(-delta_score / temp):
                    f[i] = old_val + delta
                    auto = new_auto
                    s = new_s
                    current_score = new_score
                # else reject
            # else reject
    return f, auto, s, current_score, best_score, best_f, best_auto, best_s


def construct():
    deadline = time.time() + 115  # safety margin before 120s kill
    
    # Direct construction: start with all +1
    best_f = np.ones(N)
    best_auto = _compute_autoconv(best_f)
    best_s = np.sum(best_f)
    best_score = _score_from_auto(best_auto, best_s)
    
    # Try periodic patterns directly
    best_f, best_score, best_auto, best_s = _try_periodic_patterns(
        deadline, best_f, best_score, best_auto, best_s)
    
    # Local search with incremental autoconv update
    if time.time() < deadline - 20:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng = np.random.default_rng(123)
        
        f, auto, s, current_score, best_score, best_f, best_auto, best_s = _local_search(
            f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng)
    
    # Simulated annealing with incremental autoconv update
    if time.time() < deadline - 5:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng = np.random.default_rng(456)
        
        f, auto, s, current_score, best_score, best_f, best_auto, best_s = _simulated_annealing(
            f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng)
    
    # Final local polish
    if time.time() < deadline - 2:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng = np.random.default_rng(789)
        
        f, auto, s, current_score, best_score, best_f, best_auto, best_s = _local_search(
            f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng)
    
    return best_f
