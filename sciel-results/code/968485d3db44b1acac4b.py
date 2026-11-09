"""Direct construction: signed step heights minimizing max|autoconv| / (sum)^2."""
import numpy as np
import time

N = 400


def _autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, size)[:2 * n - 1]
    return auto


def _objective(f):
    """Compute max(|autoconv|) / (sum(f))^2"""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    auto = _autoconv(f)
    max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)


def _objective_fast(f, auto):
    """Compute objective given precomputed autoconv."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with all +1
    f = np.ones(N)
    best_f = f.copy()
    best_score = _objective(f)
    
    # Local search: flip individual bits to improve
    # Key: flipping bit i changes autoconv in a predictable way
    # autoconv_new = autoconv_old + 2*f[i]*f + (f[i]_old - f[i]_new)*something
    # Actually, let's just recompute
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Greedy local search
    improved = True
    while improved and time.time() < deadline - 40:
        improved = False
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 40:
                break
            old = f[i]
            f[i] = -old
            new_score = _objective(f)
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_f = f.copy()
                improved = True
            else:
                f[i] = old
    
    # Phase 2: Simulated annealing with restarts
    if time.time() < deadline - 10:
        f = best_f.copy()
        current_score = best_score
        start_temp = 0.5
        end_temp = 0.001
        n_iters = 50000
        start_time = time.time()
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            progress = it / n_iters
            temp = start_temp * (end_temp / start_temp) ** progress
            i = rng.integers(N)
            old = f[i]
            f[i] = -old
            new_score = _objective(f)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
            else:
                delta = new_score - current_score
                if delta > 0 and temp > 0:
                    if rng.random() < np.exp(-delta / temp):
                        current_score = new_score
                    else:
                        f[i] = old
                else:
                    f[i] = old
    
    # Phase 3: Final local polish
    if time.time() < deadline - 5:
        f = best_f.copy()
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            order = rng.permutation(N)
            for i in order:
                if time.time() > deadline - 2:
                    break
                old = f[i]
                f[i] = -old
                new_score = _objective(f)
                if new_score < best_score - 1e-12:
                    best_score = new_score
                    best_f = f.copy()
                    improved = True
                else:
                    f[i] = old
    
    return best_f
