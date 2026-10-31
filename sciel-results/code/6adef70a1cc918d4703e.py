"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs_fft(x):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(x)
    # Use FFT to compute autoconvolution efficiently
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    autoconv = np.fft.irfft(X * X.conj(), size)[:2*n-1]
    # The linear autoconvolution is the first n elements (lags 0 to n-1)
    # Actually, irfft gives the circular convolution, but with zero-padding it's linear
    # The result is length size, and the linear autoconv is at indices 0 to n-1
    # Wait, np.correlate(x, x, 'full') gives lags from -(n-1) to n-1
    # For autocorrelation, c[k] = sum x[i]*x[i+k], k=0..n-1
    # Let's just use the first n elements of the autoconv
    return np.max(np.abs(autoconv[:n]))


def _score(x):
    """Compute the objective: max_abs_autoconv / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    max_ac = _autoconv_max_abs_fft(x)
    return max_ac / (s * s)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with all ones (known good: value = 1/N = 0.0025)
    x = np.ones(N)
    best_x = x.copy()
    best_score = _score(x)
    
    # Try local sign-flip optimization with FFT-based scoring
    improved = True
    passes = 0
    while improved and time.time() < deadline and passes < 20:
        improved = False
        passes += 1
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = x[i]
            x[i] = -old_val
            new_score = _score(x)
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_x = x.copy()
                improved = True
            else:
                x[i] = old_val
    
    # Try some structured initializations
    # Barker sequence for length 13, then tile
    barker_13 = np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    tiles = (N + len(barker_13) - 1) // len(barker_13)
    x_barker = np.tile(barker_13, tiles)[:N]
    s_barker = np.sum(x_barker)
    if abs(s_barker) > 1e-10:
        score_barker = _score(x_barker)
        if score_barker < best_score:
            best_score = score_barker
            best_x = x_barker.copy()
            # Local search from Barker
            for _ in range(5):
                if time.time() > deadline:
                    break
                imp = True
                while imp and time.time() < deadline:
                    imp = False
                    for i in range(N):
                        if time.time() > deadline:
                            break
                        old_val = best_x[i]
                        best_x[i] = -old_val
                        new_score = _score(best_x)
                        if new_score < best_score - 1e-12:
                            best_score = new_score
                            imp = True
                        else:
                            best_x[i] = old_val
    
    # Random restarts
    rng = np.random.default_rng(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        x_rand = rng.choice([-1.0, 1.0], size=N)
        s_rand = np.sum(x_rand)
        if abs(s_rand) < 50:  # Skip if integral too small
            continue
        score_rand = _score(x_rand)
        if score_rand < best_score:
            best_score = score_rand
            best_x = x_rand.copy()
            # Short local search
            for _ in range(3):
                if time.time() > deadline:
                    break
                imp = True
                while imp and time.time() < deadline:
                    imp = False
                    for i in range(N):
                        if time.time() > deadline:
                            break
                        old_val = best_x[i]
                        best_x[i] = -old_val
                        new_score = _score(best_x)
                        if new_score < best_score - 1e-12:
                            best_score = new_score
                            imp = True
                        else:
                            best_x[i] = old_val
    
    return best_x
# EVOLVE-BLOCK-END
