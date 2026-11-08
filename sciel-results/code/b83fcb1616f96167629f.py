import numpy as np
import time

N = 400

def _autocorr(h):
    """Compute circular autocorrelation via FFT. Returns array of length N."""
    H = np.fft.rfft(h)
    R = np.fft.irfft(np.abs(H) ** 2)
    return R

def _score(h):
    """Max absolute autocorrelation at non-zero lags, normalized by N."""
    R = _autocorr(h)
    # R[0] is the energy; we want max |R[k]| for k != 0
    return np.max(np.abs(R[1:])) / N

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_h = None
    best_score = np.inf

    # Try multiple random restarts with local search
    np.random.seed(42)
    n_restarts = 100  # will be limited by deadline

    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break

        # Random ±1 start
        h = np.random.choice([-1.0, 1.0], size=N)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

        # Local search: flip single bits
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try all positions, but in random order to avoid bias
            order = np.random.permutation(N)
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                old_val = h[i]
                h[i] = -old_val
                s_new = _score(h)
                if s_new < best_score:
                    best_score = s_new
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = old_val  # revert

    if best_h is None:
        best_h = np.ones(N)

    return best_h
