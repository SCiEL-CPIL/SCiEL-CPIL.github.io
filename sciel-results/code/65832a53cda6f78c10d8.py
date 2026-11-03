import time
import numpy as np

N = 100
_DEADLINE = time.time() + 110  # safety margin before 120s

def _autocorr(h):
    # circular autocorrelation via FFT
    f = np.fft.fft(h)
    ac = np.fft.ifft(f * np.conj(f)).real
    return ac

def _max_overlap(h):
    ac = _autocorr(h)
    # O_k = N/2 - ac[k] for k in 1..N-1
    return N/2 - np.min(ac[1:])  # max O = N/2 - min ac

def construct():
    global _DEADLINE
    rng = np.random.default_rng(0)
    # balanced binary start
    h = np.zeros(N, dtype=float)
    idx = rng.permutation(N)
    h[idx[:N//2]] = 1.0
    best = h.copy()
    best_val = _max_overlap(best)
    cur = h.copy()
    cur_val = best_val
    T = 5.0
    step = 0
    while time.time() < _DEADLINE:
        # flip one 1 and one 0
        ones = np.flatnonzero(cur == 1.0)
        zeros = np.flatnonzero(cur == 0.0)
        if len(ones)==0 or len(zeros)==0: break
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        cur[i] = 0.0; cur[j] = 1.0
        new_val = _max_overlap(cur)
        delta = new_val - cur_val  # we minimize
        if delta < 0 or rng.random() < np.exp(-delta / max(T,1e-9)):
            cur_val = new_val
            if new_val < best_val:
                best_val = new_val
                best = cur.copy()
        else:
            cur[i] = 1.0; cur[j] = 0.0
        step += 1
        T = max(0.05, T * 0.9999)
    # return list of floats
    return [float(x) for x in best]
