import numpy as np
import time

N = 100
HALF = N // 2
DEADLINE = 110.0

def _autocorr(h):
    H = np.fft.fft(h)
    return np.fft.ifft(H * np.conj(H)).real

def _max_overlap(h):
    r = _autocorr(h)
    return HALF - float(np.min(r[1:]))

def _random_bitseq(rng):
    x = rng.random(N)
    # pick top HALF as 1
    idx = np.argsort(x)
    seq = np.zeros(N)
    seq[idx[HALF:]] = 1.0
    return seq

def _sa(rng, start, iters, T0, T1):
    h = start.copy()
    cur = _max_overlap(h)
    best = h.copy()
    bestc = cur
    for it in range(iters):
        # swap a random 1 and a random 0
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        h[i], h[j] = h[j], h[i]
        newc = _max_overlap(h)
        T = T0 * (T1 / T0) ** (it / max(1, iters - 1))
        if newc < cur or rng.random() < np.exp(-(newc - cur) / max(T, 1e-9)):
            cur = newc
            if newc < bestc:
                bestc = newc
                best = h.copy()
        else:
            h[i], h[j] = h[j], h[i]  # revert
    return best, bestc

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best = None
    bestc = np.inf
    # diverse starts
    starts = []
    # alternating
    alt = np.zeros(N); alt[::2] = 1.0
    starts.append(alt)
    # block
    blk = np.zeros(N); blk[:HALF] = 1.0
    starts.append(blk)
    # several random
    for _ in range(6):
        starts.append(_random_bitseq(rng))
    # run SA on each within deadline
    for s in starts:
        if time.time() - t0 > DEADLINE - 20:
            break
        # adaptive iters
        h, c = _sa(rng, s, iters=20000, T0=2.0, T1=0.05)
        if c < bestc:
            bestc = c
            best = h
    if best is None:
        best = np.full(N, 0.5)
    return best
