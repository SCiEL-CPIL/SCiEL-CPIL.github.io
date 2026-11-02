"""Optimal-control / flat-autocorrelation design for the Erdos minimum-overlap task.

We minimize the worst-case overlap of h with shifted (1-h).
With sum(h) fixed = N/2, overlap(k) = N/2 - autocorr(k), so minimizing
max_k overlap(k) is equivalent to maximizing min_k autocorr(k).
We therefore build a balanced 0/1 sequence with as flat (high) an
autocorrelation as possible via a two-point "Hamiltonian" local search
(swap-based simulated annealing) that directly drives up the minimum lag.
"""
import numpy as np

N = 100


def _autocorr_min(x):
    # x is 0/1 array of length N, sum = N/2
    # autocorr(k) = sum_i x[i]*x[(i+k)%N]
    # compute all lags efficiently via FFT (real, circular)
    n = len(x)
    # circular autocorrelation
    f = np.fft.fft(x.astype(np.float64))
    ac = np.fft.ifft(f * np.conj(f)).real
    # ac[k] = sum_i x[i] x[(i+k) mod n] (circular)
    return ac.min()


def _max_overlap(x):
    # overlap(k) = sum_i x[i]*(1 - x[(i+k)%N]); max over k (k>=1)
    n = len(x)
    f = np.fft.fft(x.astype(np.float64))
    ac = np.fft.ifft(f * np.conj(f)).real
    # overlap at lag k = sum x - ac[k] = n/2 - ac[k]
    # worst (max) overlap = n/2 - min_k ac[k] for k in 1..n-1
    ac_nonzero = ac[1:]
    return n / 2.0 - ac_nonzero.min()


def _swap_gain(x, i, j, ac):
    # gain in min-autocorr if we flip x[i] and x[j] (must be 1 and 0)
    # We just recompute min autocorr cheaply via delta update on ac.
    # ac[k] changes only where x[i],x[j] contribute.
    n = len(x)
    di = x[i]  # 1
    dj = x[j]  # 0
    # after swap: x[i]=0, x[j]=1
    # delta to ac[k] = sum over contributions
    # ac[k] = sum_t x[t] x[(t+k)%n]
    # terms involving i: x[i]*x[(i+k)%n] and x[(i-k)%n]*x[i]
    # We'll just recompute ac fully (N=100, cheap).
    f = np.fft.fft(x.astype(np.float64))
    new_ac = np.fft.ifft(f * np.conj(f)).real
    return new_ac[1:].min()


def construct():
    n = N
    # Start from Thue-Morse (flat autocorr, balanced) or checkerboard.
    # Thue-Morse: t[i] = popcount(i) mod 2
    def thue_morse(n):
        t = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # parity of bit count
            t[i] = bin(i).count('1') % 2
        return t

    x = thue_morse(n).astype(np.float64)
    # ensure exactly n/2 ones; adjust
    target = n // 2
    s = int(x.sum())
    if s > target:
        idx = np.where(x == 1)[0]
        for k in range(s - target):
            x[idx[k]] = 0
    elif s < target:
        idx = np.where(x == 0)[0]
        for k in range(target - s):
            x[idx[k]] = 1

    best = x.copy()
    best_min = _autocorr_min(best)

    # Simulated annealing over swaps of a 1 and a 0 (keeps sum fixed)
    rng = np.random.default_rng(12345)
    T = 2.0
    cooling = 0.999
    iters = 400000
    ones = np.where(x == 1)[0]
    zeros = np.where(x == 0)[0]

    for it in range(iters):
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        if i == j:
            continue
        x[i], x[j] = x[j], x[i]
        new_min = _autocorr_min(x)
        delta = new_min - best_min
        if delta > 0 or rng.random() < np.exp(delta / T):
            best_min = new_min
            best = x.copy()
            # update index sets
            ones = np.where(x == 1)[0]
            zeros = np.where(x == 0)[0]
        else:
            x[i], x[j] = x[j], x[i]
        T *= cooling
        if T < 1e-4:
            break

    # local polish: try all single swaps accepting only improvements
    improved = True
    while improved:
        improved = False
        ones = np.where(best == 1)[0]
        zeros = np.where(best == 0)[0]
        cur = _autocorr_min(best)
        for i in ones:
            for j in zeros:
                best[i], best[j] = best[j], best[i]
                nm = _autocorr_min(best)
                if nm > cur + 1e-12:
                    cur = nm
                    improved = True
                else:
                    best[i], best[j] = best[j], best[i]

    # Return step heights in [0,1] with sum exactly n/2.
    # We can optionally relax 0/1 to fractional for a slight overlap reduction,
    # but 0/1 already gives sum = n/2 exactly and is a valid step-height vector.
    h = best.astype(np.float64)
    # exactness guard
    if abs(h.sum() - n / 2.0) > 1e-12:
        # fix by nudging a single element
        diff = n / 2.0 - h.sum()
        h[0] += diff
    return h
