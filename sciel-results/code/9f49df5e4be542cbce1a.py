import numpy as np
import time

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).

    Overlap at shift k = sum_i h[i]*(1-h[(i+k)%N]) = N/2 - autocorr(k).
    So minimizing max overlap == maximizing the minimum off-peak autocorrelation.

    A balanced binary sequence (equal 0s and 1s, sum = N/2) with low
    off-peak autocorrelation beats the constant h=0.5 baseline (score 1.0).
    We search over balanced binary sequences via pair-flip local search.
    """
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s

    N = 100
    half = N // 2  # need exactly half 1s for sum = N/2

    def autocorr_offpeak(h):
        """Return off-peak autocorrelation values for shifts 1..N-1."""
        h_pad = np.concatenate([h, np.zeros(N)])
        F = np.fft.fft(h_pad, n=2 * N)
        ac = np.real(np.fft.ifft(F * np.conj(F)))[:N]
        return ac[1:]

    def peak_overlap(h):
        """Max overlap = N/2 - min(off-peak autocorr)."""
        ac = autocorr_offpeak(h)
        return N / 2.0 - np.min(ac)

    def normalized_score_value(h):
        """The raw overlap value (lower is better)."""
        return peak_overlap(h) / (N / 4.0)

    def make_balanced_random(seed):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        h = np.zeros(N)
        h[idx[:half]] = 1.0
        return h

    def try_flip_pair(h, i, j):
        """Flip h[i] 0->1 and h[j] 1->0 (or vice versa), preserving sum."""
        h[i] = 1.0 - h[i]
        h[j] = 1.0 - h[j]

    def local_search(h, time_limit):
        """Pair-flip local search minimizing peak overlap."""
        best = peak_overlap(h)
        best_h = h.copy()
        n = len(h)
        # Precompute list of 1-indices and 0-indices for candidate pairs
        while time.time() < time_limit:
            improved = False
            ones = np.where(h > 0.5)[0]
            zeros = np.where(h < 0.5)[0]
            # Sample candidate pairs to keep it fast
            rng = np.random.default_rng()
            n_pairs = 4000
            for _ in range(n_pairs):
                if time.time() > time_limit:
                    break
                i = ones[rng.integers(len(ones))]
                j = zeros[rng.integers(len(zeros))]
                # flip: i (1->0), j (0->1)
                h[i] = 0.0
                h[j] = 1.0
                val = peak_overlap(h)
                if val < best - 1e-12:
                    best = val
                    best_h = h.copy()
                    improved = True
                else:
                    # revert
                    h[i] = 1.0
                    h[j] = 0.0
            if not improved:
                # stuck; random restart within this local search to escape
                # do a few random pair flips (hill-climbing kick)
                for _ in range(50):
                    i = ones[rng.integers(len(ones))]
                    j = zeros[rng.integers(len(zeros))]
                    h[i] = 0.0
                    h[j] = 1.0
                # recompute
                val = peak_overlap(h)
                if val < best - 1e-12:
                    best = val
                    best_h = h.copy()
                else:
                    h = best_h.copy()
                # break out to a new random restart handled by caller
                break
        return best_h, best

    best_h = None
    best_val = np.inf

    # Try a set of random balanced binary starts with local search
    n_starts = 40
    per_start = (deadline - time.time()) / max(1, n_starts)
    for s in range(n_starts):
        if time.time() > deadline - 5.0:
            break
        h0 = make_balanced_random(s * 7919 + 13)
        h_cand, val = local_search(h0, min(time.time() + per_start, deadline - 5.0))
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()

    if best_h is None:
        # fallback: constant 0.5 (score 1.0)
        return np.full(N, 0.5)

    # Final: ensure exact sum = N/2 and values in [0,1]
    h = best_h.copy()
    h = np.clip(h, 0.0, 1.0)
    # For binary, sum is already half; but guard
    diff = N / 2.0 - h.sum()
    if abs(diff) > 1e-9:
        # nudge: if diff>0 add to zeros, else remove from ones
        if diff > 0:
            zeros = np.where(h < 0.5)[0]
            if len(zeros) > 0:
                h[zeros[0]] += min(diff, 1.0 - h[zeros[0]])
        else:
            ones = np.where(h > 0.5)[0]
            if len(ones) > 0:
                h[ones[0]] += max(diff, -h[ones[0]])
        h = np.clip(h, 0.0, 1.0)

    return h
