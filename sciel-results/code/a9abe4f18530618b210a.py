N = 400


def construct():
    import math
    import numpy as np

    n = N
    m = n // 2
    h = np.empty(n, dtype=np.float64)

    # Try a few structured sign-changing patterns and pick the one that
    # gives the smallest objective value.  All candidates are scaled so
    # that sum(h) is non-zero; the objective is scale-invariant.

    def score(v):
        s = v.sum()
        if s == 0:
            return float("inf")
        v = v / s
        # full autoconvolution
        c = np.convolve(v, v)
        mx = np.max(c)
        return abs(2.0 * n * mx)

    # Candidate 1: two halves, positive first, negative second
    h1 = np.zeros(n)
    h1[:m] = 1.0
    h1[m:] = -1.0

    # Candidate 2: four quarters alternating
    q = n // 4
    h2 = np.zeros(n)
    h2[0:q] = 1.0
    h2[q:2*q] = -1.0
    h2[2*q:3*q] = 1.0
    h2[3*q:4*q] = -1.0

    # Candidate 3: cosine with one full period, mean-zero
    idx = np.arange(n)
    h3 = np.cos(2.0 * np.pi * idx / n)
    h3 -= h3.mean()

    # Candidate 4: cosine with two periods
    h4 = np.cos(4.0 * np.pi * idx / n)
    h4 -= h4.mean()

    # Candidate 5: sine with one period
    h5 = np.sin(2.0 * np.pi * idx / n)

    # Candidate 6: triangular wave, two periods, mean-zero
    h6 = np.abs(((2.0 * idx / n) % 1.0) - 0.5)
    h6 = h6 - h6.mean()

    # Candidate 7: three-lobe: positive, negative, positive with widths 1/3
    a = n // 3
    h7 = np.zeros(n)
    h7[0:a] = 1.0
    h7[a:2*a] = -1.0
    h7[2*a:3*a] = 1.0
    # remaining (n - 3*a) entries stay zero; adjust by setting them to 0

    # Candidate 8: optimized single-knot step: h[0:k]=a, h[k:]=b with a*0+b*1=1
    # i.e. k*a + (n-k)*b = 1.  Try several k values.
    best_h = h1
    best_val = score(h1)

    for cand in (h2, h3, h4, h5, h6, h7):
        val = score(cand)
        if val < best_val:
            best_val = val
            best_h = cand.copy()

    # Search over single-knot patterns more finely
    for k in range(1, n):
        h_k = np.empty(n)
        h_k[:k] = 1.0
        h_k[k:] = -1.0
        val = score(h_k)
        if val < best_val:
            best_val = val
            best_h = h_k

    # Local perturbation around the best candidate
    # Use a simple deterministic refinement: small random-ish shifts
    # We do a coordinate-wise "nudge" in both directions and keep improvements.
    h_best = best_h.copy()
    improved = True
    while improved and improved:
        improved = False
        cur_val = score(h_best)
        for i in range(n):
            old = h_best[i]
            # try small nudge in + and - direction
            for delta in (0.05 * max(1.0, abs(old)), -0.05 * max(1.0, abs(old))):
                h_try = h_best.copy()
                h_try[i] += delta
                val_try = score(h_try)
                if val_try < cur_val - 1e-12:
                    h_best = h_try
                    cur_val = val_try
                    improved = True
                    break
            else:
                continue
            break
    # second pass with smaller step
    step = 0.01
    improved2 = True
    cur_val = score(h_best)
    while improved2:
        improved2 = False
        for i in range(n):
            old = h_best[i]
            for delta in (step * max(1.0, abs(old)), -step * max(1.0, abs(old))):
                h_try = h_best.copy()
                h_try[i] += delta
                val_try = score(h_try)
                if val_try < cur_val - 1e-13:
                    h_best = h_try
                    cur_val = val_try
                    improved2 = True
                    break
            else:
                continue
            break

    return h_best
