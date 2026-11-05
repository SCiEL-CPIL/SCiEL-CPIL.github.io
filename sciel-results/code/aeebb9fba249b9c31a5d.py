import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    def score(h):
        s = np.sum(h)
        if abs(s) < 1e-10 or not np.isfinite(s):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2.0 * N * np.max(z) / (s * s))

    rng = np.random.RandomState(42)

    best_h = np.ones(N)
    best_val = score(best_h)

    # Fast scoring using FFT (much faster than np.convolve for large N)
    def fast_score(h):
        s = np.sum(h)
        if abs(s) < 1e-10 or not np.isfinite(s):
            return float('inf')
        # Use FFT for convolution: circular convolve, but we need linear
        # For linear convolution of length N with N, result length 2N-1
        # Pad to next power of 2 >= 2N-1
        size = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, size)
        Z = np.fft.irfft(H * H, size)[:2 * N - 1]
        return abs(2.0 * N * np.max(Z) / (s * s))

    # Verify fast_score matches score on a few random vectors
    for _ in range(3):
        h_test = rng.randn(N)
        if abs(np.sum(h_test)) > 1e-6:
            v1 = score(h_test)
            v2 = fast_score(h_test)
            assert abs(v1 - v2) < 1e-6, f"Score mismatch: {v1} vs {v2}"
            break

    # Phase 1: Broad random search with FFT-based scoring
    n_random = 5000
    for i in range(n_random):
        if time.time() > deadline:
            break
        h = rng.randn(N)
        if abs(np.sum(h)) < 1e-8:
            continue
        val = fast_score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Phase 2: Structured initializations
    t = np.arange(N)

    # Chirps
    for freq in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        if time.time() > deadline:
            break
        for h in [np.cos(2 * np.pi * freq * t * t / (N * N)),
                  np.sin(2 * np.pi * freq * t * t / (N * N))]:
            if abs(np.sum(h)) < 1e-8:
                continue
            val = fast_score(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()

    # Low-pass filtered random
    for i in range(200):
        if time.time() > deadline:
            break
        h = rng.randn(N)
        H = np.fft.rfft(h)
        cutoff = rng.choice([N // 20, N // 10, N // 5, N // 4])
        H[cutoff:] = 0
        h_lp = np.fft.irfft(H, N)
        if abs(np.sum(h_lp)) < 1e-8:
            continue
        val = fast_score(h_lp)
        if val < best_val:
            best_val = val
            best_h = h_lp.copy()

    # Frequency-domain construction: pick few bins
    for i in range(500):
        if time.time() > deadline:
            break
        H = np.zeros(N // 2 + 1, dtype=complex)
        n_bins = rng.randint(2, 8)
        bins = rng.choice(range(1, N // 2), min(n_bins, N // 2 - 1), replace=False)
        for b in bins:
            amp = rng.randn() + 1j * rng.randn()
            H[b] = amp
        H[0] = rng.randn()
        h = np.fft.irfft(H, N)
        if abs(np.sum(h)) < 1e-8:
            continue
        val = fast_score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Phase 3: Local refinement using projected gradient descent
    # Minimize f(h) = max(conv(h,h)) subject to sum(h) = 1 (normalization)
    # Since the objective is scale-invariant, fix sum(h)=1.
    # Use subgradient of max + random perturbation to escape local minima.

    def refine(h0, steps=3000):
        h = h0.astype(float).copy()
        s = np.sum(h)
        if abs(s) < 1e-10:
            return h0, float('inf')
        h = h / s  # normalize to sum=1

        best_local = h.copy()
        best_local_val = fast_score(h)

        # Find which lag achieves the max
        size = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, size)
        Z = np.fft.irfft(H * H, size)[:2 * N - 1]
        k_max = np.argmax(Z)

        # Subgradient of max(Z) w.r.t. h:
        # Z[k] = sum_{i+j=k} h[i]h[j]
        # dZ[k]/dh[m] = h[m + (k-m)] if 0 <= k-m < N, plus symmetric term
        # Actually for the entry at lag k_max:
        # dZ[k_max]/dh[m] = h[m + k_max - m]... 
        # Let's think: Z[k] = sum_i h[i] h[k-i] (with appropriate bounds)
        # dZ[k]/dh[m] = h[k-m] if 0<=k-m<N, plus h[m] if 0<=m<N (from the other factor)
        # Wait, Z[k] = sum_{i=0}^{N-1} h[i] * h[k-i] where h[j]=0 if j<0 or j>=N
        # dZ[k]/dh[m] = h[k-m] (if 0<=k-m<N) + h[m] (if 0<=m<N, from the first factor when i=m... no)
        # Actually: Z[k] = sum_i h[i] h[k-i]
        # dZ[k]/dh[m] = (d/dhm sum_i h[i] h[k-i]) = h[k-m] if 0<=k-m<N (from i=m term: dh[m]/dh[m] * h[k-m])
        # plus if k-m is not in [0,N), the first factor doesn't contribute.
        # But also, when i = k-m, we get h[i]*h[k-i] = h[k-m]*h[m], derivative w.r.t. h[m] is h[k-m].
        # And when the second index equals m: h[i]*h[m] where k-i=m, so i=k-m. Same term.
        # So dZ[k]/dh[m] = h[k-m] if 0 <= k-m < N, else 0.
        # Wait, that's only one term. Let me reconsider.
        # Z[k] = sum_{i=0}^{N-1} h[i] * h[k-i]  (with h out of range = 0)
        # For fixed m, Z[k] has terms where i=m: h[m]*h[k-m], and where k-i=m i.e. i=k-m: h[k-m]*h[m].
        # If m != k-m (i.e., 2m != k), these are two distinct terms, each contributing h[k-m] and h[m] respectively.
        # So dZ[k]/dh[m] = h[k-m] (if 0<=k-m<N) + h[m] (if 0<=m<N and m != k-m... no)
        # Actually both terms involve h[m]: the term i=m gives h[m]*h[k-m], derivative is h[k-m].
        # The term i=k-m gives h[k-m]*h[m], derivative is h[k-m] (derivative of h[m] is 1, times h[k-m]).
        # Wait no: if i=k-m, the term is h[k-m]*h[m]. d/dh[m] of this = h[k-m].
        # And if i=m, the term is h[m]*h[k-m]. d/dh[m] of this = h[k-m].
        # So if m != k-m: dZ[k]/dh[m] = 2*h[k-m] (if both indices valid)
        # If m == k-m (2m=k): only one term h[m]*h[m], derivative = 2*h[m].
        # Hmm, this is getting complicated. Let me just use a simpler approach.

        # Simpler: use random restarts with local search via coordinate descent or small perturbations.
        # Or: since the problem is non-convex, use a simple stochastic local search.

        # Alternative: use the fact that we want flat autocorrelation.
        # Try to minimize max(Z) by adjusting h to reduce the peak.

        # Let's use a simple approach: iteratively reduce the contribution to the peak.
        # The peak is at k_max. We want to reduce Z[k_max].
        # Z[k_max] = sum_i h[i] h[k_max - i]
        # To reduce this, we can try to make h[i] and h[k_max-i] have opposite signs or smaller magnitudes.

        # For now, use a simple local search: perturb randomly and keep if better.
        for step in range(steps):
            if time.time() > deadline:
                break
            # Random perturbation
            perturb = rng.randn(N) * 0.01
            h_new = h + perturb
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-10:
                continue
            h_new = h_new / s_new
            val = fast_score(h_new)
            if val < best_local_val:
                best_local_val = val
                best_local = h_new.copy()
                h = h_new
            else:
                # Occasionally accept worse to escape local minima
                if rng.rand() < 0.1:
                    h = h_new

        return best_local, best_local_val

    # Refine from best random/structured
    h_ref, val_ref = refine(best_h, steps=2000)
    if val_ref < best_val:
        best_val = val_ref
        best_h = h_ref

    # Refine from several other good candidates
    candidates_to_refine = []
    for _ in range(50):
        if time.time() > deadline:
            break
        h = rng.randn(N)
        if abs(np.sum(h)) < 1e-8:
            continue
        val = fast_score(h)
        if val < best_val * 1.5:  # only refine reasonably good candidates
            candidates_to_refine.append(h.copy())

    for h_cand in candidates_to_refine:
        if time.time() > deadline:
            break
        h_ref, val_ref = refine(h_cand, steps=500)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Final verification with exact np.convolve
    final_val = score(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
