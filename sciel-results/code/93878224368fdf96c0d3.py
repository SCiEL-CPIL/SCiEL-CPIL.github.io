import numpy as np
import time

N = 600

def compute_score(h):
    """
    Compute the normalized maximum autoconvolution score.
    Score = max_k S_k / S_0, where S_k = sum_i h_i h_{i+k}
    """
    H = np.sum(h)
    if H < 1e-12:
        return np.inf

    # Compute autoconvolution coefficients S_k using FFT for efficiency
    n_fft = 2 * N
    H_fft = np.fft.rfft(h, n=n_fft)
    S_fft = H_fft * np.conj(H_fft)
    S = np.fft.irfft(S_fft, n=n_fft)[:N]

    S_0 = S[0]
    if S_0 < 1e-12:
        return np.inf

    return np.max(S) / S_0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Initialize with uniform distribution
    h = np.ones(N)
    best_h = h.copy()
    best_score = compute_score(h)

    # We want to minimize max_k S_k / S_0.
    # For non-negative h, S_0 = sum(h^2).
    # For uniform h, all S_k are equal, so max(S)/S_0 = 1.
    # This is the Cauchy-Schwarz bound.
    # We can try to find a non-uniform h that achieves max(S)/S_0 < 1.

    # Strategy: use iterative water-filling / gradient-like method
    # to push down the peaks of the autoconvolution.
    # The autoconvolution S_k = sum_i h_i * h_{i+k}.
    # We want to minimize max_k S_k / S_0.
    # This is equivalent to minimizing max_k S_k subject to S_0 = 1 (by scaling).
    # But scaling doesn't change the ratio, so we can work with any normalization.

    # Alternative: use a projection method.
    # Start with uniform, then iteratively reduce h at positions that contribute
    # most to the largest S_k.

    # Let's try a more sophisticated approach:
    # Use the fact that S_k = <h, shift_k(h)>.
    # We want to minimize max_k <h, shift_k(h)> / <h, h>.
    # This is a max-min type problem.

    # Simple heuristic: iteratively adjust h to equalize the S_k values.
    # If S_k is the maximum, we want to reduce it.
    # Reducing h at position i reduces S_k by approximately h_{i+k} + h_{i-k} (for the relevant k).

    # Let's try a different approach: use scipy's minimize.
    # Define f(h) = max_k S_k / S_0.
    # This is not smooth, but we can use a smooth approximation or subgradient methods.

    # For now, let's try a simple iterative scheme:
    # 1. Compute S_k for current h.
    # 2. Find the k* that maximizes S_k.
    # 3. For each i, compute the "contribution" to S_{k*} from h_i.
    #    S_{k*} = sum_i h_i * h_{i+k*}.
    #    The contribution of h_i to S_{k*} is h_{i+k*} (and also h_{i-k*} from the other term).
    # 4. Reduce h_i proportional to its contribution, then renormalize.

    # Actually, let's try a simpler idea: use a Gaussian-like or windowed distribution.
    # A triangular or raised-cosine window might have lower peak autoconvolution.

    # Let's try several parametric families and pick the best.

    t = np.arange(N) / N

    # Family 1: Raised cosine
    for alpha in np.linspace(0.1, 1.0, 10):
        p = 1.0 + alpha * np.cos(2 * np.pi * t)
        p = np.clip(p, 0, None)
        if np.sum(p) < 1e-12:
            continue
        s = compute_score(p)
        if s < best_score:
            best_score = s
            best_h = p.copy()

    # Family 2: Triangular
    for center in np.linspace(0.3, 0.7, 7):
        for width in np.linspace(0.1, 0.5, 9):
            p = np.maximum(0, 1 - np.abs(t - center) / width)
            if np.sum(p) < 1e-12:
                continue
            s = compute_score(p)
            if s < best_score:
                best_score = s
                best_h = p.copy()

    # Family 3: Bump-like (Gaussian)
    for center in np.linspace(0.3, 0.7, 7):
        for sigma in np.linspace(0.05, 0.3, 9):
            p = np.exp(-0.5 * ((t - center) / sigma) ** 2)
            if np.sum(p) < 1e-12:
                continue
            s = compute_score(p)
            if s < best_score:
                best_score = s
                best_h = p.copy()

    # Family 4: Two bumps
    for c1 in np.linspace(0.2, 0.45, 6):
        for c2 in np.linspace(0.55, 0.8, 6):
            for sigma in np.linspace(0.05, 0.2, 5):
                p = np.exp(-0.5 * ((t - c1) / sigma) ** 2) + np.exp(-0.5 * ((t - c2) / sigma) ** 2)
                if np.sum(p) < 1e-12:
                    continue
                s = compute_score(p)
                if s < best_score:
                    best_score = s
                    best_h = p.copy()

    # Family 5: Cosine window (Blackman-like)
    for a in np.linspace(0.1, 0.5, 5):
        p = 0.5 * (1 + np.cos(2 * np.pi * (t - 0.5) / (1 - a)))
        p = np.clip(p, 0, None)
        s = compute_score(p)
        if s < best_score:
            best_score = s
            best_h = p.copy()

    # Try an iterative refinement on the best so far
    h_work = best_h.copy()
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break

        n_fft = 2 * N
        H_fft = np.fft.rfft(h_work, n=n_fft)
        S_fft = H_fft * np.conj(H_fft)
        S = np.fft.irfft(S_fft, n=n_fft)[:N]
        S_0 = S[0]
        if S_0 < 1e-12:
            break
        ratio = S / S_0
        max_ratio = np.max(ratio)
        k_max = np.argmax(ratio)

        # Compute gradient-like adjustment
        # dS_k/dh_i = h_{i+k} + h_{i-k} (with wrap-around for circular, but we have linear)
        # For linear autoconvolution: S_k = sum_{i=0}^{N-k-1} h_i * h_{i+k}
        # dS_k/dh_i = h_{i+k} if i < N-k, plus h_{i-k} if i >= k
        contrib = np.zeros(N)
        for i in range(N):
            if i + k_max < N:
                contrib[i] += h_work[i + k_max]
            if i - k_max >= 0:
                contrib[i] += h_work[i - k_max]

        # Reduce h where contribution to max S_k is high
        step = 0.05
        h_work -= step * contrib / (np.max(contrib) + 1e-12)
        h_work = np.clip(h_work, 0, None)
        if np.sum(h_work) < 1e-12:
            break
        s = compute_score(h_work)
        if s < best_score:
            best_score = s
            best_h = h_work.copy()
        else:
            # Revert if it got worse
            h_work = best_h.copy()
            break

    return list(best_h)

construct()
