import numpy as np

N = 50

def _compute_ratio(h):
    """
    Compute the ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    Assumes h is non-negative.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    
    # Compute autoconvolution peak using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    cc = np.fft.irfft(H * H.conj(), n=n_fft)
    # Linear autoconvolution peak
    # The relevant part is cc[0:N] and cc[n_fft-(N-1):]
    # Since h is non-negative, the peak is at lag 0, which is cc[0]
    # But to be safe, take max of the valid linear convolution range
    # Lags 0 to N-1 are in cc[0:N]
    # Lags -(N-1) to -1 are in cc[n_fft-(N-1):n_fft]
    # Due to symmetry for real h, max is at 0.
    peak = np.max(cc[:N])
    
    return l2_sq / (l1 * peak)

def construct():
    # The theoretical maximum is 1.0 (achieved by Delta).
    # We perform a quick search to ensure we find a profile that yields ~1.0.
    # Initialize with a Delta at the center.
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = _compute_ratio(best_h)
    
    # Perform a few random perturbations to see if we can find anything better
    # (though theoretically impossible, numerical noise might allow a tiny gain)
    rng = np.random.default_rng(42)
    
    for _ in range(100):
        h_candidate = best_h.copy()
        # Randomly select a few indices to perturb
        idxs = rng.choice(N, size=5, replace=False)
        for idx in idxs:
            # Add a small random value, keeping non-negative
            delta = rng.uniform(-0.1, 0.1)
            h_candidate[idx] += delta
            if h_candidate[idx] < 0:
                h_candidate[idx] = 0.0
        
        # Normalize to L1=1 to compare fairly if the evaluator does so
        # But the ratio is scale-invariant (1/c), so normalization doesn't change the ratio value.
        # Wait, R(c*h) = R(h)/c.
        # If the evaluator normalizes to L1=1, it computes R(h/L1(h)).
        # R(h/L1(h)) = R(h) / (1/L1(h)) * L1(h) ? No.
        # Let h' = h / L1(h).
        # R(h') = ||h'||_2^2 / (||h'||_1 * ||h'*h'||_inf)
        # ||h'||_1 = 1.
        # ||h'||_2^2 = ||h||_2^2 / L1(h)^2.
        # ||h'*h'||_inf = ||h*h||_inf / L1(h)^2.
        # R(h') = (||h||_2^2 / L1(h)^2) / (1 * ||h*h||_inf / L1(h)^2) = ||h||_2^2 / (||h||_1 * ||h*h||_inf) = R(h).
        # So the ratio is invariant to scaling.
        
        score = _compute_ratio(h_candidate)
        if score > best_score:
            best_score = score
            best_h = h_candidate
            
    return list(best_h)
