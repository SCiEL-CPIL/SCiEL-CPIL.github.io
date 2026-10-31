import numpy as np
import itertools

N = 600

def _score(h):
    """Compute the normalized maximum of the autoconvolution (lower is better)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return float('inf')
    h = np.maximum(h, 0.0)
    # Full autoconvolution
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)
    total = np.sum(h)
    if total <= 0:
        return float('inf')
    # Normalized metric: peak / (total^2 / N)  -- this makes constant give 1.0
    # Actually, let's use the standard normalized metric that gives 1.0 for constant
    # For constant h=1: conv peak = N, sum = N, sum^2/N = N. So peak/(sum^2/N) = 1.
    return peak / (total * total / N)

def construct():
    """
    Return a sequence of N nonnegative step heights.
    
    We search for a nonnegative step function that minimizes the normalized maximum
    of its autoconvolution. The constant function gives score 1.0. We try to find
    something better by exploring structured patterns.
    
    Key insight: For nonnegative functions, the autoconvolution peak is at the center.
    The normalized metric is scale-invariant. We need to shape the function to reduce
    the peak relative to the average.
    
    Strategy: Try various structured patterns and pick the best.
    """
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try a few structured patterns
    patterns = []
    
    # Pattern 1: Two equal blocks
    for split in range(N//4, 3*N//4, N//20):
        h = np.zeros(N)
        h[:split] = 1.0
        h[split:] = 1.0
        # This is just constant, skip
        # Try: block on, block off, block on
        h2 = np.zeros(N)
        third = N // 3
        h2[:third] = 1.0
        h2[2*third:] = 1.0
        s = _score(h2)
        if s < best_score:
            best_score = s
            best_h = h2.copy()
    
    # Pattern 2: Cosine-like shape (tapered ends)
    x = np.linspace(0, 2*np.pi, N)
    h_cos = 0.5 + 0.5 * np.cos(x)
    h_cos = np.maximum(h_cos, 0)
    s = _score(h_cos)
    if s < best_score:
        best_score = s
        best_h = h_cos.copy()
    
    # Pattern 3: Triangle peak in center
    h_tri = np.zeros(N)
    center = N // 2
    for i in range(N):
        d = abs(i - center)
        h_tri[i] = max(0, 1 - d / (N/2))
    s = _score(h_tri)
    if s < best_score:
        best_score = s
        best_h = h_tri.copy()
    
    # Pattern 4: Uniform on a subset (sparse)
    for frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
        width = int(N * frac)
        start = (N - width) // 2
        h_sub = np.zeros(N)
        h_sub[start:start+width] = 1.0
        s = _score(h_sub)
        if s < best_score:
            best_score = s
            best_h = h_sub.copy()
    
    # Pattern 5: Two separated blocks
    for gap_frac in [0.1, 0.2, 0.3]:
        gap = int(N * gap_frac)
        block = (N - gap) // 2
        h_2b = np.zeros(N)
        h_2b[:block] = 1.0
        h_2b[N-block:] = 1.0
        s = _score(h_2b)
        if s < best_score:
            best_score = s
            best_h = h_2b.copy()
    
    # Pattern 6: Smooth bump (Gaussian-like)
    x = np.arange(N) - N/2
    sigma = N / 4
    h_gauss = np.exp(-x**2 / (2 * sigma**2))
    s = _score(h_gauss)
    if s < best_score:
        best_score = s
        best_h = h_gauss.copy()
    
    # Pattern 7: Sine squared
    x = np.linspace(0, np.pi, N)
    h_sin = np.sin(x)**2
    s = _score(h_sin)
    if s < best_score:
        best_score = s
        best_h = h_sin.copy()
    
    # Pattern 8: Inverse cosine taper (smooth edges)
    x = np.linspace(0, np.pi, N)
    h_taper = 0.5 * (1 + np.cos(x))
    s = _score(h_taper)
    if s < best_score:
        best_score = s
        best_h = h_taper.copy()
    
    # Pattern 9: Flat top with smooth edges
    for edge_frac in [0.05, 0.1, 0.15, 0.2]:
        edge = int(N * edge_frac)
        h_ft = np.ones(N)
        # Taper edges
        for i in range(edge):
            h_ft[i] *= (i + 1) / edge
            h_ft[N-1-i] *= (i + 1) / edge
        s = _score(h_ft)
        if s < best_score:
            best_score = s
            best_h = h_ft.copy()
    
    # Pattern 10: Power law taper
    for p in [0.5, 1.0, 2.0, 3.0]:
        x = np.linspace(0, 1, N)
        h_pl = np.minimum(x, 1-x) ** p
        h_pl = np.maximum(h_pl, 0)
        # Normalize to have same mean
        if np.mean(h_pl) > 0:
            h_pl = h_pl / np.mean(h_pl)
        s = _score(h_pl)
        if s < best_score:
            best_score = s
            best_h = h_pl.copy()
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
