"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np

N = 400


def construct():
    """
    Construct step heights h of length N=400 to minimize:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    Strategy: Use a flat-spectrum (constant magnitude DFT) sequence to minimize
    the peak of the autoconvolution. We construct h such that |DFT(h)| is as
    flat as possible, which minimizes the maximum entry of convolve(h,h).
    
    We use a chirp sequence or a randomly-phased flat-spectrum sequence.
    """
    # Construct h with flat magnitude spectrum
    # h[i] = exp(2*pi*i*phi_i) where phi_i are chosen to give flat spectrum
    # Simple approach: use a quadratic chirp
    # h[i] = exp(2*pi*i*(i/N)) = exp(2*pi*i^2/N)
    
    # A Zadoff-Chu-like sequence or simply a chirp
    i = np.arange(N)
    
    # Use a chirp: h[i] = exp(2*pi*i*i/N)
    # This gives a flat magnitude spectrum
    h = np.exp(2j * np.pi * i * i / N)
    
    # But we need real h. Use cos or take real part
    # Actually, let's use a real sequence with flat spectrum
    # h[i] = cos(2*pi*i*i/N) + small imaginary part won't work for real
    
    # Better: use a real-valued sequence
    # h[i] = cos(2*pi*i*i/N) - this is real but not flat spectrum exactly
    # Let's try: h[i] = (-1)^i * exp(-alpha*i^2/N^2) or similar
    
    # Actually, let's try a different approach:
    # Use a random real sequence with fixed sum, then optimize
    
    # Start with a chirp-based real sequence
    h = np.cos(2 * np.pi * i * i / N)
    
    # Normalize to sum = 1
    s = np.sum(h)
    if abs(s) < 1e-10:
        # Add a small constant to make sum nonzero
        h = h + 1e-6
        s = np.sum(h)
    h = h / s
    
    # Compute the value
    ac = np.convolve(h, h)
    val = abs(2 * N * np.max(ac) / s**2)
    
    # Try also a simpler approach: alternating signs with decay
    # h[i] = (-1)^i * exp(-beta * |i - N/2| / N)
    best_h = h
    best_val = val
    
    # Try several chirp parameters
    for beta in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        center = N / 2.0
        h_try = (-1)**i * np.exp(-beta * np.abs(i - center) / N)
        s_try = np.sum(h_try)
        if abs(s_try) < 1e-10:
            h_try = h_try + 1e-8
            s_try = np.sum(h_try)
        h_try = h_try / s_try
        ac_try = np.convolve(h_try, h_try)
        val_try = abs(2 * N * np.max(ac_try) / s_try**2)
        if val_try < best_val:
            best_val = val_try
            best_h = h_try.copy()
    
    # Try flat spectrum with random phases (deterministic seed)
    np.random.seed(42)
    for trial in range(20):
        # Random flat-magnitude spectrum
        H = np.ones(N) * np.exp(2j * np.pi * np.random.rand(N))
        h_try = np.real(np.fft.ifft(H))
        s_try = np.sum(h_try)
        if abs(s_try) < 1e-10:
            h_try = h_try + 1e-8
            s_try = np.sum(h_try)
        h_try = h_try / s_try
        ac_try = np.convolve(h_try, h_try)
        val_try = abs(2 * N * np.max(ac_try) / s_try**2)
        if val_try < best_val:
            best_val = val_try
            best_h = h_try.copy()
    
    # Try the seed as fallback
    h_seed = np.ones(N)
    ac_seed = np.convolve(h_seed, h_seed)
    val_seed = abs(2 * N * np.max(ac_seed) / np.sum(h_seed)**2)
    if val_seed < best_val:
        return h_seed
    
    return best_h
