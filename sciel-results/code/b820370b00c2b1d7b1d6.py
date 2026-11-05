"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import dct, idct

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _dct_to_h(coeffs):
    """Convert DCT-II coefficients to time-domain signal."""
    # coeffs[0] is DC, coeffs[1:] are AC
    h = idct(coeffs, type=2, norm='ortho')
    return h

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses DCT spectral representation: h = DC + weighted cosine basis.
    Optimizes spectral amplitudes to minimize peak autoconvolution / sum^2.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with all-ones (score 2.0)
    best_h = np.ones(N)
    best_val = 2.0
    
    # DCT-II basis: h[i] = sum_k a_k * cos(pi*(2i+1)*k/(2N))
    # We parameterize: a_0 controls DC (sum), a_1..a_{K-1} are AC modes
    # Sum(h) = a_0 * sqrt(N) for orthonormal DCT (norm='ortho')
    # Actually with norm='ortho', idct gives h[i] = sum_k a_k * basis_k[i]
    # where basis_0[i] = 1/sqrt(N) for all i, so sum(h) = a_0 * sqrt(N)
    
    # Strategy: fix DC to give sum=1 (so a_0 = 1/sqrt(N)), optimize AC modes
    # to minimize max(conv(h,h))
    # 
    # Alternative: use a parametric family. Let's try:
    # h[i] = 1 + eps * cos(2*pi*m*i/N) for some mode m and amplitude eps
    # This keeps sum = N (since cosine has zero mean over full periods)
    # Wait, cos(2*pi*m*i/N) for i=0..N-1 has sum ~ 0 if m>0
    # So sum(h) = N, and we vary the shape
    
    # Let's try a multi-mode DCT approach with gradient-free optimization
    # on the AC coefficients
    
    # First, try simple parametric families
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N) for m=1..N//2
    for m in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        # Try several amplitudes
        for a in np.linspace(-0.9, 0.9, 37):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Family 2: two-mode: h[i] = 1 + a1*cos(2*pi*m1*i/N) + a2*cos(2*pi*m2*i/N)
    # Try a few pairs
    for m1 in [1, 2, 3, 5, 10]:
        if time.time() > deadline:
            break
        for m2 in [2, 3, 5, 10, 20]:
            if m1 == m2:
                continue
            for a1 in np.linspace(-0.5, 0.5, 11):
                if time.time() > deadline:
                    break
                for a2 in np.linspace(-0.5, 0.5, 11):
                    h = (1.0 + a1 * np.cos(2.0 * np.pi * m1 * np.arange(N) / N) 
                          + a2 * np.cos(2.0 * np.pi * m2 * np.arange(N) / N))
                    v = _eval_value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
    
    # Family 3: DCT-based with optimized coefficients using simple gradient descent
    # Represent h in DCT-II basis, optimize coefficients
    # h = idct(a, type=2, norm='ortho')
    # sum(h) = a[0] * sqrt(N) (since basis_0 = ones/sqrt(N))
    # We want sum(h) != 0, so keep a[0] fixed and optimize a[1:K]
    
    if time.time() < deadline - 30:
        # DCT optimization with coordinate descent
        K = 50  # number of DCT modes to optimize
        a = np.zeros(N)
        a[0] = 1.0 / np.sqrt(N)  # gives sum(h) = 1
        
        # Initialize with small random AC components
        rng = np.random.RandomState(42)
        a[1:K] = 0.1 * rng.randn(K - 1)
        
        best_dct_val = _eval_value(_dct_to_h(a))
        if best_dct_val < best_val:
            best_val = best_dct_val
            best_h = _dct_to_h(a).copy()
        
        # Coordinate descent on each DCT coefficient
        step = 0.05
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            improved = False
            for k in range(1, K):
                if time.time() > deadline - 10:
                    break
                old = a[k]
                for delta in [step, -step, 2*step, -2*step, 0.5*step, -0.5*step]:
                    a[k] = old + delta
                    h = _dct_to_h(a)
                    v = _eval_value(h)
                    if v < best_dct_val:
                        best_dct_val = v
                        if v < best_val:
                            best_val = v
                            best_h = h.copy()
                        improved = True
                    else:
                        a[k] = old
                if not improved and iteration > 50:
                    # Try smaller steps
                    step *= 0.5
                    if step < 0.001:
                        break
            if step < 0.001:
                break
    
    # Family 4: Block sign pattern h[i] = +1 for i<k, -1 for i>=k
    for k in range(100, 300, 5):
        if time.time() > deadline:
            break
        if k == N // 2:
            continue  # sum would be 0
        h = np.ones(N)
        h[k:] = -1.0
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Family 5: Smoothed block: h[i] = tanh(beta * (i - N/2))
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        if time.time() > deadline:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Family 6: h[i] = 1 + a*sin(2*pi*m*i/N) (sine instead of cosine)
    for m in [1, 2, 3, 5, 10]:
        if time.time() > deadline:
            break
        for a in np.linspace(-0.9, 0.9, 19):
            h = 1.0 + a * np.sin(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
