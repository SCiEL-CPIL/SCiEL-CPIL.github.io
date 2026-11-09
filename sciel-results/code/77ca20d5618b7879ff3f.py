"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START

def _score_vec(h):
    """Exact normalized max autoconvolution for a nonnegative height vector.
    
    The autoconvolution of a step function h on N equal intervals has peak
    at lag 0 equal to sum(h^2) * (interval width). The normalization divides
    by (sum(h))^2 / N to make the score scale-invariant.
    """
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s <= 0:
        return 0.0
    # The peak of autoconvolution is sum(h_i^2) * dx where dx = 1/(2N)
    # Normalized by (sum(h) * dx)^2 / dx = sum(h)^2 * dx
    # So normalized peak = sum(h^2) * dx / (sum(h)^2 * dx) = sum(h^2) / sum(h)^2
    # But the task says "normalized maximum" - let's use sum(h^2)/(sum(h)^2/N)
    # which equals N * sum(h^2) / sum(h)^2
    return float((h * h).sum()) / (s * s / N)

def _softplus(x, beta=1.0):
    """Numerically stable softplus."""
    return np.where(x > 20, x, np.log1p(np.exp(x)) / beta)

def _spectral_to_time(a):
    """Convert spectral amplitudes to time-domain heights.
    
    a: array of length N//2 + 1 (amplitudes for frequencies 0 to N//2)
    Returns: non-negative time-domain vector of length N.
    """
    n = N
    k = n // 2
    # Reconstruct full spectrum. 
    # We assume a[0] is DC, a[1:k] are positive freqs, a[k] is Nyquist (if n even).
    # We need to construct H such that ifft(|H|^2) is non-negative.
    # Actually, the strategy says: parameterize h as inverse FFT of non-negative PSD.
    # Let P = |H|^2. Then h = ifft(P).
    # We parameterize H directly. Let H be complex.
    # To ensure h is real, H must be conjugate symmetric.
    # To ensure h >= 0, we rely on optimization, but we can also use softplus on h.
    # Let's define h = softplus(ifft(|H|^2)).
    # This is a valid non-negative step function.
    
    # Construct H from amplitudes a.
    # a[0] is real positive.
    # a[1:k] are magnitudes. We need phases.
    # If we set phases to 0, H is real and symmetric? No, H[k] is Nyquist.
    # Let's just use real H for simplicity? If H is real, H[-k] = H[k].
    # Then P = H^2 is real and symmetric. h = ifft(P) is real and symmetric.
    # But h might be negative.
    
    # Let's try: H is real, symmetric.
    # H[0] = a[0]
    # H[1:k] = a[1:k]
    # H[k] = a[k] (if n even)
    # H[n-1:k+1] = H[1:k] reversed?
    
    # Standard FFT indexing:
    # H[0]
    # H[1], ..., H[k]
    # H[k+1], ..., H[n-1]
    
    # If H is real, H[n-j] = H[j].
    # So we define H[0] = a[0]
    # H[1:k] = a[1:k]
    # H[k] = a[k] (if n even, H[k] is its own pair)
    # H[k+1:n] = H[n-1-k:n-1] ... wait.
    # H[n-1] = H[1]
    # H[n-2] = H[2]
    
    H = np.zeros(n, dtype=float)
    H[0] = a[0]
    if n % 2 == 0:
        H[1:k] = a[1:k]
        H[k] = a[k]
        H[k+1:n] = H[n-1:k:-1] # H[n-1] = H[1], ..., H[k+1] = H[k-1]
    else:
        H[1:k+1] = a[1:k+1]
        H[k+1:n] = H[n-1:k:-1]
        
    P = H * H
    h = np.fft.ifft(P).real
    # Enforce non-negativity
    h = _softplus(h)
    return h

def _score_spectral(a):
    """Score and gradient for spectral amplitudes."""
    h = _spectral_to_time(a)
    s = h.sum()
    if s <= 0:
        return 0.0, np.zeros_like(a)
    
    # Objective: N * sum(h^2) / s^2
    # Let J = N * (h^2).sum() / s^2
    # dJ/da = N * [ (2 * h * dh/da * s^2 - (h^2).sum() * 2 * s * ds/da) ] / s^4
    #       = 2 * N * [ h * dh/da * s - (h^2).sum() * ds/da ] / s^3
    
    # dh/da:
    # h = softplus(ifft(P))
    # P = H^2
    # H depends on a linearly.
    
    # This gradient calculation is messy. Let's use numerical gradient or just L-BFGS-B on h?
    # No, the point is spectral optimization.
    # Let's use a simpler objective: minimize sum(h^2) subject to sum(h)=1.
    # In spectral domain, this is not convex.
    
    # Alternative: Use scipy.optimize.minimize on the spectral amplitudes a.
    # We can compute the score easily.
    # For gradient, we can use finite differences if needed, but L-BFGS-B can do that.
    
    return _score_vec(h), None

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin under the 150s kill
    
    # Initialize spectral amplitudes
    # Start with a "low-pass" filter shape to encourage concentration.
    # a[0] large, a[k] small.
    
    k = N // 2
    a0 = np.ones(k + 1)
    # Decay amplitudes with frequency
    freqs = np.arange(k + 1)
    a0 = a0 * np.exp(-0.5 * (freqs / 10.0)**2)
    
    best_h, best_val = None, float("inf")
    
    # Try a few different initial spectral shapes
    inits = []
    # 1. Gaussian decay in freq
    inits.append(a0.copy())
    
    # 2. Uniform (white noise in freq -> flat in time? No, white noise in freq is delta in time)
    # Uniform a -> H is constant -> P is constant -> h is delta at 0.
    # This is bad for our objective (maximizes peak).
    # We want h to be smooth. Smooth h means P decays fast.
    # So a should decay fast.
    
    # 3. Very fast decay (almost DC)
    a3 = np.zeros(k + 1)
    a3[0] = 1.0
    inits.append(a3)
    
    # 4. Intermediate decay
    a4 = np.exp(-0.5 * (freqs / 5.0)**2)
    inits.append(a4)
    
    for a_init in inits:
        if time.time() > deadline:
            break
            
        # Optimize a to minimize _score_vec(_spectral_to_time(a))
        # We will use L-BFGS-B.
        # Since we don't have an analytic gradient, we rely on numerical gradient.
        # To speed up, we can provide a gradient if we derive it, but let's try numerical first.
        
        def obj(a):
            return _score_vec(_spectral_to_time(a))
            
        try:
            res = minimize(obj, a_init, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            a_opt = res.x
            h_opt = _spectral_to_time(a_opt)
            val = _score_vec(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
            
    # Fallback: if spectral optimization failed or didn't find anything better than uniform
    if best_h is None or best_val > 1.0:
        # Try a simple time-domain heuristic: smooth bump
        x = np.linspace(-1, 1, N)
        h_bump = np.exp(-5 * x**2)
        best_h = h_bump
        best_val = _score_vec(h_bump)
        
    # Final refinement: Local search in time domain on the best spectral result
    if best_h is not None and time.time() < deadline:
        h_curr = best_h.copy()
        s = h_curr.sum()
        if s > 0:
            h_curr /= s
            
        # Coordinate descent / random perturbation
        rng = np.random.default_rng(42)
        for _ in range(50):
            if time.time() > deadline:
                break
            # Perturb a few random coordinates
            idx = rng.choice(N, size=10, replace=False)
            h_new = h_curr.copy()
            for i in idx:
                h_new[i] = max(0, h_new[i] + rng.standard_normal() * 0.05)
            sn = h_new.sum()
            if sn > 0:
                h_new /= sn
            sc = _score_vec(h_new)
            if sc < best_val - 1e-15:
                best_val = sc
                h_curr = h_new
    
    out = np.asarray(best_h, dtype=float)
    out = np.maximum(out, 0.0)
    s = out.sum()
    if s <= 0:
        return [1.0] * N
    out = out / s
    return [float(x) for x in out]

# EVOLVE-BLOCK-END
