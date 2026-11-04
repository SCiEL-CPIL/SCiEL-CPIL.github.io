import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0  # safety margin under 150s
    
    # Grid points: centers of the 600 intervals on [-0.25, 0.25]
    x = -0.25 + (np.arange(N) + 0.5) * (0.5 / N)
    
    # Precompute basis functions for Fourier series
    # We use cosines to maintain even symmetry, which is optimal for minimizing peak of autoconvolution
    K = 20  # Number of Fourier modes to optimize
    # Frequencies: 2 * k / L, where L=0.5. So freq = 4k.
    # Argument: 2 * pi * freq * x = 2 * pi * 4k * x = 8 * pi * k * x
    # Let's just use k * 2 * pi * x / (0.5) = 4 * pi * k * x.
    # Wait, standard Fourier series on [-L/2, L/2] uses cos(n * 2 * pi * x / L).
    # Here L = 0.5. So cos(n * 2 * pi * x / 0.5) = cos(4 * pi * n * x).
    
    # Basis matrix B of shape (N, K)
    # B[j, k] = cos(4 * pi * k * x_j) for k=1..K-1, and 1 for k=0?
    # We will optimize theta_1 .. theta_K.
    # Let's include a constant term theta_0 to allow shifting the baseline, though exp() makes it less critical.
    # Actually, since we normalize f, the absolute scale doesn't matter, only the shape.
    # So we can fix theta_0 = 0 and optimize theta_1..theta_K.
    
    basis = np.cos(4.0 * np.pi * np.outer(x, np.arange(1, K)))
    # basis shape: (N, K-1)
    
    def score_from_theta(theta):
        # theta: array of shape (K-1,)
        # g(x) = theta @ basis
        g = basis @ theta
        f = np.exp(g)
        
        # Normalize f to have sum = 1 (proxy for integral = 1)
        s = np.sum(f)
        if s == 0:
            return float('inf')
        f = f / s
        
        # Compute autoconvolution
        # Linear convolution of f with f
        # Use FFT
        M = 1
        while M < (2 * N - 1):
            M *= 2
        F = np.fft.rfft(f, M)
        conv = np.fft.irfft(F * F.conj(), M)[:2 * N - 1]
        
        peak = np.max(conv)
        # The score is peak / (sum(f))^2. Since sum(f)=1, score = peak.
        # Wait, the problem says "normalized maximum of its autoconvolution".
        # Usually this means max(f*f) / (integral f)^2 or similar.
        # If f is the step function, integral is sum(f)*h.
        # Let's check the seed program's score_proxy:
        # peak / (l1(f))^2 where l1(f) = sum(f)*h.
        # So if I normalize sum(f)=1, then l1(f) = h.
        # Score = peak / h^2.
        # Since h is constant, minimizing peak is equivalent.
        # But to be safe, let's compute the exact score as in the seed.
        
        h = 0.5 / N
        l1 = np.sum(f) * h
        return peak / (l1 * l1)
        
    # Initial theta
    np.random.seed(42)
    theta0 = np.zeros(K - 1)
    
    # Use Nelder-Mead for robustness without gradient
    # Or L-BFGS-B if we can compute gradient.
    # Gradient of score wrt theta is complex. Nelder-Mead is safer.
    
    best_score = float('inf')
    best_theta = theta0
    
    # Run optimization
    # Nelder-Mead might be slow if K is large. K=20 is okay.
    # Let's try a few restarts if time permits.
    
    try:
        res = minimize(score_from_theta, theta0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_theta = res.x
    except Exception:
        pass
        
    # Try a second run with different initialization if time allows
    if time.time() < deadline - 30:
        theta0_2 = np.random.normal(0, 0.5, K - 1)
        try:
            res2 = minimize(score_from_theta, theta0_2, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res2.fun < best_score:
                best_score = res2.fun
                best_theta = res2.x
        except Exception:
            pass

    # Construct final heights
    g = basis @ best_theta
    f = np.exp(g)
    f = np.maximum(f, 1e-10) # Ensure non-negative
    s = np.sum(f)
    f = f / s
    
    return [float(x) for x in f]
# EVOLVE-BLOCK-END
