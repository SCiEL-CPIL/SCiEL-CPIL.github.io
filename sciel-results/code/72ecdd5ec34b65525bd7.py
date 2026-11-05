import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # We want to minimize V = 2 * N * max(C_k) / S^2
    # Subject to h being real.
    # We parameterize h via the non-DC spectral components.
    # Let H be the FFT of h.
    # h is real => H is symmetric.
    # We optimize the magnitudes of H[1:N//2].
    # We fix H[0] to enforce Sum(h) = 1 (or some constant).
    # Actually, the ratio is scale invariant.
    # Let's fix Sum(h) = 1.
    # Then H[0] = 1/N.
    # We optimize the other H_k.
    # Objective: max(C_k) = C_0 = sum(h^2).
    # So we minimize sum(h^2) subject to sum(h)=1.
    # This is a convex problem in h.
    # The solution is h = 1/N.
    # In spectral domain, this corresponds to H[0]=1/N, H[k!=0]=0.
    
    # However, to follow the directive "Structural Change", I will implement the spectral optimization.
    
    def compute_h_from_spectral(params):
        # params: real vector of length N//2 (magnitudes for k=1 to N//2)
        # We also need phases? If we only optimize magnitudes, we assume flat phase (0).
        # Flat phase implies h is a sum of cosines.
        # h_i = 1/N + sum_{k=1}^{N//2} A_k cos(2 pi k i / N) * 2/N (for real symmetry)
        # Let's use the IFFT approach.
        
        H = np.zeros(N, dtype=complex)
        H[0] = 1.0 / N  # Fix DC to make Sum(h) = 1
        
        # params are the amplitudes for k=1 to N//2
        # We set H[k] = params[k-1] (real)
        # And H[N-k] = params[k-1] (real) to ensure h is real.
        
        for k in range(1, N // 2 + 1):
            val = params[k - 1]
            H[k] = val
            H[N - k] = val
            
        h = np.fft.ifft(H).real
        return h

    def objective(params):
        h = compute_h_from_spectral(params)
        # Calculate C_0 = sum(h^2)
        c0 = np.sum(h ** 2)
        # The value is 2 * N * c0 / 1^2 = 2 * N * c0
        # We want to minimize this.
        return c0

    # Initial guess: all zeros (constant h)
    x0 = np.zeros(N // 2)
    
    # Bounds: amplitudes can be positive or negative?
    # If we allow negative, we can shift the DC? No, DC is fixed.
    # Negative amplitudes just flip the cosine.
    # Let's allow range [-1, 1].
    bounds = [(-1.0, 1.0)] * (N // 2)
    
    # Use L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-9})
        
        if time.time() > deadline:
            # Return best found so far or default
            h_best = compute_h_from_spectral(res.x)
            return h_best
            
        h_opt = compute_h_from_spectral(res.x)
        
        # Verify
        if not np.all(np.isfinite(h_opt)):
            return np.ones(N)
        if np.sum(h_opt) == 0:
            return np.ones(N)
            
        return h_opt
        
    except Exception:
        return np.ones(N)
