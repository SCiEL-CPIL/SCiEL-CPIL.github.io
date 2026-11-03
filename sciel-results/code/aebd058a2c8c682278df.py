import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """Compute overlap of h with shifted (1-h) for all circular shifts."""
    h = np.asarray(h)
    H = np.fft.fft(h, n=2*N)
    autocorr = np.fft.ifft(H * np.conj(H)).real[:N]
    overlaps = N / 2.0 - autocorr
    return overlaps

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2."""
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current = np.sum(h)
    if abs(current - target) > 1e-10:
        h = h * (target / current)
        h = np.clip(h, 0, 1)
        diff = target - np.sum(h)
        if diff > 0:
            mask = h < 1.0
            if np.sum(mask) > 0:
                per = diff / np.sum(mask)
                h[mask] = np.minimum(h[mask] + per, 1.0)
        else:
            mask = h > 0.0
            if np.sum(mask) > 0:
                per = -diff / np.sum(mask)
                h[mask] = np.maximum(h[mask] - per, 0.0)
    return h

def _spectral_to_spatial(H):
    """Convert spectral coefficients to spatial domain h."""
    # H is a vector of length N containing complex FFT coefficients
    h = np.fft.ifft(H).real
    return h

def _spatial_to_spectral(h):
    """Convert spatial domain h to spectral coefficients."""
    H = np.fft.fft(h)
    return H

def _objective_spectral(x):
    """
    Objective function in spectral domain.
    x is a real vector of length 2*(N//2) + 1 + (1 if N%2==0 else 0) ?
    Simpler: x represents the real and imaginary parts of H[1..N//2-1], 
    H[0] is fixed to N/2 (real), H[N/2] is real if N even.
    
    For N=100:
    H[0] is real, fixed to N/2 = 50.
    H[50] is real.
    H[1..49] are complex.
    
    Let's parameterize x as:
    x[0] : H[50] (real)
    x[1..99] : Real and Imag parts of H[1..49]
    
    Total length: 1 + 2*49 = 99.
    """
    H = np.zeros(N, dtype=complex)
    H[0] = N / 2.0
    
    if N % 2 == 0:
        H[N//2] = x[0]
        # H[1..N//2-1]
        for k in range(1, N//2):
            H[k] = x[2*k-1] + 1j * x[2*k]
            H[N-k] = np.conj(H[k])
    else:
        # Handle odd N if necessary, but N=100 is even
        pass
        
    h = np.fft.ifft(H).real
    
    # Penalty for constraints
    # 1. h in [0, 1]
    penalty_bound = 100.0 * (np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h-1)**2))
    
    # 2. Sum constraint is automatically satisfied by H[0] = N/2?
    # sum(h) = H[0] = N/2. Yes.
    
    # Objective: Maximize overlap
    overlaps = N/2.0 - np.fft.ifft(H * np.conj(H)).real[:N]
    max_ov = np.max(overlaps)
    
    return max_ov + penalty_bound

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Initial guess: Flat spectrum (white noise) but scaled to have low variance?
    # H[0] = 50.
    # If we set H[k] = 0 for k>0, h is constant 0.5.
    # Overlap = 50 - 25 = 25.
    # We want to reduce this.
    
    # Let's try to optimize the spectral coefficients.
    # Dimension of x: 1 (for H[50]) + 2*49 = 99.
    dim = 1 + 2 * (N // 2 - 1)
    
    # Initialize x with small random values to break symmetry
    rng = np.random.default_rng(42)
    x0 = rng.normal(0, 0.1, size=dim)
    
    best_h = _project_to_feasible(np.fft.ifft(np.fft.fft(np.full(N, 0.5))).real)
    best_score = _max_overlap(best_h)
    
    # Use L-BFGS-B on the spectral parameters
    # This is a different parameterization (spectral) and potentially different landscape.
    
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            x_start = x0
        else:
            # Random restart in spectral domain
            x_start = rng.normal(0, 1.0, size=dim)
        
        result = minimize(
            _objective_spectral,
            x_start,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        # Convert back to spatial
        H_opt = np.zeros(N, dtype=complex)
        H_opt[0] = N / 2.0
        if N % 2 == 0:
            H_opt[N//2] = result.x[0]
            for k in range(1, N//2):
                H_opt[k] = result.x[2*k-1] + 1j * result.x[2*k]
                H_opt[N-k] = np.conj(H_opt[k])
        
        h_new = np.fft.ifft(H_opt).real
        h_new = _project_to_feasible(h_new)
        score_new = _max_overlap(h_new)
        
        if score_new < best_score:
            best_h = h_new
            best_score = score_new
    
    # Final polish in spatial domain with the best spectral solution
    if time.time() < deadline - 15.0:
        def obj_spatial(x):
            h = np.clip(x, 0, 1)
            overlaps = _compute_overlaps(h)
            max_ov = np.max(overlaps)
            sum_penalty = 10.0 * (np.sum(h) - N / 2.0) ** 2
            return max_ov + sum_penalty
            
        result = minimize(
            obj_spatial,
            best_h,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        h_polished = _project_to_feasible(result.x)
        score_polished = _max_overlap(h_polished)
        if score_polished < best_score:
            best_h = h_polished
            best_score = score_polished
            
    return best_h
