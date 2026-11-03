"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    dx = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, N intervals
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + dx/2 # Centers of intervals

    def autoconv_max_ratio(h, beta=50.0):
        """Compute smooth max(autoconv) / ||h||_2^2 for step function h on N equal intervals."""
        h = np.asarray(h, dtype=np.float64)
        # Use FFT for efficient autoconvolution
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        
        # Valid autoconv entries
        autoconv_vals = autoconv_circ[:2*N-1] * dx**2
        
        # Smooth max using log-sum-exp
        # max(y) approx log(sum(exp(beta * y))) / beta
        # Shift by max to prevent overflow
        m = np.max(autoconv_vals)
        if m < 1e-15:
            return 0.0
        smooth_max = (np.log(np.sum(np.exp(beta * (autoconv_vals - m)))) + beta * m) / beta
        
        l2_sq = np.sum(h**2) * dx
        if l2_sq < 1e-15:
            return float('inf')
        
        return smooth_max / l2_sq

    def coordinate_descent_refine(h, max_iters=500, step_decay=0.95, beta=100.0):
        """Refine h using coordinate descent to minimize autoconv_max_ratio."""
        h = np.asarray(h, dtype=np.float64).copy()
        current_score = autoconv_max_ratio(h, beta)
        step_size = 0.1
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try increasing and decreasing h[i]
                for direction in [1, -1]:
                    h_trial = h.copy()
                    h_trial[i] = max(0, h[i] + direction * step_size)
                    trial_score = autoconv_max_ratio(h_trial, beta)
                    if trial_score < current_score:
                        h = h_trial
                        current_score = trial_score
                        improved = True
                        break
                if not improved and direction == -1:
                    step_size *= step_decay
            if not improved:
                break
        
        return h, current_score

    def fourier_to_h(coeffs, K):
        """Convert Fourier coefficients to step function values."""
        # coeffs: [a1, b1, a2, b2, ..., aK, bK]
        h_ac = np.zeros(N)
        for k in range(1, K + 1):
            a_k = coeffs[2*(k-1)]
            b_k = coeffs[2*(k-1)+1]
            h_ac += a_k * np.cos(2 * np.pi * k * x) + b_k * np.sin(2 * np.pi * k * x)
        
        # Ensure non-negative by adding DC offset
        # We want min(h) to be close to 0, so c = -min(h_ac)
        # Add a small epsilon to avoid exactly 0 which might cause numerical issues in division if l2_sq is small
        c = -np.min(h_ac) + 0.1 
        h = c + h_ac
        return h

    def fourier_objective(coeffs, K):
        """Objective function for Fourier optimization."""
        h = fourier_to_h(coeffs, K)
        return autoconv_max_ratio(h, beta=20.0) # Lower beta for smoother landscape during optimization

    best_score = float('inf')
    best_h = None

    # Strategy 1: Fourier Optimization
    K = 10 # Number of harmonics
    n_params = 2 * K
    
    # Multiple random restarts in Fourier space
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        # Initialize with small random values
        x0 = np.random.normal(0, 0.5, n_params)
        
        try:
            res = scipy.optimize.minimize(fourier_objective, x0, args=(K,), 
                                          method='Nelder-Mead', 
                                          options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500})
            
            h_candidate = fourier_to_h(res.x, K)
            # Evaluate with higher beta for more accurate score
            score_candidate = autoconv_max_ratio(h_candidate, beta=100.0)
            
            if score_candidate < best_score:
                best_score = score_candidate
                best_h = h_candidate.copy()
                
        except Exception:
            pass

    # Strategy 2: Spatial Candidates (Gaussian etc) as fallback/alternative
    centers = np.arange(N) - N // 2
    
    candidates = []
    for sigma in [1, 2, 5, 10, 20, 50, 100]:
        h = np.exp(-centers**2 / (2 * sigma**2))
        candidates.append(h)
    
    for width in [10, 50, 100, 200, 300]:
        h = np.maximum(0, 1 - np.abs(centers) / width)
        candidates.append(h)

    for h in candidates:
        if time.time() > deadline - 5:
            break
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(h, 0)
        if np.sum(h) < 1e-10:
            continue
        score = autoconv_max_ratio(h, beta=100.0)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Refine the best candidate with coordinate descent
    if best_h is not None and time.time() < deadline - 5:
        refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=200, step_decay=0.9, beta=100.0)
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
