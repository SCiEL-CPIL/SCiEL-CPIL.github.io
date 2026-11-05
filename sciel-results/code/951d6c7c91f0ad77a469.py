import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Precompute Chebyshev basis on the grid
    # Map indices 0..N-1 to [-1, 1]
    x = np.linspace(-1, 1, N)
    
    # We use Chebyshev polynomials of the first kind T_k(x)
    # We need to ensure the sum is not zero. 
    # T_0(x) = 1, so the DC component is controlled by c_0.
    # We will fix c_0 = 1 to normalize the mean roughly, and optimize c_1...c_M.
    # Actually, scaling invariance means we can fix one coefficient.
    # Let's fix c_0 = 1.
    
    M = 8  # Number of Chebyshev terms to optimize (c_1 to c_M)
    
    # Precompute basis vectors
    # T_k(x) can be computed recursively or via roots.
    # For efficiency, we precompute the matrix B where B[i, k] = T_k(x_i)
    B = np.zeros((N, M + 1))
    B[:, 0] = 1.0
    if M >= 1:
        B[:, 1] = x
        for k in range(2, M + 1):
            B[:, k] = 2 * x * B[:, k-1] - B[:, k-2]
    
    def score_from_coeffs(coeffs):
        # coeffs are c_1...c_M, c_0 is fixed at 1
        c = np.zeros(M + 1)
        c[0] = 1.0
        c[1:] = coeffs
        h = B.dot(c)
        
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        
        # Compute autocorrelation
        # Using FFT for speed
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n_fft)
        Z = np.fft.irfft(H * np.conj(H), n_fft)[:2*N-1]
        
        # The value is abs(2 * N * max(Z) / s^2)
        # Note: Z from irfft of |H|^2 is the circular autocorrelation?
        # np.convolve(h, h) is linear convolution.
        # For large N, circular approx linear if we zero-pad enough.
        # n_fft >= 2*N - 1 ensures no aliasing for linear convolution.
        
        max_z = np.max(Z)
        return abs(2 * N * max_z / (s * s))

    def objective(coeffs):
        return score_from_coeffs(coeffs)

    best_coeffs = None
    best_score = float('inf')
    
    # Stage 1: Coarse Grid Search
    # We search over a small range of coefficients.
    # Since h should be mostly positive to keep sum(h) large relative to peak,
    # and smooth, the coefficients should be small.
    
    # Let's try a grid over the first few coefficients.
    # c_1 controls the slope. c_2 controls the curvature.
    # We can restrict the search to a smaller subset of coefficients to make grid search feasible.
    # Let's use M_search = 4 for the grid, and optimize all M later.
    
    M_search = 4
    # Range for coefficients: [-0.5, 0.5] seems reasonable to keep h positive-ish.
    # If c_k are too large, h might become negative or oscillatory.
    
    # To make it faster, we'll do a random sampling + local refinement approach.
    # Or a small grid.
    # Let's try 20 random initializations + local optimization.
    
    rng = np.random.default_rng(42)
    
    num_initials = 20
    
    for i_init in range(num_initials):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        # Bias towards small coefficients
        coeffs_init = rng.uniform(-0.5, 0.5, M)
        
        # Local optimization using L-BFGS-B
        try:
            res = minimize(
                objective, 
                coeffs_init, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            continue
            
    # If we found a good solution, try to refine with more iterations or different methods
    if best_coeffs is not None:
        # One more refinement pass
        try:
            res2 = minimize(
                objective, 
                best_coeffs, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res2.success and res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x
        except Exception:
            pass

    # Construct final h
    if best_coeffs is None:
        # Fallback to seed
        return np.ones(N)
    
    c = np.zeros(M + 1)
    c[0] = 1.0
    c[1:] = best_coeffs
    h = B.dot(c)
    
    # Sanity check
    if not np.all(np.isfinite(h)) or np.sum(h) == 0:
        return np.ones(N)
        
    return h
