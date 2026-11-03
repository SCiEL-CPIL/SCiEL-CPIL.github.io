"""
Optimized program using global stochastic search (Differential Evolution)
to minimize the uncertainty bound for the Hermite-Gaussian test function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    """
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330
    
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    a0 = c0 + 12*c4 + 168*c8 + 1330*c12
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Convolution to get coefficients of P(x)^2
    q = np.convolve(p, p)
    
    # Integrate x^{2k} e^{-x^2}
    factors = np.zeros(13)
    for k in range(13):
        factors[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I = q * factors
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10 # Invalid
        
    x2 = I[1] / I[0]
    
    # Calculate R(x) for Fourier side
    r = np.zeros(8)
    for j in range(8):
        val = 0.0
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        if j >= 1 and j <= 6:
            val += -4*j*p[j]
        if j >= 1 and j <= 7:
            val += p[j-1]
        if j <= 6:
            val += -p[j]
        r[j] = val
        
    s = np.convolve(r, r)
    
    factors_R = np.zeros(15)
    for k in range(15):
        factors_R[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I_R = s * factors_R
    norm_R_sq = I_R[0]
    
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    
    product = math.sqrt(x2 * xi2)
    
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # 180s limit, use 175s for safety
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    # Define bounds for the search. 
    # The seed values are roughly [-1.13, 0.04, 0.0003].
    # We allow a reasonable range to explore.
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Wrapper for the objective function to handle exceptions
    def objective(x):
        try:
            return _compute_bound(x[0], x[1], x[2])
        except Exception:
            return 1e10

    # Use Differential Evolution for global optimization
    # popsize=15, maxiter=100 is a good balance for 3D problems
    # tol=1e-10 ensures high precision
    try:
        res = scipy.optimize.differential_evolution(
            objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-12,
            polish=True,
            init='sobol',
            popsize=15
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except Exception:
        pass
        
    # If time permits, try a few random restarts with different seeds or bounds
    if time.time() < deadline - 10.0:
        # Try a second run with different parameters to ensure robustness
        try:
            res2 = scipy.optimize.differential_evolution(
                objective, 
                bounds=bounds, 
                seed=123, 
                maxiter=100, 
                tol=1e-12,
                polish=True,
                init='latinhypercube',
                popsize=20
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_coeffs = res2.x
        except Exception:
            pass
            
    return best_coeffs
# EVOLVE-BLOCK-END
