"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.special import hermite
import math

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin

    # Indices for Hermite polynomials
    indices = [0, 4, 8, 12]
    n_funcs = len(indices)
    
    # Precompute Hermite function values or just use the functions
    # h_n(x) = (2^n n! sqrt(pi))^-0.5 * exp(-x^2/2) * H_n(x)
    
    # We need integrals K[i,j,k,l] = int h_i h_j h_k h_l dx
    # This is symmetric.
    
    # To save time, we can compute these once.
    # The integrand is exp(-2x^2) * P(x) where P is product of H's.
    # We can use quad.
    
    K = np.zeros((n_funcs, n_funcs, n_funcs, n_funcs))
    
    # Helper to get normalized Hermite function
    def herm_func(n, x):
        # Normalization constant
        norm = 1.0 / math.sqrt(2.0**n * math.factorial(n) * math.sqrt(math.pi))
        H_n = hermite(n)(x)
        return norm * np.exp(-x**2 / 2.0) * H_n

    # Compute tensor
    # This loop might be slow if quad is called 256 times.
    # 256 * ~1ms = 0.25s. Acceptable.
    for i in range(n_funcs):
        for j in range(i, n_funcs): # Symmetry
            for k in range(j, n_funcs):
                for l in range(k, n_funcs):
                    # Define integrand
                    # Note: quad expects a function of x
                    ni, nj, nk, nl = indices[i], indices[j], indices[k], indices[l]
                    
                    def integrand(x):
                        return herm_func(ni, x) * herm_func(nj, x) * herm_func(nk, x) * herm_func(nl, x)
                    
                    val, err = integrate.quad(integrand, -np.inf, np.inf, limit=100)
                    
                    # Fill symmetric parts
                    K[i, j, k, l] = val
                    K[i, k, j, l] = val
                    K[i, k, l, j] = val
                    K[i, l, j, k] = val
                    K[i, l, k, j] = val
                    K[i, j, l, k] = val
                    
                    K[j, i, k, l] = val
                    K[j, k, i, l] = val
                    K[j, k, l, i] = val
                    K[j, l, i, k] = val
                    K[j, l, k, i] = val
                    K[j, i, l, k] = val
                    
                    K[k, i, j, l] = val
                    K[k, j, i, l] = val
                    K[k, j, l, i] = val
                    K[k, l, i, j] = val
                    K[k, l, j, i] = val
                    K[k, i, l, j] = val
                    
                    K[l, i, j, k] = val
                    K[l, j, i, k] = val
                    K[l, j, k, i] = val
                    K[l, k, i, j] = val
                    K[l, k, j, i] = val
                    K[l, i, k, j] = val

    # Check time
    if time.time() > deadline - 20.0:
        # Fallback to seed
        return np.array([-113 / 100, 1 / 25, 1 / 3240])

    # Define objective function
    # c = [c0, c4, c8]
    # c12 is derived from g(0)=0
    # h_n(0) values
    h0_0 = herm_func(0, 0.0)
    h4_0 = herm_func(4, 0.0)
    h8_0 = herm_func(8, 0.0)
    h12_0 = herm_func(12, 0.0)
    
    def objective(c):
        c0, c4, c8 = c
        # c12 = -(c0*h0_0 + c4*h4_0 + c8*h8_0) / h12_0
        c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
        c_full = np.array([c0, c4, c8, c12])
        
        # Compute I(c) = sum K[i,j,k,l] c[i] c[j] c[k] c[l]
        # This is equivalent to (c_tensor) @ K (c_tensor) where c_tensor = c ⊗ c ⊗ c ⊗ c
        # More efficiently:
        # I = np.einsum('ijkl, i, j, k, l', K, c_full, c_full, c_full, c_full)
        
        # For Fourier transform, coefficients are d_n = c_n * (-i)^n
        # d0 = c0
        # d4 = c4 * 1  (since (-i)^4 = 1)
        # d8 = c8 * 1  (since (-i)^8 = 1)
        # d12 = c12 * 1 (since (-i)^12 = 1)
        # So d_full = c_full for these even indices!
        # Thus I(d) = I(c).
        # So objective is I(c)^2.
        
        val = np.einsum('ijkl, i, j, k, l', K, c_full, c_full, c_full, c_full)
        return val * val

    # Optimization
    # We want to minimize objective(c).
    # Bounds: We can restrict to a range, e.g., [-10, 10]
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Use differential evolution for global search
    # maxiter=100, popsize=50
    # This should be fast enough.
    
    result = opt.differential_evolution(objective, bounds, maxiter=100, popsize=50, tol=1e-8, seed=42, polish=True)
    
    best_c = result.x
    
    # Verify time
    if time.time() > deadline - 5.0:
         return best_c
         
    # Optional: Refine with L-BFGS-B
    # result2 = opt.minimize(objective, best_c, method='L-BFGS-B', bounds=bounds)
    # if result2.success:
    #     best_c = result2.x
        
    return best_c
# EVOLVE-BLOCK-END
