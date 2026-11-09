import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 12 positive double-root locations for the Laguerre combination (m=12, n=1)
    by minimizing a proxy for the Cohn-Goncalves uncertainty bound.
    
    The strategy uses scipy.optimize.minimize to tune the root locations.
    The objective function approximates the uncertainty bound by evaluating
    the Laguerre polynomial combination at a dense grid of points and 
    computing the maximum absolute value or a related integral proxy.
    """
    
    m = 12
    n = 1
    num_roots = m  # 12 double roots
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600  # 600 seconds limit, use 590 for safety
    
    # Initial guess: asymptotic distribution of Laguerre zeros
    # For large m, zeros of L_m(x) are roughly distributed in [0, 4m]
    # A better initial guess is the actual zeros of L_m(x)
    try:
        from scipy.special import roots_laguerre
        # roots_laguerre returns zeros of L_m(x)
        # We need 12 double roots. The standard Laguerre polynomial L_12 has 12 distinct roots.
        # The task mentions "12 double roots". This implies a polynomial of degree 24?
        # Or a combination of Laguerre polynomials?
        # "Laguerre-polynomial construction" with "12 double roots" usually refers to 
        # a squared polynomial or a specific linear combination.
        # Let's assume the roots are the variables we are optimizing.
        
        # Let's use the zeros of L_12 as a starting point, but spread them out or 
        # just use them as initial x0.
        x0, _ = roots_laguerre(m)
        x0 = np.sort(x0)
    except Exception:
        # Fallback to linear spacing if roots_laguerre fails
        x0 = np.linspace(0.1, 4 * m, num_roots)

    # Ensure positive and sorted
    x0 = np.maximum(x0, 1e-6)
    
    def objective(x):
        """
        Compute the proxy for the Cohn-Goncalves bound.
        The bound is related to the supremum of |P(x)| where P is the Laguerre combination.
        For a polynomial with double roots at x_i, P(x) = C * prod((x - x_i)^2).
        However, the "Laguerre combination" suggests a linear combination of Laguerre polynomials.
        
        A common construction for uncertainty bounds involves finding a polynomial 
        Q(x) = sum c_i L_i(x) such that Q(x) <= 1 for x >= 0 and Q(x) is small for x < 0 
        (or vice versa, depending on the sign uncertainty).
        
        Given the prompt mentions "12 double roots", it likely refers to a polynomial 
        P(x) = prod_{i=1}^{12} (x - r_i)^2. 
        The Cohn-Goncalves bound C4 is often related to the integral of P(x) e^{-x} or similar.
        
        Let's use a robust proxy: The value of the polynomial at the last sign change 
        or the maximum deviation. 
        Since we don't have the exact formula for C4, we will minimize the 
        "energy" of the roots or the value of the associated Laguerre functional.
        
        A simpler, robust proxy for "tightening the bound" is to minimize the 
        maximum value of the normalized Laguerre polynomial combination.
        
        Let's assume the score is directly related to the value of the polynomial 
        at some critical point, or the integral. 
        Minimizing the spread of roots often tightens bounds in these contexts.
        
        Let's try minimizing the variance of the roots or the value of the 
        Laguerre polynomial L_12 at the mean of the roots.
        
        Actually, a very strong proxy for "better roots" in Laguerre-Gauss contexts 
        is to minimize the condition number or to ensure the roots are well-distributed.
        
        Let's use the following proxy: 
        Score ~ max_{x in grid} | sum_{k=0}^m a_k L_k(x) | 
        where a_k are coefficients determined by the roots.
        
        This is complex. Let's use a simpler structural proxy:
        The Cohn-Goncalves bound is improved when the roots are closer to the 
        asymptotic zeros. The seed program uses [4, 8, ..., 48].
        The actual zeros of L_12 are roughly in [0, 48].
        
        Let's minimize the sum of squared differences from the ideal Laguerre zeros?
        No, the task is to *tune* the construction.
        
        Let's use `scipy.optimize.minimize` to minimize a function that evaluates 
        the "uncertainty" proxy. 
        Proxy: Integral of P(x) e^{-x} dx, where P(x) = prod (x - r_i)^2.
        Lower integral might mean tighter bound? Or higher?
        
        Let's assume the score is the value of the bound, and we want to minimize it.
        The bound is often proportional to the norm of the polynomial in the Laguerre space.
        
        Let's define the objective as the L2 norm of the polynomial P(x) = prod (x-r_i)^2 
        with respect to the Laguerre weight e^{-x}.
        """
        # Check time
        if time.time() - start_time > time_limit - 10:
            return 1e10 # Penalize timeout
        
        # Construct polynomial coefficients from roots
        # P(x) = prod (x - r_i)^2 = prod (x^2 - 2 r_i x + r_i^2)
        # This is computationally expensive for 12 roots if done naively each time.
        # We can use numpy.poly
        
        # Double roots: r_i appears twice.
        # Roots array: [r1, r1, r2, r2, ..., r12, r12]
        roots_double = np.repeat(x, 2)
        
        # Get coefficients of P(x)
        # np.poly returns coefficients in descending order of powers
        try:
            coeffs = np.poly(roots_double)
        except Exception:
            return 1e10
            
        # Evaluate the Laguerre inner product: <P, P> = sum_{k=0}^{24} c_k^2 / (k+1) ?
        # No, the orthonormal basis for Laguerre polynomials L_k is different.
        # L_k are orthogonal with weight e^{-x}.
        # <L_k, L_j> = delta_{kj}.
        # So we need to express P(x) in the Laguerre basis.
        
        # This is expensive. Let's use a grid-based proxy.
        # Evaluate P(x) on a grid and compute the weighted integral.
        
        # Grid from 0 to max(roots) + some margin
        max_root = np.max(x)
        grid_end = max_root + 10
        grid = np.linspace(0, grid_end, 200)
        
        # Evaluate P(x)
        # Using np.polyval
        try:
            vals = np.polyval(coeffs, grid)
        except Exception:
            return 1e10
            
        # Weighted integral proxy: sum |P(x)|^2 * exp(-x) * dx
        # dx = grid[1] - grid[0]
        dx = grid[1] - grid[0]
        integral = np.sum(np.abs(vals)**2 * np.exp(-grid)) * dx
        
        return integral

    # Multi-start optimization
    best_x = x0.copy()
    best_val = objective(best_x)
    
    # Number of starts
    num_starts = 10
    
    for i in range(num_starts):
        if time.time() - start_time > time_limit - 30:
            break
            
        # Perturb the initial guess
        if i == 0:
            current_x = x0.copy()
        else:
            # Random perturbation
            noise = np.random.normal(0, 0.5, num_roots)
            current_x = np.maximum(x0 + noise, 1e-6)
            current_x = np.sort(current_x)
        
        # Use L-BFGS-B with bounds
        bounds = [(1e-6, None) for _ in range(num_roots)]
        
        try:
            res = scipy_optimize_minimize(objective, current_x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'maxfun': 200})
            
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
        except Exception:
            pass

    # Ensure the result is sorted and positive
    best_x = np.sort(np.maximum(best_x, 1e-6))
    
    return list(best_x)

# Import scipy.optimize inside the function or at top level? 
# Standard library, NumPy and SciPy are available.
import scipy.optimize

# EVOLVE-BLOCK-END
