import numpy as np
import scipy.optimize as opt
import scipy.special as sp
import time

def construct():
    """
    Constructs 12 positive double-root locations for the Laguerre combination (m=12, n=1)
    that minimize the upper bound on C4.
    """
    start_time = time.time()
    deadline = 580.0  # Safety margin before 600s limit
    
    m = 12
    n = 1
    
    # Initial guess: roots of the standard Laguerre polynomial L_m^(n)(x)
    # These are positive and well-distributed.
    roots_init = sp.roots_genlaguerre(m, n)
    # Ensure they are positive (they are for m>=1, n>-1)
    # Sort them
    roots_init = np.sort(roots_init.real)
    
    # Define the cost function
    # We minimize the ratio: 
    # R = (Integral of (P')^2 e^-x x^n dx) / (Integral of P^2 e^-x x^n dx)
    # P(x) = prod (x - r_i)^2
    
    # Use Gauss-Laguerre quadrature for efficiency and accuracy
    # Degree of P is 2*m = 24.
    # Degree of P^2 is 48.
    # Degree of (P')^2 is 46.
    # Gauss-Laguerre with N nodes is exact for polynomials up to degree 2N-1.
    # We need 2N-1 >= 48 => 2N >= 49 => N >= 24.5 => N=25.
    # Let's use N=40 to be safe and accurate for the weight x^1.
    # Note: standard Gauss-Laguerre is for e^-x. The weight is x^n e^-x.
    # We can use general Gauss-Laguerre or just include x^n in the integrand.
    # Let's use standard nodes x_i, w_i for e^-x and multiply by x_i^n.
    
    quad_nodes, quad_weights = np.polynomial.laguerre.laggauss(40)
    
    def cost_func(roots_flat):
        # Check time
        if time.time() > deadline:
            return 1e10  # Large penalty to stop optimization
        
        # Reshape to 12 roots
        r = roots_flat
        # Ensure positivity (bounds will handle this, but just in case)
        if np.any(r <= 0):
            return 1e10
        
        # Compute P(x) and P'(x) at quadrature nodes
        # P(x) = prod (x - r_i)^2
        # P'(x) = P(x) * sum 2/(x - r_i)
        
        # Vectorized calculation
        # r shape (12,), quad_nodes shape (40,)
        # diff shape (40, 12)
        diff = quad_nodes[:, np.newaxis] - r[np.newaxis, :]
        
        # Calculate log(P(x)) to avoid overflow/underflow?
        # P(x) = prod (diff^2)
        # log_P = sum(2 * log(abs(diff)))
        # However, we need P(x)^2 and (P'(x))^2.
        
        # Let's compute P directly first, checking for magnitude.
        # For N=12, values can be large.
        
        # P_i = prod_{j=1}^12 (x_i - r_j)^2
        # log_P_i = 2 * sum_{j=1}^12 log(|x_i - r_j|)
        
        # To handle the ratio R = Num / Den:
        # Num = sum w_i x_i^n (P'(x_i))^2
        # Den = sum w_i x_i^n (P(x_i))^2
        
        # P'(x) = P(x) * 2 * sum_{j=1}^12 1/(x - r_j)
        # So (P')^2 = P^2 * 4 * (sum 1/(x-r_j))^2
        
        # R = sum w_i x_i^n P_i^2 S_i^2 / sum w_i x_i^n P_i^2
        # where S_i = 2 * sum_{j=1}^12 1/(x_i - r_j)
        
        # This avoids calculating huge numbers if we cancel P_i^2?
        # No, we can't cancel across the sum.
        # But we can compute log_P_i to scale.
        
        # Let's compute P_i^2 * w_i * x_i^n.
        # log_term_i = log(w_i) + n*log(x_i) + 4 * sum(log(|x_i - r_j|))
        # This might be more stable.
        
        # However, with N=12, direct calculation might be fine if we normalize.
        # Let's try direct calculation first, with scaling if needed.
        
        # Calculate S_i = sum 1/(x_i - r_j)
        # Avoid division by zero (roots should not be at nodes, but check)
        # Add small epsilon if close
        inv_diff = 1.0 / diff
        # If any diff is 0, replace with inf (contribution 0) or handle separately
        # Roots are continuous variables, probability of hitting node is 0.
        S = 2.0 * np.sum(inv_diff, axis=1)
        
        # Calculate P_i^2 * weight
        # P_i = prod (diff^2)
        # P_i^2 = prod (diff^4)
        # log_P2_i = 4 * sum(log(|diff|))
        
        # To prevent overflow in exp, we can subtract the max log value.
        # But we need the ratio.
        # Let A_i = w_i * x_i^n * P_i^2
        # log_A_i = log(w_i) + n*log(x_i) + 4 * sum(log(|diff|))
        
        # Note: w_i > 0, x_i > 0.
        # diff can be negative, but we use abs for log.
        
        log_abs_diff = np.log(np.abs(diff))
        
        # Sum over j
        sum_log_abs_diff = np.sum(log_abs_diff, axis=1)
        
        log_A = np.log(quad_weights) + n * np.log(quad_nodes) + 4.0 * sum_log_abs_diff
        
        # Normalize log_A to avoid overflow
        max_log_A = np.max(log_A)
        A_scaled = np.exp(log_A - max_log_A)
        
        # Numerator terms: A_i * S_i^2
        num_terms = A_scaled * (S ** 2)
        # Denominator terms: A_i
        den_terms = A_scaled
        
        num = np.sum(num_terms)
        den = np.sum(den_terms)
        
        if den == 0:
            return 1e10
            
        return num / den

    # Optimization
    # Bounds: roots must be positive
    bounds = [(1e-6, None) for _ in range(m)]
    
    # Use L-BFGS-B
    # Initial point
    x0 = roots_init
    
    # Run optimization
    try:
        result = opt.minimize(cost_func, x0, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': 500, 'maxfun': 1000, 'disp': False})
        
        if time.time() > deadline:
            # Return best found so far? result.x might be partial.
            # Let's just return the initial roots if we are close to deadline
            pass
        
        # If optimization succeeded or ran out of iterations, use result.x
        # Ensure results are positive
        opt_roots = result.x
        opt_roots = np.maximum(opt_roots, 1e-6)
        
        # Sort the roots
        opt_roots = np.sort(opt_roots)
        
        # Final check
        if time.time() < deadline:
             # We can try a second pass or just return
             return opt_roots.tolist()
        else:
             return opt_roots.tolist()
             
    except Exception as e:
        # If optimization fails, return initial roots
        return roots_init.tolist()

