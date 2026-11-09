"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 double-root locations z_1..z_12.
    
    Strategy: Use the positive roots of the derivative of the generalized Laguerre
    polynomial L_13^(-1/2)(x) as the initial guess for the double roots.
    
    The key insight is that the Laguerre polynomials form an orthogonal basis,
    and their roots have nice interlacing properties. By choosing the z_i to be
    related to the roots of a Laguerre polynomial, we align the double-root constraints
    with the natural orthogonality structure.
    
    For L_n^(alpha)(x) with alpha = -1/2, the derivative L_n^(alpha)'(x) has n-1 
    positive roots that interlace with the roots of L_n^(alpha)(x).
    
    We'll use the roots of d/dx L_13^(-1/2)(x), which gives us 12 positive roots.
    """
    deadline = time.time() + 600  # 600 second deadline
    
    try:
        # The derivative of L_13^(-1/2)(x) can be computed using the relation:
        # d/dx L_n^(alpha)(x) = -L_{n-1}^(alpha+1)(x)
        # So d/dx L_13^(-1/2)(x) = -L_12^(1/2)(x)
        # The roots of L_12^(1/2)(x) are the critical points of L_13^(-1/2)(x)
        
        # Get the roots of L_12^(1/2)(x)
        # scipy's roots_laguerre gives roots of L_n^(alpha) for alpha >= -0.5
        # For alpha = 1/2, this should work
        
        # Actually, let me use a different approach. Let's compute the roots
        # of the derivative of L_13^(-1/2)(x) numerically.
        
        # The generalized Laguerre polynomial L_n^(alpha)(x) can be evaluated
        # using scipy.special.eval_genlaguerre
        from scipy.special import eval_genlaguerre
        
        # L_13^(-1/2)(x) = sum_{k=0}^{13} (-1)^k * binom(13 - 1/2, 13-k) / k! * x^k
        # Its derivative is a polynomial of degree 12
        
        # Let's compute the coefficients of L_13^(-1/2)(x) and then differentiate
        n = 13
        alpha = -0.5
        
        # Coefficients of L_n^(alpha)(x):
        # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * C(n+alpha, n-k) / k! * x^k
        # where C(a, b) = a*(a-1)*...*(a-b+1)/b!
        
        coeffs = []
        for k in range(n + 1):
            # C(n + alpha, n - k)
            m = n - k
            if m == 0:
                c_val = 1.0
            else:
                c_val = 1.0
                for j in range(m):
                    c_val *= (n + alpha - j)
                c_val /= math.factorial(m)
            
            coeff_k = ((-1) ** k) * c_val / math.factorial(k)
            coeffs.append(coeff_k)
        
        # coeffs[k] is the coefficient of x^k in L_13^(-1/2)(x)
        # Now differentiate: d/dx sum coeffs[k] * x^k = sum k * coeffs[k] * x^(k-1)
        # The derivative has degree n-1 = 12
        deriv_coeffs = []
        for k in range(1, n + 1):
            deriv_coeffs.append(k * coeffs[k - 1])
        
        # deriv_coeffs[j] is the coefficient of x^j in L_13^(-1/2)'(x)
        # Find the roots of this polynomial
        # Use numpy's roots function
        roots = np.roots(deriv_coeffs[::-1])  # numpy expects highest degree first
        
        # Filter for positive real roots
        positive_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10 and r.real > 1e-10:
                positive_roots.append(r.real)
        
        # We should have exactly 12 positive roots
        if len(positive_roots) != 12:
            # Fallback: use a simple geometric progression
            positive_roots = [0.1 * (i + 1) for i in range(12)]
        
        # Sort the roots
        positive_roots.sort()
        
        # Convert to list of floats
        z_list = [float(z) for z in positive_roots]
        
        # Verify they are finite, > 0, and pairwise distinct
        for z in z_list:
            if not math.isfinite(z) or z <= 0:
                raise ValueError(f"Invalid root: {z}")
        
        for i in range(len(z_list)):
            for j in range(i + 1, len(z_list)):
                if abs(z_list[i] - z_list[j]) < 1e-12:
                    raise ValueError(f"Roots not distinct: {z_list[i]} and {z_list[j]}")
        
        return z_list
        
    except Exception as e:
        # Fallback to a simple valid configuration
        return [0.5 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
