"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import hermite
import sympy as sp


def construct():
    """Find optimal coefficients c0, c1, c2 to minimize the uncertainty constant."""
    
    # Use sympy for exact-ish polynomial construction to verify coefficients if needed,
    # but we will use hardcoded coefficients for speed in the inner loop.
    
    # Coefficients for H0, H4, H8, H12 (Physicists' Hermite)
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12 = 4096x^12 - 98304x^10 + 1032192x^8 - 5308416x^6 + 12096000x^4 - 12902400x^2 + 4435200
    
    def compute_Q_coefficients(c0, c1, c2):
        """
        Returns coefficients of Q(u) = g(x)/x^2 where u=x^2.
        Q(u) = a5*u^5 + a4*u^4 + a3*u^3 + a2*u^2 + a1*u + a0
        """
        c3 = -(c0 + 12*c1 + 1680*c2) / 4435200.0
        
        # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
        # Powers of x^2 (u):
        # u^6 (x^12): c3 * 4096
        # u^5 (x^10): c3 * (-98304)
        # u^4 (x^8):  c2*256 + c3*1032192
        # u^3 (x^6):  c2*(-3072) + c3*(-5308416)
        # u^2 (x^4):  c1*16 + c2*11520 + c3*12096000
        # u^1 (x^2):  c1*(-48) + c2*(-13440) + c3*(-12902400)
        # u^0 (x^0):  c0 + c1*12 + c2*1680 + c3*4435200 = 0
        
        # q(x) = g(x)/x^2
        # So Q(u) = g(sqrt(u))/u
        # Coeff of u^5 in Q is coeff of x^12 in g: c3 * 4096
        # Coeff of u^4 in Q is coeff of x^10 in g: c3 * (-98304)
        # Coeff of u^3 in Q is coeff of x^8 in g: c2*256 + c3*1032192
        # Coeff of u^2 in Q is coeff of x^6 in g: c2*(-3072) + c3*(-5308416)
        # Coeff of u^1 in Q is coeff of x^4 in g: c1*16 + c2*11520 + c3*12096000
        # Coeff of u^0 in Q is coeff of x^2 in g: c1*(-48) + c2*(-13440) + c3*(-12902400)
        
        a5 = c3 * 4096.0
        a4 = c3 * (-98304.0)
        a3 = c2*256.0 + c3*1032192.0
        a2 = c2*(-3072.0) + c3*(-5308416.0)
        a1 = c1*16.0 + c2*11520.0 + c3*12096000.0
        a0 = c1*(-48.0) + c2*(-13440.0) + c3*(-12902400.0)
        
        return np.array([a5, a4, a3, a2, a1, a0])

    def objective(coeffs):
        c0, c1, c2 = coeffs
        try:
            Q_coeffs = compute_Q_coefficients(c0, c1, c2)
            # Check if leading coefficient is zero (degenerate)
            if np.all(np.abs(Q_coeffs) < 1e-12):
                return 1e100
                
            # Find roots of Q(u)
            roots = np.roots(Q_coeffs)
            
            # Filter for real positive roots
            # Use a tolerance for imaginary part
            real_roots = roots[np.abs(roots.imag) < 1e-9].real
            positive_roots = real_roots[real_roots > 1e-9]
            
            if len(positive_roots) == 0:
                return 1e100

            # We need the largest real root where Q changes sign.
            # For a polynomial, if the leading coefficient is positive,
            # Q(u) -> +inf as u -> +inf.
            # A sign change at a simple root means the function crosses zero.
            # We want the LARGEST such root.
            
            # Sort positive roots in descending order
            positive_roots.sort(reverse=True)
            
            max_r2 = 0.0
            for r in positive_roots:
                # Check sign change
                # Evaluate Q at r - eps and r + eps
                eps = 1e-7 * max(1.0, abs(r))
                Q_left = np.polyval(Q_coeffs, r - eps)
                Q_right = np.polyval(Q_coeffs, r + eps)
                
                if Q_left * Q_right < 0:
                    # Sign change found
                    max_r2 = r*r
                    break # Since sorted descending, first one is largest
                    
            if max_r2 == 0.0:
                return 1e100
                
            return 1.0 / max_r2 # We want to minimize 1/r^2 to maximize r^2
            
        except Exception:
            return 1e100

    # Seed
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Bounds for Differential Evolution
    # Based on seed: c0 ~ -1, c1 ~ 0.04, c2 ~ 0.0003
    # We allow a wide range to ensure we don't miss the global optimum
    bounds = [
        (-5.0, 5.0),    # c0
        (-1.0, 1.0),    # c1
        (-0.01, 0.01)   # c2
    ]
    
    # Run Differential Evolution
    # This is a global optimizer, robust to local minima
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            popsize=20, 
            tol=1e-12,
            polish=True  # This runs Nelder-Mead on the best result
        )
        
        best_coeffs = de_result.x
        best_val = de_result.fun
        
        # If DE failed or found a bad value, try local optimization from seed
        if best_val > 1e90:
             raise Exception("DE failed")
             
        # Final polish with Nelder-Mead starting from DE result
        # DE's polish might be insufficient due to objective noise/discontinuity
        local_result = minimize(
            objective, 
            best_coeffs, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        
        if local_result.fun < best_val:
            best_coeffs = local_result.x
            best_val = local_result.fun
            
        # Also try local optimization from seed just in case
        local_seed_result = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        if local_seed_result.fun < best_val:
            best_coeffs = local_seed_result.x
            best_val = local_seed_result.fun

        return best_coeffs

    except Exception as e:
        # Fallback to seed if everything fails
        return x0

# EVOLVE-BLOCK-END
