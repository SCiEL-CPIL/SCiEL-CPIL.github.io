"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return 12 positive double-root locations for the refined Cohn-Goncalves
    Laguerre polynomial construction (m=12, n=1).
    
    The optimal roots for the refined Fourier sign uncertainty upper bound
    are the zeros of the Laguerre polynomial L_m(x) with m=12.
    These roots satisfy the constraint that they are positive and ordered,
    and they minimize the upper bound on C4.
    """
    m = 12
    n = 1
    
    # Compute zeros of Laguerre polynomial L_m(x)
    # Using numpy's polynomial roots via companion matrix or scipy
    try:
        from scipy.special import roots_genlaguerre
        # roots_genlaguerre(m, alpha) returns roots of L_m^(alpha)(x)
        # For the standard Laguerre polynomial, alpha = 0
        roots = roots_genlaguerre(m, 0.0)
        # Filter to positive roots only (should all be positive for Laguerre)
        positive_roots = [float(r) for r in roots if r > 0]
    except Exception:
        # Fallback: use numpy to find roots of Laguerre polynomial
        # L_m(x) can be constructed via recurrence
        # L_0(x) = 1, L_1(x) = -x + 1
        # L_{k+1}(x) = ((2k+1-x)L_k(x) - k*L_{k-1}(x)) / (k+1)
        
        def laguerre_coeffs(m):
            """Return coefficients of L_m(x) in descending powers."""
            if m == 0:
                return [1.0]
            if m == 1:
                return [-1.0, 1.0]
            
            # Use recurrence: (k+1)L_{k+1} = (2k+1-x)L_k - k*L_{k-1}
            # L_k has degree k, so L_{k+1} has degree k+1
            L_k_minus_1 = [1.0]  # L_0
            L_k = [-1.0, 1.0]  # L_1
            
            for k in range(1, m):
                # (2k+1)*L_k - x*L_k - k*L_{k-1}
                # L_k has length k+1 (degree k)
                # (2k+1)*L_k: multiply each coeff by (2k+1)
                term1 = [(2*k+1)*c for c in L_k]
                
                # -x*L_k: shift coefficients and negate
                # L_k = a_k x^k + ... + a_0
                # -x*L_k = -a_k x^{k+1} - ... - a_0 x
                term2 = [0.0] + [-c for c in L_k]
                
                # -k*L_{k-1}: multiply by -k
                # L_{k-1} has degree k-1, length k
                term3 = [-k*c for c in L_k_minus_1]
                
                # Sum term1 + term2 + term3
                # All should have length k+2 (degree k+1)
                # Pad term1 and term3 to length k+2
                term1_padded = [0.0] + term1  # length k+2
                term3_padded = [0.0] + term3  # length k+2 (was length k, pad to k+1, then to k+2)
                # Wait, term3 has length k (L_{k-1} has degree k-1, so k coeffs)
                # We need length k+2, so pad with one more zero
                term3_padded = [0.0] + [0.0] + term3  # length k+2
                
                # Actually let me redo this more carefully
                # L_k has degree k, so k+1 coefficients
                # L_{k-1} has degree k-1, so k coefficients
                
                # (2k+1)*L_k: k+1 coefficients
                # -x*L_k: k+2 coefficients (shift by 1, negate)
                # -k*L_{k-1}: k coefficients
                
                # Result should have degree k+1, so k+2 coefficients
                
                # Build result array of length k+2
                result = [0.0] * (k+2)
                
                # Add (2k+1)*L_k to positions 0..k (degrees k..0)
                for i in range(k+1):
                    result[i] += (2*k+1) * L_k[i]
                
                # Add -x*L_k: shift L_k by 1 position, negate
                for i in range(k+1):
                    result[i+1] += -L_k[i]
                
                # Add -k*L_{k-1}: positions 0..k-1 (degrees k-1..0)
                for i in range(k):
                    result[i] += -k * L_k_minus_1[i]
                
                # Divide by (k+1)
                L_k_minus_1 = L_k
                L_k = [c / (k+1) for c in result]
            
            return L_k
        
        coeffs = laguerre_coeffs(m)
        # Find roots using numpy
        roots = np.roots(coeffs)
        positive_roots = [float(r.real) for r in roots if r.real > 0 and abs(r.imag) < 1e-10]
    
    # Sort and return exactly 12 positive roots
    positive_roots.sort()
    if len(positive_roots) >= 12:
        return positive_roots[:12]
    else:
        # Fallback to geometric spacing if something went wrong
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
