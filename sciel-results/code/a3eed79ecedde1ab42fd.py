"""
Combined program: Uses a log-spaced geometric progression for the double roots z_i,
optimizing the base and ratio via a coarse 2D grid search with a fast float-based
approximation of the score. This exploits the exponential decay of Laguerre polynomials
to push the largest root of the quotient q(x) toward zero.
"""
import time
import math
import sys
import numpy as np
from scipy.special import eval_genlaguerre, binom
import numpy.polynomial.polynomial as P

# EVOLVE-BLOCK-START

def _approx_score(z_list):
    """
    Fast float-based approximation of the score B (not divided by 2*pi).
    Returns B for optimization.
    """
    try:
        z = np.array(z_list, dtype=float)
        if np.any(z <= 0) or np.any(np.diff(np.sort(z)) < 1e-10):
            return 1e10 # Penalty for invalid
        
        # Construct Laguerre polynomials L_{2k}^{(-1/2)}(x) for k=0..25
        # We need their values and derivatives at x=0 and x=z_i
        # L_n^{(alpha)}(x) can be computed using scipy.special.eval_genlaguerre
        # Derivative: L_n^{(\alpha)\prime}(x) = -L_{n-1}^{(\alpha+1)}(x)
        
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Precompute Laguerre values and derivatives at 0 and z_i
        for k in range(26):
            n = 2 * k
            alpha = -0.5
            
            # Value at 0
            L_0 = eval_genlaguerre(n, alpha, 0)
            # Derivative at 0
            if n > 0:
                Ld_0 = -eval_genlaguerre(n-1, alpha+1, 0)
            else:
                Ld_0 = 0.0
                
            A[0, k] = L_0
            A[1, k] = Ld_0
            
            for i in range(12):
                zi = z[i]
                L_zi = eval_genlaguerre(n, alpha, zi)
                if n > 0:
                    Ld_zi = -eval_genlaguerre(n-1, alpha+1, zi)
                else:
                    Ld_zi = 0.0
                
                A[2 + 2*i, k] = L_zi
                A[3 + 2*i, k] = Ld_zi
        
        b[1] = 1.0
        
        # Solve linear system
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e10 # Singular system penalty
            
        # Construct g(x)
        # g(x) = sum c_k L_{2k}^{(-1/2)}(x)
        # Get coefficients of each L_{2k}^{(-1/2)}(x)
        # L_n^{(alpha)}(x) = sum_{j=0}^n (-1)^j binomial(n+alpha, n-j) x^j / j!
        
        g_coeffs = np.zeros(51) # Degree up to 50
        
        for k in range(26):
            n = 2 * k
            alpha = -0.5
            # Coefficients of L_n^{(alpha)}(x)
            coeffs_L = np.zeros(n+1)
            for j in range(n+1):
                binom_val = binom(n + alpha, n - j)
                coeffs_L[j] = binom_val * ((-1)**j) / math.factorial(j)
            
            g_coeffs[:n+1] += c[k] * coeffs_L
        
        # Denominator: x * prod_{i=1}^{12} (x - z_i)^2
        # Degree 1 + 24 = 25
        # Construct polynomial with roots 0, z_1, z_1, ..., z_12, z_12
        roots_denom = np.concatenate(([0.0], np.repeat(z, 2)))
        denom_coeffs = np.poly(roots_denom) # Descending order
        
        # Convert g_coeffs to descending order
        g_desc = g_coeffs[::-1]
        
        # Polynomial division
        # q(x) = g(x) / denom(x)
        # Use np.polydiv
        q_desc, rem = np.polydiv(g_desc, denom_coeffs)
        
        if np.any(np.abs(rem) > 1e-8):
            # Not exact division? Should be exact theoretically.
            pass
        
        # Find roots of q(x)
        if len(q_desc) == 0:
            return 0.0
        if len(q_desc) == 1:
            return 0.0
            
        q_roots = np.roots(q_desc)
        
        # Filter real roots
        real_roots = []
        for r in q_roots:
            if np.isreal(r) and np.real(r) > 1e-10:
                real_roots.append(np.real(r))
                
        if not real_roots:
            return 0.0
            
        return max(real_roots)
        
    except Exception as e:
        return 1e10

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Strategy: Use a log-spaced geometric progression for z_i, optimizing base and ratio.
    """
    start_time = time.time()
    deadline = start_time + 580.0 # 20s margin
    
    best_z = None
    best_score = 1e10
    
    # Coarse 2D grid search over base z_1 and ratio r
    # z_i = z_1 * r^(i-1) for i=1..12
    # We want z_i > 0 and distinct.
    # Try a range of bases and ratios.
    
    # Log-spaced bases from 0.1 to 100
    bases = np.logspace(-1, 2, 20)
    # Log-spaced ratios from 0.5 to 5
    ratios = np.logspace(-0.3, 0.7, 20)
    
    # To save time, we can first do a very coarse grid, then refine.
    # But 20x20 = 400 evaluations. Each evaluation takes ~0.1s?
    # 400 * 0.1 = 40s. That's okay.
    
    # Let's do a slightly finer grid if time permits.
    # First pass: coarse
    bases_coarse = np.logspace(-1, 2, 10)
    ratios_coarse = np.logspace(-0.3, 0.7, 10)
    
    for base in bases_coarse:
        if time.time() > deadline - 60:
            break
        for ratio in ratios_coarse:
            z_list = [base * (ratio ** i) for i in range(12)]
            
            # Check validity
            if any(z <= 0 for z in z_list):
                continue
            if any(abs(z_list[i] - z_list[j]) < 1e-10 for i in range(12) for j in range(i+1, 12)):
                continue
                
            score = _approx_score(z_list)
            if score < best_score:
                best_score = score
                best_z = z_list[:]
                
    # Second pass: refine around the best base and ratio
    if best_z is not None:
        # Extract best base and ratio from best_z
        # z_1 = base, z_2 = base * ratio => ratio = z_2 / z_1
        base_best = best_z[0]
        ratio_best = best_z[1] / best_z[0]
        
        # Local search around base_best and ratio_best
        base_range = np.linspace(base_best * 0.8, base_best * 1.2, 10)
        ratio_range = np.linspace(ratio_best * 0.9, ratio_best * 1.1, 10)
        
        for base in base_range:
            if time.time() > deadline - 60:
                break
            for ratio in ratio_range:
                z_list = [base * (ratio ** i) for i in range(12)]
                
                if any(z <= 0 for z in z_list):
                    continue
                if any(abs(z_list[i] - z_list[j]) < 1e-10 for i in range(12) for j in range(i+1, 12)):
                    continue
                    
                score = _approx_score(z_list)
                if score < best_score:
                    best_score = score
                    best_z = z_list[:]
    
    # Final Validation and Return
    if best_z is None:
        # Fallback
        final_z = [4.0 * (i + 1) for i in range(12)]
    else:
        final_z = list(best_z)
        for i in range(12):
            if not np.isfinite(final_z[i]) or final_z[i] <= 0:
                final_z = [4.0 * (i + 1) for i in range(12)]
                break
        for i in range(12):
            for j in range(i+1, 12):
                if abs(final_z[i] - final_z[j]) < 1e-9:
                    final_z = [4.0 * (i + 1) for i in range(12)]
                    break
            else:
                continue
            break
            
    return final_z

# EVOLVE-BLOCK-END
