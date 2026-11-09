"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import roots_laguerre, factorial

def construct():
    t_start = time.time()
    DEADLINE = 615.0
    
    m = 12
    n = 1
    
    # Get the zeros of L_12(x)
    try:
        x0 = roots_laguerre(m)
        zeros = x0[0].real
        zeros = sorted(zeros)
    except Exception:
        # Fallback: compute manually
        c = []
        for k in range(m + 1):
            val = ((-1) ** k) * math.comb(m + k, k) / math.factorial(k)
            c.append(val)
        c_rev = list(reversed(c))
        roots = np.roots(c_rev)
        zeros = sorted([r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0])
    
    if len(zeros) != m:
        return [4.0 * (i + 1) for i in range(12)]
    
    # The C4 upper bound for the sign uncertainty with m double roots
    # in the Cohn-Goncalves Laguerre construction.
    # 
    # For the refined construction, the upper bound on the number of sign
    # changes involves the L4 norm of a certain auxiliary function.
    # 
    # The key quantity is: given roots x_1 < ... < x_m (all positive),
    # form the polynomial P(x) = prod_{i=1}^m (x - x_i)^2
    # and the Laguerre-based weight. The C4 bound is:
    #
    # C4 = (1/2) * integral_0^infty |f(x)|^4 dx
    #
    # where f is constructed from the Laguerre polynomial evaluated at
    # the scaled roots.
    #
    # For the specific Cohn-Goncalves construction with m double roots
    # at positions s*x_i (where x_i are L_m zeros and s is a scale factor),
    # the upper bound can be computed as:
    #
    # C4(s) = sum_{i=1}^m w_i(s)
    #
    # where w_i depends on the Laguerre polynomial values at the roots.
    
    def compute_C4(scale):
        """Compute the C4 upper bound for a given scale factor."""
        if scale <= 0:
            return 1e10
        
        # Scaled roots
        scaled_roots = scale * np.array(zeros)
        
        # The C4 bound for the Cohn-Goncalves construction with m double roots
        # is related to the sum of squared Laguerre polynomial values at the roots.
        # 
        # Specifically, for the sign uncertainty problem, the bound is:
        # C4 = (1/2) * sum_{i=1}^m (L_m'(s*x_i) / L_m(s*x_i))^2 * (some weight)
        #
        # But more precisely, in the Cohn-Goncalves framework, the bound
        # involves the Christoffel-Darboux kernel or related quantities.
        
        # A simpler formulation: the upper bound on the number of sign changes
        # for a function with m double zeros at positions x_i is bounded by
        # a function of the spacing and the Laguerre weight.
        
        # For the refined Cohn-Goncalves construction, the C4 bound is:
        # C4 = (1/2) * (m + 1) * (1 + 2*sum_{i=1}^m (x_i / sum(x_j))^2)
        # This is a simplified model. Let's use a more principled approach.
        
        # The actual C4 for the sign uncertainty with Laguerre weight:
        # The bound is proportional to the L4 norm of the reproducing kernel
        # evaluated at the double roots.
        
        # Using the Christoffel-Darboux formula for Laguerre polynomials:
        # K_m(x,y) = sum_{k=0}^m L_k(x) L_k(y) / h_k
        # where h_k = 1 (for standard Laguerre with weight e^{-x})
        
        # The C4 bound involves:
        # C4 = (1/2) * sum_{i=1}^m K_m(s*x_i, s*x_i)
        
        # But K_m(x,x) = sum_{k=0}^m L_k(x)^2
        
        # Let's compute this directly.
        scaled = scale * np.array(zeros)
        
        # Compute L_k(x) for k=0..m at each scaled root
        # L_k(x) = sum_{j=0}^k binom(k, j) * (-x)^j / j!
        
        C4_val = 0.0
        for x in scaled:
            kernel_val = 0.0
            for k in range(m + 1):
                # Compute L_k(x)
                Lk = 0.0
                for j in range(k + 1):
                    Lk += math.comb(k, j) * ((-x) ** j) / math.factorial(j)
                kernel_val += Lk ** 2
            C4_val += kernel_val
        
        C4_val *= 0.5
        return C4_val
    
    # Optimize the scale factor
    best_scale = 1.0
    best_C4 = compute_C4(1.0)
    
    if time.time() - t_start > DEADLINE:
        return [float(s) for s in (best_scale * np.array(zeros))]
    
    # Use scipy to minimize C4 over scale
    try:
        result = minimize_scalar(
            lambda s: compute_C4(s),
            bounds=(0.1, 5.0),
            method='bounded',
            options={'xatol': 1e-8, 'maxiter': 200}
        )
        if result.success and result.fun < best_C4:
            best_scale = result.x
            best_C4 = result.fun
    except Exception:
        pass
    
    if time.time() - t_start > DEADLINE:
        return [float(s) for s in (best_scale * np.array(zeros))]
    
    # Fine-tune with a local optimization
    try:
        result2 = minimize_scalar(
            lambda s: compute_C4(s),
            bounds=(best_scale * 0.9, best_scale * 1.1),
            method='bounded',
            options={'xatol': 1e-10, 'maxiter': 100}
        )
        if result2.success and result2.fun < best_C4:
            best_scale = result2.x
            best_C4 = result2.fun
    except Exception:
        pass
    
    # Compute final roots
    final_roots = best_scale * np.array(zeros)
    final_roots = sorted(final_roots)
    
    if time.time() - t_start > DEADLINE:
        return [float(r) for r in final_roots]
    
    return [float(r) for r in final_roots]

# EVOLVE-BLOCK-END
