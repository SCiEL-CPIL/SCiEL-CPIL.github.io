"""Refined uncertainty inequality: Dyson-gas mean-field fixed point for Laguerre roots."""
import numpy as np
from scipy.optimize import brentq
import math

# Constants
M = 12
N = 1  # dimension

def _laguerre_zeros_alpha_beta(alpha, beta, degree):
    """
    Compute zeros of generalized Laguerre polynomial L_degree^(alpha)(x)
    using Golub-Welsch on Jacobi matrix.
    For Laguerre: weight x^alpha e^{-x}, alpha > -1.
    """
    n = degree
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = 2 * (i + 1) + alpha + 1
    for i in range(n - 1):
        val = (i + 1) * (i + 1 + alpha)
        J[i, i + 1] = math.sqrt(val)
        J[i + 1, i] = math.sqrt(val)
    
    eigvals = np.linalg.eigvalsh(J)
    return np.sort(eigvals)

def _compute_score(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    if np.any(np.diff(roots) <= 0):
        return float('inf')
    
    seed_roots = 4.0 * np.arange(1, 13)
    seed_score = np.exp(np.mean(np.log(1 + seed_roots)))
    current_score = np.exp(np.mean(np.log(1 + roots)))
    
    return current_score / seed_score

def _compute_laguerre_zeros(n, alpha=0):
    """Compute the n zeros of L_n^(alpha)(x)."""
    return _laguerre_zeros_alpha_beta(alpha, 0, n)

def _dyson_gas_equilibrium(roots, beta=2.0, mu=1.0):
    """
    Solve the Dyson gas equilibrium for double roots.
    
    For P(x) = prod (x - r_i)^2, the logarithmic derivative is:
    P'(x)/P(x) = 2 * sum_j 1/(x - r_j)
    
    At equilibrium, the confining potential V'(x) = mu * x + c balances
    the repulsive force. For Laguerre weight x^alpha * exp(-x), the natural
    confining potential is linear: V'(x) = x + alpha/x.
    
    The equilibrium condition for each root r_i is:
    2 * sum_{j != i} 1/(r_i - r_j) = r_i + alpha/r_i
    
    We solve this system via successive over-relaxation with brentq.
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    alpha = 0.0  # For standard Laguerre L_n^(0)
    
    for iteration in range(1000):
        new_roots = roots.copy()
        max_change = 0.0
        
        for i in range(n):
            other_roots = np.delete(roots, i)
            other_sorted = np.sort(other_roots)
            r_guess = roots[i]
            
            # Determine bracket between neighboring roots
            insert_pos = np.searchsorted(other_sorted, r_guess)
            
            if insert_pos == 0:
                left = 1e-10
                right = other_sorted[0] - 1e-8
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-8
                right = r_guess * 1.5 + 10.0
            else:
                left = other_sorted[insert_pos - 1] + 1e-8
                right = other_sorted[insert_pos] - 1e-8
            
            if left < 1e-10:
                left = 1e-10
            if right <= left:
                # Expand bracket
                left = max(1e-10, r_guess * 0.5)
                right = r_guess * 1.5 + 1.0
            
            def f(r):
                if r <= 1e-12:
                    return 1e15
                if np.any(np.abs(r - other_roots) < 1e-12):
                    return 1e15
                s = np.sum(1.0 / (r - other_roots))
                return 2.0 * s - r - alpha / r
            
            found = False
            for attempt in range(100):
                try:
                    f_left = f(left)
                    f_right = f(right)
                    
                    if not np.isfinite(f_left) or not np.isfinite(f_right):
                        left = max(1e-10, left * 0.5)
                        right = right * 1.5 + 1.0
                        continue
                    
                    if f_left * f_right < 0:
                        r_new = brentq(f, left, right, xtol=1e-15, rtol=1e-15, maxiter=300)
                        new_roots[i] = r_new
                        found = True
                        break
                    elif abs(f_left) < 1e-15:
                        new_roots[i] = left
                        found = True
                        break
                    elif abs(f_right) < 1e-15:
                        new_roots[i] = right
                        found = True
                        break
                    else:
                        # Widen bracket
                        left = max(1e-10, left * 0.5)
                        right = right * 1.5 + 1.0
                except Exception:
                    left = max(1e-10, left * 0.5)
                    right = right * 1.5 + 1.0
            
            if not found:
                new_roots[i] = r_guess
            
            max_change = max(max_change, abs(new_roots[i] - roots[i]))
        
        roots = new_roots
        
        if max_change < 1e-15:
            break
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-10)
    
    return roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses Dyson gas mean-field fixed point with double-root modification.
    The equilibrium condition is: 2*sum_{j!=i} 1/(r_i - r_j) = r_i (for alpha=0)
    """
    # Start with Laguerre zeros as initial guess
    init_roots = _compute_laguerre_zeros(12, alpha=0)
    
    # Solve the Dyson gas equilibrium for double roots
    # With beta=2 (double root repulsion) and linear confining potential V'(x)=x
    roots = _dyson_gas_equilibrium(init_roots, beta=2.0, mu=1.0)
    
    # Verify constraints
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        roots = _compute_laguerre_zeros(12, alpha=0)
    
    return [float(r) for r in roots]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    roots = construct()
    print("Roots:", roots)
    print("Score:", _compute_score(roots))
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    print("Seed score:", _compute_score(seed_roots))
