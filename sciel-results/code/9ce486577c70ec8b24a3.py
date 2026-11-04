"""Refined uncertainty inequality: Dyson-gas mean-field fixed point for Laguerre roots."""
import numpy as np
from scipy.optimize import minimize, brentq
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

def _dyson_gas_equilibrium(roots, n_dim=1):
    """
    Solve the Dyson gas equilibrium for double roots with Laguerre weight x^n_dim * exp(-x).
    
    The equilibrium condition for each root x_i is:
        sum_{j != i} 2/(x_i - x_j) = 1 - n_dim/x_i
    
    This comes from the potential V(x) = x - n_dim * ln(x), so V'(x) = 1 - n_dim/x.
    
    We use fixed-point iteration with brentq for each coordinate.
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    
    for iteration in range(2000):
        new_roots = roots.copy()
        max_change = 0.0
        
        for i in range(n):
            other_roots = np.delete(roots, i)
            other_sorted = np.sort(other_roots)
            r_guess = roots[i]
            
            # Find bracket between adjacent roots
            insert_pos = np.searchsorted(other_sorted, r_guess)
            
            if insert_pos == 0:
                left = 1e-10
                right = other_sorted[0] - 1e-10
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-10
                right = max(200.0, r_guess + 20.0)
            else:
                left = other_sorted[insert_pos - 1] + 1e-10
                right = other_sorted[insert_pos] - 1e-10
            
            if left < 1e-10:
                left = 1e-10
            if right <= left:
                left = max(1e-10, r_guess * 0.5)
                right = r_guess * 1.5
            
            def f(r):
                if r <= 1e-12:
                    return 1e15
                if np.any(np.abs(r - other_roots) < 1e-12):
                    return 1e15
                s = np.sum(1.0 / (r - other_roots))
                # Equilibrium: 2*sum(1/(r - r_j)) = 1 - n_dim/r
                return 2.0 * s - (1.0 - n_dim / r)
            
            found = False
            for attempt in range(100):
                try:
                    f_left = f(left)
                    f_right = f(right)
                    
                    if np.isnan(f_left) or np.isnan(f_right):
                        left = max(1e-10, left * 0.5)
                        right = right * 2.0
                        continue
                    
                    if f_left * f_right < 0:
                        r_new = brentq(f, left, right, xtol=1e-15, rtol=1e-15, maxiter=300)
                        new_roots[i] = r_new
                        found = True
                        break
                    elif abs(f_left) < 1e-14:
                        new_roots[i] = left
                        found = True
                        break
                    elif abs(f_right) < 1e-14:
                        new_roots[i] = right
                        found = True
                        break
                    else:
                        # Expand bracket
                        mid = (left + right) / 2.0
                        if f_left > 0:
                            left = max(1e-10, left - (right - left) * 0.5)
                        else:
                            right = right + (right - left) * 0.5
                except Exception:
                    left = max(1e-10, left * 0.5)
                    right = right * 2.0
            
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
    Uses Dyson gas mean-field fixed point with double-root modification and
    Laguerre weight x^1 * exp(-x) confining potential.
    """
    # Start with Laguerre zeros as initial guess (alpha=1 for n=1 dimension)
    init_roots = _compute_laguerre_zeros(12, alpha=1)
    
    # Solve the Dyson gas equilibrium for double roots with n_dim=1
    # Equilibrium: sum_{j!=i} 2/(x_i - x_j) = 1 - 1/x_i
    roots = _dyson_gas_equilibrium(init_roots, n_dim=1)
    
    # Verify constraints
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        # Fallback: use Laguerre zeros with alpha=1
        roots = _compute_laguerre_zeros(12, alpha=1)
    
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
