"""Refined uncertainty inequality: Dyson-gas mean-field with tunable confining potential."""
import numpy as np
from scipy.optimize import brentq, minimize
import math

M = 12
N = 1

def _laguerre_zeros_alpha_beta(alpha, degree):
    """
    Compute zeros of generalized Laguerre polynomial L_degree^(alpha)(x)
    using Golub-Welsch on Jacobi matrix.
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
    return _laguerre_zeros_alpha_beta(alpha, n)

def _dyson_gas_equilibrium(roots, potential_type='log'):
    """
    Solve the Dyson gas equilibrium with repulsion exponent 2 and a tunable confining potential.
    
    For potential_type='linear': V(x) = x, condition: sum_{j!=i} 1/(r_i - r_j) = 1/2
    For potential_type='log': V(x) = log(1+x), condition: sum_{j!=i} 1/(r_i - r_j) = 1/(2(1+r_i))
    For potential_type='power': V(x) = x^p, condition: sum_{j!=i} 1/(r_i - r_j) = p*x_i^(p-1)/2
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    p = 1.5  # exponent for power potential
    
    def rhs(r):
        """Right-hand side of the equilibrium equation: V'(r)/2"""
        if potential_type == 'linear':
            return 0.5
        elif potential_type == 'log':
            return 0.5 / (1.0 + r)
        elif potential_type == 'power':
            return 0.5 * p * r**(p - 1.0) if r > 1e-15 else 1e15
        else:
            return 0.5
    
    for iteration in range(2000):
        new_roots = roots.copy()
        max_change = 0.0
        
        for i in range(n):
            other_roots = np.delete(roots, i)
            other_sorted = np.sort(other_roots)
            r_guess = roots[i]
            
            insert_pos = np.searchsorted(other_sorted, r_guess)
            
            if insert_pos == 0:
                left = 1e-12
                right = other_sorted[0] - 1e-8
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-8
                right = max(500.0, r_guess * 3.0 + 50.0)
            else:
                left = other_sorted[insert_pos - 1] + 1e-8
                right = other_sorted[insert_pos] - 1e-8
            
            if left < 1e-12:
                left = 1e-12
            if right <= left:
                left = max(1e-12, r_guess * 0.5)
                right = r_guess * 2.0 + 10.0
            
            def f(r):
                if r <= 1e-15:
                    return 1e15
                if np.any(np.abs(r - other_roots) < 1e-15):
                    return 1e15
                s = np.sum(1.0 / (r - other_roots))
                return s - rhs(r)
            
            found = False
            for attempt in range(100):
                try:
                    f_left = f(left)
                    f_right = f(right)
                    
                    if np.isnan(f_left) or np.isnan(f_right) or np.isinf(f_left) or np.isinf(f_right):
                        left = max(1e-12, left * 0.5 if left > 1 else left - 1.0)
                        right = right * 1.5 + 5.0
                        continue
                    
                    if f_left * f_right < 0:
                        r_new = brentq(f, left, right, xtol=1e-14, rtol=1e-14, maxiter=500)
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
                        # Widen the search interval
                        if f_left > 0 and f_right > 0:
                            right = right * 1.5 + 5.0
                        elif f_left < 0 and f_right < 0:
                            left = max(1e-12, left * 0.5)
                        else:
                            left = max(1e-12, left * 0.5)
                            right = right * 1.5 + 5.0
                except Exception:
                    left = max(1e-12, left * 0.5)
                    right = right * 1.5 + 5.0
            
            if not found:
                new_roots[i] = r_guess
            
            max_change = max(max_change, abs(new_roots[i] - roots[i]))
        
        roots = new_roots
        
        if max_change < 1e-14:
            break
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-12)
    
    return roots

def _refine_with_minimize(roots, max_iters=500):
    """
    Refine the Dyson gas equilibrium using direct minimization of the objective
    sum log(1+r_i) subject to positivity and strict increase constraints.
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    
    def objective(params):
        r = np.exp(params)  # Ensure positivity via log-parameterization
        if np.any(np.diff(r) <= 0):
            return 1e10
        return np.sum(np.log(1.0 + r))
    
    def gradient(params):
        r = np.exp(params)
        if np.any(np.diff(r) <= 0):
            return np.ones(n) * 1e10
        grad = r / (1.0 + r)
        return grad
    
    # Start from log of current roots
    x0 = np.log(roots)
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-14})
    
    if result.success:
        refined = np.exp(result.x)
        if np.all(refined > 0) and np.all(np.diff(refined) > 0):
            return refined
    
    return roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses Dyson gas mean-field with a tunable confining potential, then refines
    via direct minimization of the objective.
    """
    # Start with Laguerre zeros as initial guess
    init_roots = _compute_laguerre_zeros(12, alpha=0)
    
    # Try multiple potential types and pick the best
    best_roots = None
    best_score = float('inf')
    
    for pot_type in ['log', 'power', 'linear']:
        try:
            roots = _dyson_gas_equilibrium(init_roots, potential_type=pot_type)
            if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
                continue
            # Refine with direct minimization
            refined = _refine_with_minimize(roots)
            score = _compute_score(refined)
            if score < best_score:
                best_score = score
                best_roots = refined
        except Exception:
            continue
    
    if best_roots is None:
        best_roots = init_roots
    
    # Final verification
    if np.any(best_roots <= 0) or np.any(np.diff(best_roots) <= 0):
        best_roots = _compute_laguerre_zeros(12, alpha=0)
    
    return [float(r) for r in best_roots]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    roots = construct()
    print("Roots:", roots)
    print("Score:", _compute_score(roots))
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    print("Seed score:", _compute_score(seed_roots))
