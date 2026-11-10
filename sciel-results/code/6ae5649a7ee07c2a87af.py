"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
import sympy

# EVOLVE-BLOCK-START

def _evaluate_score(z_list):
    """
    Compute the score B/(2*pi) for a given list of 12 positive distinct roots.
    Returns None if invalid (singular system, etc.).
    """
    try:
        # Convert to exact rationals
        z_rational = [sympy.Rational(float(z)) for z in z_list]
        z_rational_sorted = sorted(z_rational)
        
        # Define symbols
        x = sympy.Symbol('x')
        alpha = sympy.Rational(-1, 2)
        
        # Build the generalized Laguerre polynomials L_0(x), L_2(x), ..., L_50(x)
        # Note: The task says L_d^(alpha)(x) but g(x) = sum c_k L_2k(x).
        # The prompt says "Let g(x) = sum_k c_k L_2k(x), k = 0..25 (degrees 0, 2, ..., 50)."
        # And "With alpha = -1/2 and L_d the generalized Laguerre polynomial L_d^(alpha)(x)"
        # So L_2k(x) in the sum likely means L_{2k}^{(-1/2)}(x).
        
        laguerres = []
        for k in range(26):
            d = 2 * k
            L_d = sympy.laguerre(d, alpha, x)
            laguerres.append(L_d)
        
        # Coefficients c_0 ... c_25
        c = [sympy.Symbol(f'c_{k}') for k in range(26)]
        
        # g(x) = sum c_k L_{2k}(x)
        g = sum(c[k] * laguerres[k] for k in range(26))
        
        # Conditions:
        # 1. g(0) = 0
        # 2. g'(0) = 1
        # 3. For each i in 0..11: g(z_i) = 0 and g'(z_i) = 0
        
        eqs = []
        eqs.append(g.subs(x, 0))  # g(0) = 0
        eqs.append(g.diff(x).subs(x, 0) - 1)  # g'(0) = 1
        
        for i in range(12):
            z_i = z_rational_sorted[i]
            eqs.append(g.subs(x, z_i))
            eqs.append(g.diff(x).subs(x, z_i))
        
        # Solve the linear system
        # There are 2 + 2*12 = 26 equations and 26 unknowns.
        try:
            solution = sympy.linsolve(eqs, c)
            if len(solution) != 1:
                return None
            sol_tuple = list(solution)[0]
            c_vals = list(sol_tuple)
        except Exception:
            return None
        
        # Construct g with solved coefficients
        g_solved = sum(c_vals[k] * laguerres[k] for k in range(26))
        
        # Denominator: x * prod_i (x - z_i)^2
        denom = x
        for i in range(12):
            z_i = z_rational_sorted[i]
            denom *= (x - z_i)**2
        
        # q(x) = g(x) / denom
        # Perform exact polynomial division
        try:
            q = sympy.div(g_solved, denom, domain=sympy.QQ)
            # div returns (quotient, remainder)
            # For the division to be exact, remainder should be 0
            if q[1] != 0:
                return None
            q_poly = q[0]
        except Exception:
            return None
        
        # Find real roots of q(x)
        try:
            real_roots = sympy.real_roots(q_poly)
        except Exception:
            return None
        
        # Count roots with sign change in a tiny interval
        counted_roots = []
        for r in real_roots:
            # Check sign change at r - 1e-198 and r + 1e-198
            # Since we are using exact rationals, we can't use 1e-198 directly in sympy easily without converting.
            # The prompt says "sympy.real_roots" counts if q has strictly opposite signs at r - 1e-198 and r + 1e-198.
            # Let's use a very small rational epsilon.
            eps = sympy.Rational(1, 10**198)
            
            try:
                q_minus = q_poly.subs(x, r - eps)
                q_plus = q_poly.subs(x, r + eps)
                
                if q_minus == 0 or q_plus == 0:
                    continue
                
                if (q_minus > 0 and q_plus < 0) or (q_minus < 0 and q_plus > 0):
                    counted_roots.append(float(r))
            except Exception:
                continue
        
        if not counted_roots:
            B = 0.0
        else:
            B = max(0.0, max(counted_roots))
        
        score = B / (2 * math.pi)
        return score

    except Exception:
        return None


def construct():
    """
    Returns a list of 12 positive, distinct, finite double-root locations.
    Strategy: Log-spaced geometric progression z_i = z_1 * r^(i-1).
    We perform a coarse grid search over z_1 and r to minimize the score.
    Due to time constraints, we limit the number of evaluations.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds total budget
    
    best_score = float('inf')
    best_z = None
    
    # Grid search parameters
    # z_1: start from small to moderate
    # r: ratio between consecutive roots
    # We want roots to span several orders of magnitude.
    
    # Let's try a small grid first to see if it works and how long it takes.
    # If one evaluation takes ~10s, we can do ~30-60 evaluations.
    
    # z_1 values
    z1_vals = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    # r values
    r_vals = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    evaluations = 0
    
    for z1 in z1_vals:
        for r in r_vals:
            if time.time() > deadline - 120:  # Safety margin
                break
            
            # Generate roots
            z_list = [z1 * (r ** i) for i in range(12)]
            
            # Check validity: finite, > 0, pairwise distinct
            if any(not np.isfinite(z) or z <= 0 for z in z_list):
                continue
            if len(set([round(z, 10) for z in z_list])) < 12:
                continue
            
            # Evaluate
            score = _evaluate_score(z_list)
            evaluations += 1
            
            if score is not None:
                if score < best_score:
                    best_score = score
                    best_z = z_list[:]
            
            # If time is running out, break early
            if time.time() > deadline - 60:
                break
    
    # If no valid result was found, return a default
    if best_z is None:
        return [4.0 * (i + 1) for i in range(12)]
    
    return best_z

# EVOLVE-BLOCK-END
