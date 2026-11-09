import sympy as sp
import numpy as np
from scipy.special import roots_genlaguerre
import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 540  # Leave 60s margin for 600s limit
    
    # Candidate 1: Roots of L_12^(-1/2)(x)
    # These are positive and distinct.
    try:
        roots_lag = roots_genlaguerre(12, -0.5)
        z_candidates_1 = [float(r) for r in roots_lag]
        # Ensure they are sorted and positive
        z_candidates_1 = sorted([z for z in z_candidates_1 if z > 0])
        if len(z_candidates_1) != 12:
            z_candidates_1 = [float(i) for i in range(1, 13)]
    except Exception:
        z_candidates_1 = [float(i) for i in range(1, 13)]

    # Candidate 2: Uniform 1..12
    z_candidates_2 = [float(i) for i in range(1, 13)]
    
    # Candidate 3: Uniform 0.5..6
    z_candidates_3 = [0.5 * i for i in range(1, 13)]
    
    # Candidate 4: Uniform 0.1..1.2
    z_candidates_4 = [0.1 * i for i in range(1, 13)]

    candidates = [z_candidates_1, z_candidates_2, z_candidates_3, z_candidates_4]
    
    best_z = None
    best_score = float('inf')
    
    # Pre-compute symbols and polynomials if possible? 
    # Sympy polynomials depend on z, so we must do it inside.
    
    for z_list in candidates:
        if time.time() > deadline:
            break
            
        # Check validity: finite, > 0, distinct
        if not all(math.isfinite(z) for z in z_list):
            continue
        if not all(z > 0 for z in z_list):
            continue
        if len(set(z_list)) != len(z_list):
            continue
            
        try:
            score = evaluate(z_list, start_time, deadline)
            if score < best_score:
                best_score = score
                best_z = z_list
        except Exception:
            pass

    if best_z is None:
        # Fallback to seed-like behavior if all failed
        return [4.0 * (i + 1) for i in range(12)]
        
    return best_z

def evaluate(z_list, start_time, deadline):
    """
    Evaluates the score for a given list of z_i.
    Returns B / (2*pi) or inf if invalid/too slow.
    """
    # Check time again
    if time.time() > deadline:
        return float('inf')
        
    x = sp.symbols('x')
    alpha = sp.Rational(-1, 2)
    
    # Construct Laguerre polynomials L_0, L_2, ..., L_50
    # L_k^(alpha)(x)
    L_pols = []
    for k in range(26):
        deg = 2 * k
        L_k = sp.genlaguerre(deg, alpha, x)
        L_pols.append(L_k)
        
    # Variables c_0 ... c_25
    c = sp.symbols('c0:26')
    
    # g(x) = sum c_k L_2k(x)
    g = sum(c[i] * L_pols[i] for i in range(26))
    
    # Equations:
    # 1. g(0) = 0
    # 2. g'(0) = 1
    # 3. g(z_i) = 0 for i in 1..12
    # 4. g'(z_i) = 0 for i in 1..12
    
    eqs = []
    eqs.append(g.subs(x, 0))
    eqs.append(sp.diff(g, x).subs(x, 0) - 1)
    
    for z in z_list:
        z_r = sp.Rational(float(z))
        eqs.append(g.subs(x, z_r))
        eqs.append(sp.diff(g, x).subs(x, z_r))
        
    # Solve linear system
    # Matrix A and vector b
    # A is 26x26
    # We can extract the matrix directly for efficiency
    # A[j, i] is the coefficient of c_i in eq_j
    
    # Instead of solving symbols, let's build the matrix explicitly
    # eqs are linear in c
    # Let's use sp.LUsolve
    
    try:
        # Convert eqs to linear form
        # sp.LinAlgError if singular
        sol = sp.LUsolve(sp.Matrix([[sp.Poly(eq, c[i]).coeff_monomial(c[i]) for i in range(26)] for eq in eqs]), 
                         sp.Matrix([-sp.Poly(eq, c[i]).coeff_monomial(1) for i in range(26)])) 
        # Wait, the equations are sum A_ij c_j = b_i.
        # My eqs are sum A_ij c_j + b_i = 0. So sum A_ij c_j = -b_i.
        # sp.LUsolve(A, b) solves A x = b.
        
        # Let's re-construct A and b properly
        A = sp.zeros(26, 26)
        b = sp.zeros(26, 1)
        
        for i, eq in enumerate(eqs):
            poly_eq = sp.Poly(eq, *c)
            for j in range(26):
                A[i, j] = poly_eq.coeff_monomial(c[j])
            b[i, 0] = -poly_eq.coeff_monomial(1)
            
        if A.det() == 0:
            return float('inf')
            
        sol = sp.LUsolve(A, b)
        
        # Substitute c into g
        g_sol = g.subs(dict(zip(c, sol)))
        g_sol = sp.simplify(g_sol)
        
        # Construct denominator D(x) = x * prod (x - z_i)^2
        D = x
        for z in z_list:
            z_r = sp.Rational(float(z))
            D = D * (x - z_r)**2
            
        # q(x) = g(x) / D(x)
        # Exact polynomial division
        q = sp.Poly(g_sol, x) / sp.Poly(D, x)
        
        if not q.is_exact:
            return float('inf')
            
        q_expr = q.as_expr()
        
        # Find real roots of q
        roots = sp.real_roots(q_expr)
        
        if not roots:
            B = 0.0
        else:
            # Check sign change at r +/- 1e-198
            # 1e-198 is not a rational, but we can use sp.Rational(1, 10**198)
            eps = sp.Rational(1, 10**198)
            max_r = 0
            for r in roots:
                # r is a sympy expression (RootOf or Rational)
                # Evaluate q at r - eps and r + eps
                val_minus = q_expr.subs(x, r - eps)
                val_plus = q_expr.subs(x, r + eps)
                
                # Check if strictly opposite signs
                # val_minus and val_plus are rationals or expressions
                # If they are numbers, we can check sign
                # If they are complex, real_roots returns real roots, so subs should be real?
                # Not necessarily, if r is a root, q(r)=0.
                # We need the sign of the derivative or just evaluate nearby.
                
                # Since q is a polynomial, and r is a root, the sign change indicates an odd multiplicity.
                # The task says: "counts if q has strictly opposite signs at r - 1e-198 and r + 1e-198"
                
                try:
                    vm = float(val_minus)
                    vp = float(val_plus)
                    if vm * vp < 0:
                        # It counts
                        if float(r) > max_r:
                            max_r = float(r)
                except (TypeError, ValueError):
                    # If conversion fails, try to determine sign symbolically?
                    # Or assume it doesn't count if we can't verify?
                    # Given the strictness, maybe skip if not floatable.
                    pass
            
            B = max(0, max_r)
            
        score = B / (2 * math.pi)
        return score
        
    except Exception:
        return float('inf')

# EVOLVE-BLOCK-END
