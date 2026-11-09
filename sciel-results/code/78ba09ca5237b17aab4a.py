"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sympy as sp
import numpy as np
import time

def _evaluate_candidate(z_list):
    """
    Evaluates the candidate z_list exactly.
    Returns B if valid, None if invalid.
    """
    try:
        # Convert to exact rationals
        z_rat = [sp.Rational(float(z)) for z in z_list]
        
        # Define x and Laguerre polynomials
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)
        
        # Precompute L_2k(x) for k=0..25
        # L_n^(alpha)(x)
        Ls = []
        Ls_derivs = []
        for k in range(26):
            L = sp.Laguerre(2*k, alpha, x)
            Ls.append(L)
            Ls_derivs.append(sp.diff(L, x))
            
        # g(x) = sum c_k L_2k(x)
        cs = sp.symbols('c0:26')
        
        # Construct Matrix M and rhs vector b for M c = b
        # M is 26x26
        M = sp.zeros(26, 26)
        b = sp.zeros(26, 1)
        
        # Eq 0: g(0) = 0
        # sum c_k L_2k(0) = 0
        for k in range(26):
            M[0, k] = Ls[k].subs(x, 0)
        b[0] = 0
        
        # Eq 1: g'(0) = 1
        # sum c_k L'_2k(0) = 1
        for k in range(26):
            M[1, k] = Ls_derivs[k].subs(x, 0)
        b[1] = 1
        
        idx = 2
        for z in z_rat:
            # g(z) = 0
            for k in range(26):
                M[idx, k] = Ls[k].subs(x, z)
            b[idx] = 0
            idx += 1
            
            # g'(z) = 0
            for k in range(26):
                M[idx, k] = Ls_derivs[k].subs(x, z)
            b[idx] = 0
            idx += 1
            
        # Solve
        try:
            sol = M.LUsolve(b)
        except sp.matrices.exceptions.SingularMatrixException:
            return None
        except Exception:
            return None
            
        # Substitute cs back into g
        # g(x) = sum sol[k] * Ls[k]
        g_sol = sum(sol[k] * Ls[k] for k in range(26))
        
        # q(x) = g(x) / (x * prod (x - z_i)^2)
        denom = x
        for z in z_rat:
            denom *= (x - z)**2
            
        # Polynomial division
        # g_sol is a polynomial of degree 50.
        # denom is a polynomial of degree 1 + 12*2 = 25.
        # q should be a polynomial of degree 25.
        q_poly, rem = sp.div(g_sol, denom)
        
        # The problem states "exact polynomial division", implying rem should be 0.
        # However, due to floating point conversion of z_i to rational, or numerical issues?
        # No, all arithmetic is exact rational.
        # If the system is consistent, rem should be 0.
        # If rem != 0, is it invalid? The prompt says "A singular system is invalid".
        # It doesn't explicitly say non-zero remainder is invalid, but "Let q(x) = ... an exact polynomial division"
        # suggests q is the quotient.
        # Let's assume q_poly is the polynomial to check.
        
        # Find real roots of q_poly
        # sympy.real_roots
        try:
            roots = sp.real_roots(q_poly)
        except Exception:
            return None
            
        if not roots:
            return 0.0
            
        max_r = 0.0
        for r in roots:
            # r is a sympy expression
            # Check if it's a real number (it should be from real_roots)
            # and > 0
            try:
                r_float = float(r)
            except:
                continue
                
            if not np.isfinite(r_float) or r_float <= 0:
                continue
                
            # Sign change check
            delta = 1e-198
            # Evaluate q_poly at r_float - delta and r_float + delta
            # Note: q_poly is a sympy expression.
            # Substituting a float makes it a float expression?
            # Better to substitute Rational?
            # The prompt says: "counts if q has strictly opposite signs at r - 1e-198 and r + 1e-198".
            # r is the root. 1e-198 is a float.
            # If r is rational, r - 1e-198 is a float.
            # So we evaluate q at these float points.
            
            val_minus = q_poly.subs(x, r_float - delta)
            val_plus = q_poly.subs(x, r_float + delta)
            
            try:
                v_minus = float(val_minus)
                v_plus = float(val_plus)
            except:
                continue
                
            if v_minus * v_plus < 0:
                if r_float > max_r:
                    max_r = r_float
                    
        return max_r
        
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 580.0 # Leave 50s margin
    
    best_B = float('inf')
    best_z = [4.0 * (i + 1) for i in range(12)] # Default to seed
    
    # Strategy:
    # 1. Evaluate the seed to get a baseline.
    # 2. Try some heuristic candidates that might lower B.
    #    B is the largest real root of q(x).
    #    q(x) is degree 25.
    #    The roots of q are the "extra" roots of g(x) besides 0 and z_i (double).
    #    To minimize the largest root, we might want to "push" the roots of q to the left.
    #    This might be achieved by clustering z_i or choosing them specifically.
    
    candidates = []
    
    # Seed
    candidates.append([4.0 * (i + 1) for i in range(12)])
    
    # Try smaller z_i
    candidates.append([1.0 * (i + 1) for i in range(12)])
    
    # Try larger z_i
    candidates.append([10.0 * (i + 1) for i in range(12)])
    
    # Try clustered z_i
    candidates.append([0.5 * (i + 1) for i in range(12)])
    
    # Try logarithmic spacing
    candidates.append([np.exp(0.5 * i) for i in range(12)])
    
    # Try roots of a specific polynomial?
    # Maybe roots of L_12(-1/2)?
    try:
        x = sp.Symbol('x')
        L12 = sp.Laguerre(12, sp.Rational(-1, 2), x)
        roots_L12 = sp.nroots(L12, n=50, maxsteps=100)
        real_roots = [float(r) for r in roots_L12 if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0]
        if len(real_roots) == 12:
            candidates.append(real_roots)
    except:
        pass

    # Try random candidates
    # We have 580s. Each eval takes "seconds to about a minute".
    # Let's say 30s per eval. We can do ~15 evals.
    # Let's prioritize the most promising ones first.
    
    # Shuffle candidates to ensure we don't get stuck if the first few are bad?
    # No, seed is first.
    
    for z_cand in candidates:
        if time.time() > deadline:
            break
            
        # Check validity of z_cand
        # Must be finite, > 0, pairwise distinct
        if len(z_cand) != 12:
            continue
        if any(not np.isfinite(z) or z <= 0 for z in z_cand):
            continue
        if len(set(z_cand)) != 12:
            continue
            
        B = _evaluate_candidate(z_cand)
        
        if B is not None:
            if B < best_B:
                best_B = B
                best_z = z_cand
    
    # If we have time, do a local search?
    # The landscape is likely complex.
    # Let's stick to the evaluated candidates for now to ensure we return something.
    
    return best_z

# EVOLVE-BLOCK-END
