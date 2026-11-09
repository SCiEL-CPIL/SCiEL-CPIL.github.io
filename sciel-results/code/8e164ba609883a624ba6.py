import sympy as sp
import numpy as np
import time
import math

def evaluate_candidate(z_list):
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
        Ls = []
        for k in range(26):
            Ls.append(sp.Laguerre(2*k, alpha, x))
        
        # Build matrix M and rhs directly
        M = sp.zeros(26, 26)
        rhs = sp.zeros(26, 1)
        
        # Eq 0: g(0) = 0
        for k in range(26):
            M[0, k] = Ls[k].subs(x, 0)
        rhs[0] = 0
        
        # Eq 1: g'(0) = 1
        for k in range(26):
            M[1, k] = sp.diff(Ls[k], x).subs(x, 0)
        rhs[1] = 1
        
        idx = 2
        for z in z_rat:
            for k in range(26):
                M[idx, k] = Ls[k].subs(x, z)
            rhs[idx] = 0
            idx += 1
            
            for k in range(26):
                M[idx, k] = sp.diff(Ls[k], x).subs(x, z)
            rhs[idx] = 0
            idx += 1
        
        # Solve
        try:
            sol = M.LUsolve(rhs)
        except sp.matrices.exceptions.SingularMatrixException:
            return None
        except Exception:
            return None
        
        # Build g_sol
        cs = sp.symbols('c0:26')
        g = sum(cs[k] * Ls[k] for k in range(26))
        g_sol = g.subs(list(zip(cs, sol)))
        
        # q(x) = g(x) / (x * prod (x - z_i)^2)
        denom = x
        for z in z_rat:
            denom *= (x - z)**2
            
        q = sp.div(g_sol, denom)
        q_poly = q[0]
        
        # Find real roots of q_poly
        try:
            roots = sp.real_roots(q_poly)
        except Exception:
            return None
            
        if not roots:
            return 0.0
            
        max_r = 0.0
        for r in roots:
            try:
                r_float = float(r)
            except:
                continue
            if not np.isfinite(r_float) or r_float <= 0:
                continue
            
            delta = 1e-198
            try:
                val_minus = float(q_poly.subs(x, r_float - delta))
                val_plus = float(q_poly.subs(x, r_float + delta))
            except:
                continue
            
            if val_minus * val_plus < 0:
                if r_float > max_r:
                    max_r = r_float
                    
        return float(max_r)
        
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 550.0  # Leave 80s margin for safety and final check
    
    best_B = float('inf')
    best_z = None
    
    # Strategy: Use Bessel J_{1/2} zeros as primary initialization.
    # J_{1/2}(x) = sqrt(2/(pi x)) * sin(x). Zeros are at x = n*pi.
    # The argument of Laguerre L_n^{alpha}(x) relates to Bessel argument via sqrt(x).
    # So if we set z_i such that sqrt(z_i) = i * pi, then z_i = i^2 * pi^2.
    
    # Candidate 1: Exact Bessel zeros
    pi_sq = math.pi ** 2
    z_bessel = [pi_sq * (i + 1)**2 for i in range(12)]
    
    # Candidate 2: Slightly scaled Bessel zeros (in case of finite-n effects)
    # Try scaling factors close to 1
    scales = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3]
    
    candidates = []
    candidates.append(z_bessel)
    
    for s in scales:
        if time.time() > deadline:
            break
        z_scaled = [s * pi_sq * (i + 1)**2 for i in range(12)]
        candidates.append(z_scaled)
        
    # Candidate 3: If Bessel fails or is slow, fallback to a known good structure?
    # The seed used 4*(i+1). Let's include it as a last resort if Bessel is invalid.
    candidates.append([4.0 * (i + 1) for i in range(12)])
    
    for z in candidates:
        if time.time() > deadline:
            break
            
        # Validate
        z_sorted = sorted(z)
        if len(set(z_sorted)) != 12:
            continue
        if any(z_i <= 0 for z_i in z_sorted):
            continue
        valid = True
        for i in range(len(z_sorted) - 1):
            if abs(z_sorted[i+1] - z_sorted[i]) < 1e-10:
                valid = False
                break
        if not valid:
            continue
            
        # Evaluate
        try:
            B = evaluate_candidate(z)
        except Exception:
            B = None
            
        if B is not None and B < best_B:
            best_B = B
            best_z = z
            
            # If we find a very good B, we might stop early to save time?
            # But B=0 is the theoretical minimum. If B is small, it's good.
            # However, we should try a couple more to be sure we aren't stuck in a local "bad" valid point.
            # Given the cost, we will just continue until time runs out or we've tried all candidates.
            
    if best_z is None:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]
        
    return best_z
