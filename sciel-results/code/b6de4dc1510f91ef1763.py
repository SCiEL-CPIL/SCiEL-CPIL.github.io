import sympy as sp
import numpy as np
import time

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
        
        # g(x) = sum c_k L_2k(x)
        cs = sp.symbols('c0:26')
        g = sum(cs[k] * Ls[k] for k in range(26))
        
        # Build the 26x26 linear system
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
            # g(z) = 0
            for k in range(26):
                M[idx, k] = Ls[k].subs(x, z)
            rhs[idx] = 0
            idx += 1
            
            # g'(z) = 0
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
            
        # Substitute cs back into g
        g_sol = g.subs(list(zip(cs, sol)))
        
        # q(x) = g(x) / (x * prod (x - z_i)^2)
        denom = x
        for z in z_rat:
            denom *= (x - z)**2
            
        # Polynomial division
        q = sp.div(g_sol, denom)
        q_poly = q[0]
        
        # Find real roots of q_poly
        try:
            roots = sp.real_roots(q_poly)
        except Exception:
            return None
            
        if not roots:
            B = 0.0
        else:
            max_r = 0.0
            for r in roots:
                r_float = float(r)
                if not np.isfinite(r_float) or r_float <= 0:
                    continue
                
                delta = 1e-198
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
                        
            B = max_r
            
        return float(B)
        
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_B = float('inf')
    best_z = None
    
    # Precompute Laguerre polynomials for potential reuse
    # But each evaluation is independent, so we just try different z sets
    
    # Strategy: Try various geometric progressions and other patterns
    # to find z configurations that minimize B
    
    candidate_sets = []
    
    # 1. Seed
    candidate_sets.append([4.0 * (i + 1) for i in range(12)])
    
    # 2. Smaller arithmetic
    candidate_sets.append([1.0 * (i + 1) for i in range(12)])
    candidate_sets.append([0.5 * (i + 1) for i in range(12)])
    candidate_sets.append([0.25 * (i + 1) for i in range(12)])
    
    # 3. Geometric progressions starting small
    for start in [0.1, 0.2, 0.5, 1.0, 2.0]:
        for ratio in [1.2, 1.5, 1.8, 2.0, 2.5]:
            zs = [start * (ratio ** i) for i in range(12)]
            if all(z > 0 for z in zs) and len(set(zs)) == 12:
                candidate_sets.append(zs)
    
    # 4. Roots of Laguerre L_12^(alpha)
    try:
        x = sp.Symbol('x')
        L12 = sp.Laguerre(12, sp.Rational(-1, 2), x)
        roots_L12 = sp.nroots(L12, n=50, maxsteps=100)
        real_roots = [float(r) for r in roots_L12 if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0]
        if len(real_roots) == 12:
            candidate_sets.append(sorted(real_roots))
    except:
        pass
    
    # 5. Roots of L_11, L_13, etc. for nearby degrees
    for deg in [10, 11, 13, 14, 15]:
        try:
            x = sp.Symbol('x')
            Ld = sp.Laguerre(deg, sp.Rational(-1, 2), x)
            roots_Ld = sp.nroots(Ld, n=50, maxsteps=100)
            real_roots = [float(r) for r in roots_Ld if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0]
            if len(real_roots) == 12:
                candidate_sets.append(sorted(real_roots))
            elif len(real_roots) >= 12:
                # Take the 12 smallest positive roots
                candidate_sets.append(sorted(real_roots[:12]))
        except:
            pass
    
    # 6. Clustered near specific values
    for center in [1.0, 2.0, 5.0, 10.0]:
        for spread in [0.1, 0.5, 1.0, 2.0]:
            zs = [center + spread * (i - 5.5) for i in range(12)]
            zs = [max(0.01, z) for z in zs]  # Ensure positive
            if len(set(zs)) == 12:
                candidate_sets.append(sorted(zs))
    
    # 7. Try to place z_i to "cancel" large roots - use values that might be roots of q
    # Heuristic: try z_i near 0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
    candidate_sets.append([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 15.0, 20.0])
    candidate_sets.append([0.3, 0.6, 1.0, 1.5, 2.2, 3.0, 4.2, 5.8, 8.0, 11.0, 15.0, 20.0])
    
    # Evaluate candidates
    for z in candidate_sets:
        if time.time() > deadline:
            break
        
        # Check validity
        if len(set(z)) != 12:
            continue
        if any(z_i <= 0 for z_i in z):
            continue
        
        B = evaluate_candidate(z)
        
        if B is not None and B < best_B:
            best_B = B
            best_z = z
            # If we find a very good B, we can be more aggressive
            if best_B < 5.0:
                break
    
    if best_z is None:
        return [4.0 * (i + 1) for i in range(12)]
        
    return best_z
