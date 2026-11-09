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
    deadline = start_time + 570.0  # Leave 60s margin
    
    candidates = []
    
    # Seed
    candidates.append([4.0 * (i + 1) for i in range(12)])
    
    # Small values - hypothesis: small z_i may push remaining roots lower
    candidates.append([0.5 * (i + 1) for i in range(12)])
    candidates.append([0.1 * (i + 1) for i in range(12)])
    candidates.append([0.01 * (i + 1) for i in range(12)])
    
    # Medium values
    candidates.append([1.0 * (i + 1) for i in range(12)])
    candidates.append([2.0 * (i + 1) for i in range(12)])
    
    # Large values - hypothesis: large z_i may absorb large roots
    candidates.append([10.0 * (i + 1) for i in range(12)])
    candidates.append([50.0 * (i + 1) for i in range(12)])
    candidates.append([100.0 * (i + 1) for i in range(12)])
    
    # Logarithmic spacing
    candidates.append([np.exp(i * 0.3) for i in range(12)])
    candidates.append([np.exp(i * 0.5) for i in range(12)])
    candidates.append([np.exp(i * 0.8) for i in range(12)])
    
    # Power law spacing
    candidates.append([(i + 1)**1.5 for i in range(12)])
    candidates.append([(i + 1)**2.0 for i in range(12)])
    
    # Random with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        np.random.seed(seed)
        candidates.append(list(np.random.uniform(0.1, 30.0, 12)))
    
    # Geometric progression
    for ratio in [1.5, 2.0, 2.5, 3.0]:
        candidates.append([0.5 * ratio**i for i in range(12)])
    
    # Very small cluster
    candidates.append([0.01 + 0.001 * i for i in range(12)])
    
    # Mix of small and large
    candidates.append([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    candidates.append([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4])
    
    best_B = float('inf')
    best_z = None
    
    for z in candidates:
        if time.time() > deadline:
            break
            
        # Check if z is valid (positive, distinct)
        z_sorted = sorted(z)
        if len(set(z_sorted)) != 12:
            continue
        if any(z_i <= 0 for z_i in z_sorted):
            continue
        # Check pairwise distinct with tolerance
        valid = True
        for i in range(len(z_sorted) - 1):
            if abs(z_sorted[i+1] - z_sorted[i]) < 1e-10:
                valid = False
                break
        if not valid:
            continue
            
        B = evaluate_candidate(z)
        
        if B is not None and B < best_B:
            best_B = B
            best_z = z
            
    if best_z is None:
        # Fallback to seed if all failed
        return [4.0 * (i + 1) for i in range(12)]
        
    return best_z
