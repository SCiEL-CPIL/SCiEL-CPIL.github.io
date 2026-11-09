import sympy as sp
import numpy as np
import time
import random
import math

def evaluate_candidate(z_list):
    """
    Evaluates the candidate z_list exactly.
    Returns B if valid, None if invalid.
    """
    try:
        # Convert to exact rationals
        # Sort for consistency? The problem says "the list is sorted" in the task description
        # "Each z_i is converted exactly to a rational... and the list is sorted."
        # This implies the evaluator sorts them or expects them sorted?
        # "returning 12 double-root locations z_1..z_12 (list or array of floats). They must be finite, > 0 and pairwise distinct... Each z_i is converted exactly to a rational... and the list is sorted."
        # It seems the *evaluation* sorts them. Or the *return* should be sorted?
        # The task says: "Write construct(), returning 12 double-root locations... They must be... distinct... Each z_i is converted... and the list is sorted."
        # This phrasing "the list is sorted" likely refers to the internal processing or the specific order in which the evaluator treats them.
        # However, since the system of equations is symmetric with respect to the set {z_i} (the product is the same, the sum of conditions is the same), the order shouldn't matter for the resulting g(x) and q(x).
        # Let's sort them to be safe and consistent with "the list is sorted".
        z_list_sorted = sorted(z_list)
        z_rat = [sp.Rational(float(z)) for z in z_list_sorted]
        
        # Define x and Laguerre polynomials
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)
        
        # Precompute L_2k(x) for k=0..25
        Ls = []
        for k in range(26):
            Ls.append(sp.Laguerre(2*k, alpha, x))
            
        # g(x) = sum c_k L_2k(x)
        cs = sp.symbols('c0:26')
        
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
        g_sol = sum(sol[k] * Ls[k] for k in range(26))
        
        # q(x) = g(x) / (x * prod (x - z_i)^2)
        denom = x
        for z in z_rat:
            denom *= (x - z)**2
            
        # Polynomial division
        q = sp.div(g_sol, denom)
        # The task says "exact polynomial division". If remainder is not 0, is it invalid?
        # "A singular system is invalid." It doesn't explicitly say non-zero remainder is invalid.
        # But q is defined as the quotient.
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
            # r is a sympy expression
            try:
                r_float = float(r)
            except:
                continue
                
            if not np.isfinite(r_float) or r_float <= 0:
                continue
                
            # Check sign change at r +/- 1e-198
            # "counts if q has strictly opposite signs at r - 1e-198 and r + 1e-198"
            # We evaluate the *polynomial* q_poly at these points.
            # Note: r is a root of q_poly, so q_poly(r) = 0.
            # For a simple root, the sign changes. For a double root, it doesn't.
            # The task asks to check the signs.
            
            delta = 1e-198
            # Using float evaluation for speed, as exact rational eval with 1e-198 offset is tricky/slow
            # The task says "All arithmetic is exact rational" for the *evaluation* of the system,
            # but the sign check uses 1e-198 which is a float.
            # "A real root r of q ... counts if q has strictly opposite signs at r - 1e-198 and r + 1e-198."
            # This implies floating point evaluation of q at these points.
            
            try:
                val_minus = float(q_poly.subs(x, r_float - delta))
                val_plus = float(q_poly.subs(x, r_float + delta))
            except Exception:
                continue
                
            if val_minus * val_plus < 0:
                if r_float > max_r:
                    max_r = r_float
                    
        return float(max_r)
        
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 570.0  # Leave 60s margin for the evaluator's own overhead/kill signal
    
    candidates = []
    
    # 1. Seed (baseline)
    candidates.append([4.0 * (i + 1) for i in range(12)])
    
    # 2. Hypothesis: Small z_i might push the "residual" roots (roots of q) to smaller values
    #    The Laguerre polynomials L_n^(-1/2) have roots roughly in [0, 2n+1].
    #    If we place z_i very small, we are "using up" the low-degree capacity of g to cancel roots near 0.
    #    This might force the remaining roots of g (which become roots of q) to be... where?
    #    Let's test very small z_i.
    candidates.append([0.1 * (i + 1) for i in range(12)])
    candidates.append([0.01 * (i + 1) for i in range(12)])
    candidates.append([0.5 * (i + 1) for i in range(12)])
    
    # 3. Hypothesis: Large z_i might "absorb" large roots of g, leaving small roots for q?
    #    Or maybe it pushes them further out?
    #    Let's test large z_i.
    candidates.append([10.0 * (i + 1) for i in range(12)])
    candidates.append([50.0 * (i + 1) for i in range(12)])
    candidates.append([100.0 * (i + 1) for i in range(12)])
    
    # 4. Geometric spacing (covers a wide range)
    for base in [0.5, 1.0, 2.0]:
        for ratio in [1.5, 2.0, 3.0]:
            candidates.append([base * (ratio ** i) for i in range(12)])
            
    # 5. Power law
    for p in [1.5, 2.0, 3.0]:
        candidates.append([(i + 1) ** p for i in range(12)])
        
    # 6. Roots of Laguerre L_12^(-1/2)(x)
    #    These are natural "nodes" for the basis.
    try:
        x = sp.Symbol('x')
        L12 = sp.Laguerre(12, sp.Rational(-1, 2), x)
        roots_L12 = sp.nroots(L12, n=50, maxsteps=100)
        real_roots = [float(r) for r in roots_L12 if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0]
        if len(real_roots) == 12:
            candidates.append(real_roots)
    except:
        pass
        
    # 7. Random search with various ranges
    random.seed(42)
    for _ in range(10):
        # Mix of small and large
        z = [random.uniform(0.01, 50.0) for _ in range(12)]
        # Ensure distinct and positive
        z = sorted(z)
        # Check distinctness
        valid = True
        for i in range(11):
            if abs(z[i+1] - z[i]) < 1e-6:
                valid = False
                break
        if valid:
            candidates.append(z)
            
    # 8. Optimized search: Local refinement
    #    We can't do gradient descent easily with exact arithmetic.
    #    But we can do a coarse-to-fine grid search on the parameters if we parameterize them.
    #    However, evaluating one candidate takes seconds.
    #    Let's stick to a diverse set of hand-picked candidates first.
    
    # 9. Specific patterns
    #    Clustered near 1
    candidates.append([1.0 + 0.1 * i for i in range(12)])
    #    Clustered near 10
    candidates.append([10.0 + 1.0 * i for i in range(12)])
    
    best_B = float('inf')
    best_z = None
    
    # Sort candidates by a heuristic?
    # Maybe candidates with z_i closer to the expected root distribution of L_12 are better?
    # We'll just iterate.
    
    for z in candidates:
        if time.time() > deadline:
            break
            
        # Validate z
        if len(z) != 12:
            continue
        if any(z_i <= 0 for z_i in z):
            continue
        if len(set(z)) != 12:
            continue
        # Check distinctness with tolerance
        z_sorted = sorted(z)
        valid = True
        for i in range(11):
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
