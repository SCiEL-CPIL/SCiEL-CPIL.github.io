import time
import math
import numpy as np
import sympy as sp

def construct():
    """Return 12 double-root locations z_1..z_12."""
    
    # Helper to compute Laguerre polynomial L_n^(alpha)(x) exactly using sympy
    def laguerre_poly(n, alpha, x):
        return sp.laguerre(n, alpha, x)
    
    def _score_exact(z_list):
        try:
            z_arr = np.array(sorted(z_list), dtype=float)
            if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
                return float('inf')
            if len(np.unique(z_arr)) < len(z_arr):
                return float('inf')
            
            # Convert z_i to exact rationals
            # We use Rational(float(z_i)) as per task description
            z_sympy = [sp.Rational(float(zi)) for zi in z_arr]
            
            alpha = sp.Rational(-1, 2)
            x = sp.symbols('x')
            
            # Precompute Laguerre polynomials L_{2k}^{-1/2}(x) for k=0..25
            Ls = []
            for k in range(26):
                n = 2 * k
                Ls.append(sp.laguerre(n, alpha, x))
            
            # Setup linear system A c = b
            # Unknowns: c_0, ..., c_25
            c_syms = sp.symbols('c0:26')
            
            # g(x) = sum c_k L_{2k}(x)
            g_expr = sum(c_syms[k] * Ls[k] for k in range(26))
            
            # Derivative g'(x)
            g_diff = sp.diff(g_expr, x)
            
            # Conditions:
            # 1. g(0) = 0
            # 2. g'(0) = 1
            # 3. g(z_i) = 0 for i=0..11
            # 4. g'(z_i) = 0 for i=0..11
            
            A_mat = sp.zeros(26, 26)
            b_vec = sp.zeros(26, 1)
            
            # Row 0: g(0) = 0
            for k in range(26):
                A_mat[0, k] = Ls[k].subs(x, 0)
            b_vec[0, 0] = 0
            
            # Row 1: g'(0) = 1
            for k in range(26):
                A_mat[1, k] = sp.diff(Ls[k], x).subs(x, 0)
            b_vec[1, 0] = 1
            
            for i in range(12):
                zi = z_sympy[i]
                # Row 2+2i: g(zi) = 0
                for k in range(26):
                    A_mat[2+2*i, k] = Ls[k].subs(x, zi)
                b_vec[2+2*i, 0] = 0
                
                # Row 3+2i: g'(zi) = 0
                for k in range(26):
                    A_mat[3+2*i, k] = sp.diff(Ls[k], x).subs(x, zi)
                b_vec[3+2*i, 0] = 0
            
            # Solve A c = b
            try:
                c_sol = sp.LUsolve(A_mat, b_vec)
            except sp.MatrixNonInvertibleError:
                return float('inf')
            except Exception:
                return float('inf')
            
            # Construct g(x) with solved coefficients
            g_sol = sum(c_sol[k] * Ls[k] for k in range(26))
            
            # Construct D(x) = x * prod (x - z_i)^2
            D_expr = x
            for zi in z_sympy:
                D_expr *= (x - zi)**2
            
            # q(x) = g(x) / D(x)
            # Perform polynomial division
            q_poly, rem = sp.div(g_sol, D_expr, x)
            
            if rem != 0:
                # Check if remainder is negligible (should be 0 for exact solution)
                if sp.simplify(rem) != 0:
                    return float('inf')
            
            # Find real roots of q(x)
            # q_poly is a sympy expression
            # Use real_roots
            try:
                q_roots = sp.real_roots(q_poly, x)
            except Exception:
                return float('inf')
            
            max_r = 0.0
            eps = sp.Rational(1, 10)**198 # 1e-198
            
            for r in q_roots:
                # r is a sympy number (Rational or AlgebraicNumber)
                # Check sign change
                q_minus = q_poly.subs(x, r - eps)
                q_plus = q_poly.subs(x, r + eps)
                
                # Check if signs are opposite
                # q_minus * q_plus < 0
                prod = q_minus * q_plus
                if prod < 0:
                    # It's a simple root (sign change)
                    # Convert r to float for comparison
                    r_val = float(r)
                    if r_val > max_r:
                        max_r = r_val
            
            B = max(0, max_r)
            return B / (2 * math.pi)
            
        except Exception:
            return float('inf')

    # Candidate generation
    # We want to minimize the score.
    
    # 1. Bessel zeros squared (good heuristic for Laguerre-related problems)
    def bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    z_bessel = [bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # 2. Uniform spacing
    z_uniform = [i for i in range(1, 13)]
    
    # 3. Geometric spacing
    z_geometric = [0.5 * (1.5 ** i) for i in range(12)]
    
    # 4. Optimized Laguerre nodes (approximate roots of L_12^{-1/2}(x))
    # Use scipy to find roots of Laguerre polynomial
    try:
        from scipy.special import roots_genlaguerre
        # Roots of L_12^{-1/2}(x)
        roots_12 = roots_genlaguerre(12, -0.5)
        z_laguerre_12 = [float(r) for r in roots_12.roots]
        # Ensure positive and sorted
        z_laguerre_12 = sorted([z for z in z_laguerre_12 if z > 0])
        if len(z_laguerre_12) < 12:
            # Pad with additional points if needed
            z_laguerre_12 = z_laguerre_12 + [max(z_laguerre_12) * 1.1 ** i for i in range(12 - len(z_laguerre_12))]
        z_laguerre_12 = z_laguerre_12[:12]
    except Exception:
        z_laguerre_12 = z_uniform
    
    # 5. Mixed: some Bessel, some uniform
    z_mixed = [bessel_zero_J0(i)**2 for i in range(1, 7)] + [i + 6 for i in range(6)]
    
    candidates = [
        z_bessel,
        z_uniform,
        z_geometric,
        z_laguerre_12,
        z_mixed
    ]
    
    best_z = None
    best_score = float('inf')
    
    deadline = time.time() + 580.0
    
    for z_cand in candidates:
        if time.time() > deadline:
            break
        score = _score_exact(z_cand)
        if score < best_score:
            best_score = score
            best_z = z_cand
    
    # If we have time, try a simple local search on the best candidate
    if best_z is not None and time.time() < deadline - 60:
        current_z = list(best_z)
        current_score = best_score
        np.random.seed(42)
        
        # Phase 1: Coarse local search
        for step in range(30):
            if time.time() > deadline - 30:
                break
            
            candidate = list(current_z)
            # Perturb one random index
            idx = np.random.randint(0, 12)
            factor = np.random.uniform(0.85, 1.15)
            candidate[idx] *= factor
            
            if any(z <= 0 for z in candidate):
                continue
                
            c_sorted = sorted(candidate)
            distinct = True
            for j in range(11):
                if abs(c_sorted[j] - c_sorted[j+1]) < 1e-6:
                    distinct = False
                    break
            if not distinct:
                continue
            
            score = _score_exact(candidate)
            if score < current_score:
                current_score = score
                current_z = candidate
                best_z = candidate
                best_score = score
        
        # Phase 2: Fine polish if we have time
        if time.time() < deadline - 30:
            for step in range(20):
                if time.time() > deadline - 10:
                    break
                
                candidate = list(current_z)
                # Perturb one random index with smaller factor
                idx = np.random.randint(0, 12)
                factor = np.random.uniform(0.97, 1.03)
                candidate[idx] *= factor
                
                if any(z <= 0 for z in candidate):
                    continue
                    
                c_sorted = sorted(candidate)
                distinct = True
                for j in range(11):
                    if abs(c_sorted[j] - c_sorted[j+1]) < 1e-6:
                        distinct = False
                        break
                if not distinct:
                    continue
                
                score = _score_exact(candidate)
                if score < current_score:
                    current_score = score
                    current_z = candidate
                    best_z = candidate
                    best_score = score
    
    if best_z is None:
        # Fallback
        return z_bessel
        
    return best_z
