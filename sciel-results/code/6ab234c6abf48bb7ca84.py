import time
import math
import numpy as np
import sympy as sp

def construct():
    """Return 12 double-root locations z_1..z_12."""
    
    def _score_exact(z_list):
        """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
        try:
            z_arr = sorted(z_list)
            if any(z <= 0 or not math.isfinite(z) for z in z_arr):
                return float('inf')
            # Check pairwise distinct
            for i in range(11):
                if abs(z_arr[i] - z_arr[i+1]) < 1e-12:
                    return float('inf')
            
            # Convert z_i to exact rationals
            z_sympy = [sp.Rational(float(zi)) for zi in z_arr]
            
            alpha = sp.Rational(-1, 2)
            x = sp.symbols('x')
            
            # Precompute Laguerre polynomials L_{2k}^{-1/2}(x) for k=0..25
            Ls = []
            for k in range(26):
                n = 2 * k
                Ls.append(sp.laguerre(n, alpha, x))
            
            # Setup linear system A c = b
            c_syms = sp.symbols('c0:26')
            g_expr = sum(c_syms[k] * Ls[k] for k in range(26))
            g_diff = sp.diff(g_expr, x)
            
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
            q_poly, rem = sp.div(g_sol, D_expr, x)
            
            if rem != 0:
                if sp.simplify(rem) != 0:
                    return float('inf')
            
            # Find real roots of q(x)
            try:
                q_roots = sp.real_roots(q_poly, x)
            except Exception:
                return float('inf')
            
            max_r = 0.0
            eps = sp.Rational(1, 10)**198
            
            for r in q_roots:
                q_minus = q_poly.subs(x, r - eps)
                q_plus = q_poly.subs(x, r + eps)
                prod = q_minus * q_plus
                if prod < 0:
                    r_val = float(r)
                    if r_val > max_r:
                        max_r = r_val
            
            B = max(0.0, max_r)
            return B / (2 * math.pi)
            
        except Exception:
            return float('inf')

    def _bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

    # Generate multiple candidate sets
    candidates = []
    
    # 1. Bessel zeros squared
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    candidates.append(z_bessel)
    
    # 2. Laguerre nodes (roots of L_12^{-1/2})
    try:
        from scipy.special import roots_genlaguerre
        roots_12 = roots_genlaguerre(12, -0.5)
        z_lag = sorted([float(r) for r in roots_12.roots if float(r) > 0])
        if len(z_lag) == 12:
            candidates.append(z_lag)
        else:
            # Pad if needed
            while len(z_lag) < 12:
                z_lag.append(max(z_lag) * 1.1)
            candidates.append(z_lag[:12])
    except Exception:
        pass
    
    # 3. Uniform spacing
    z_uniform = [float(i) for i in range(1, 13)]
    candidates.append(z_uniform)
    
    # 4. Geometric spacing
    z_geo = [0.5 * (1.5 ** i) for i in range(12)]
    candidates.append(z_geo)
    
    # 5. Power-law spacing (tunable)
    for p in [1.5, 2.0, 2.5, 3.0]:
        z_pow = [0.3 * (i+1)**p for i in range(12)]
        candidates.append(z_pow)
    
    # 6. Mix: first 6 Bessel, last 6 uniform
    z_mixed = [_bessel_zero_J0(i)**2 for i in range(1, 7)] + [float(i+6) for i in range(6)]
    candidates.append(z_mixed)
    
    # 7. Spaced with gap at origin (avoid small z)
    z_gap = [2.0 + 1.5*i for i in range(12)]
    candidates.append(z_gap)
    
    # Evaluate candidates
    best_z = None
    best_score = float('inf')
    deadline = time.time() + 580.0
    
    for z_cand in candidates:
        if time.time() > deadline:
            break
        # Validate
        z_sorted = sorted(z_cand)
        if any(z <= 0 or not math.isfinite(z) for z in z_sorted):
            continue
        valid = True
        for i in range(11):
            if abs(z_sorted[i] - z_sorted[i+1]) < 1e-8:
                valid = False
                break
        if not valid:
            continue
        
        score = _score_exact(z_cand)
        if score < best_score:
            best_score = score
            best_z = list(z_cand)
    
    if best_z is None:
        return z_bessel
    
    # Local search refinement
    if time.time() < deadline - 60:
        current_z = list(best_z)
        current_score = best_score
        np.random.seed(12345)
        
        # Phase 1: Coarse search - perturb one element at a time
        for step in range(40):
            if time.time() > deadline - 30:
                break
            
            idx = np.random.randint(0, 12)
            factor = np.random.uniform(0.7, 1.3)
            candidate = list(current_z)
            candidate[idx] *= factor
            
            if candidate[idx] <= 0:
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
        
        # Phase 2: Fine polish
        if time.time() < deadline - 30:
            for step in range(30):
                if time.time() > deadline - 15:
                    break
                
                idx = np.random.randint(0, 12)
                factor = np.random.uniform(0.95, 1.05)
                candidate = list(current_z)
                candidate[idx] *= factor
                
                if candidate[idx] <= 0:
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
        
        # Phase 3: Very fine polish
        if time.time() < deadline - 15:
            for step in range(15):
                if time.time() > deadline - 5:
                    break
                
                idx = np.random.randint(0, 12)
                factor = np.random.uniform(0.99, 1.01)
                candidate = list(current_z)
                candidate[idx] *= factor
                
                if candidate[idx] <= 0:
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
    
    return best_z
