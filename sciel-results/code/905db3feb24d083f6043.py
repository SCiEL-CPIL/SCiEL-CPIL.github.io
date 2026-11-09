"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    t0 = time.time()
    DEADLINE = 620  # seconds, safety margin under 630s
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    def evaluate_z(z_list):
        """Evaluate the score for a given list of 12 z values.
        Returns the score B/(2*pi), or None if invalid/too slow.
        """
        if time_left() < 120:
            return None
        
        import sympy as sp
        
        z = [sp.Rational(float(zi)) for zi in z_list]
        
        # Check validity: all positive, distinct
        if any(zi <= 0 for zi in z):
            return None
        if len(set(z)) != len(z):
            return None
        
        alpha = sp.Rational(-1, 2)
        
        # Build the system
        # g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
        # Unknowns: c_0, c_1, ..., c_25 (26 unknowns)
        # Equations:
        #   g(0) = 0
        #   g'(0) = 1
        #   For each i in 0..11: g(z_i) = 0, g'(z_i) = 0 (24 equations)
        # Total: 2 + 24 = 26 equations
        
        c = sp.symbols('c0:26')
        
        x = sp.Symbol('x')
        
        # Build g(x)
        g = sp.Integer(0)
        for k in range(26):
            L_2k = sp.gen_laguerre(2 * k, alpha)
            g += c[k] * L_2k
        
        # Derivative
        g_prime = sp.diff(g, x)
        
        # Build the linear system
        # The system is linear in c_k since g is a linear combination
        # We need: g(0) = 0, g'(0) = 1, g(z_i) = 0, g'(z_i) = 0 for all i
        
        # For g(0) = 0: substitute x=0 into g
        # g(0) = sum c_k * L_{2k}(0)
        # For g'(0) = 1: substitute x=0 into g_prime
        # For g(z_i) = 0: substitute x=z_i into g
        # For g'(z_i) = 0: substitute x=z_i into g_prime
        
        # Since everything is linear in c, we can build the matrix directly
        # M * c = b where M is 26x26 and b is 26x1
        
        M = []
        b = []
        
        # Row 0: g(0) = 0
        row = []
        for k in range(26):
            L_2k = sp.gen_laguerre(2 * k, alpha)
            row.append(L_2k.subs(x, 0))
        M.append(row)
        b.append(sp.Integer(0))
        
        # Row 1: g'(0) = 1
        row = []
        for k in range(26):
            L_2k = sp.gen_laguerre(2 * k, alpha)
            dL_2k = sp.diff(L_2k, x)
            row.append(dL_2k.subs(x, 0))
        M.append(row)
        b.append(sp.Integer(1))
        
        # Rows 2-25: for each z_i, g(z_i) = 0 and g'(z_i) = 0
        for i in range(12):
            # g(z_i) = 0
            row = []
            for k in range(26):
                L_2k = sp.gen_laguerre(2 * k, alpha)
                row.append(L_2k.subs(x, z[i]))
            M.append(row)
            b.append(sp.Integer(0))
            
            # g'(z_i) = 0
            row = []
            for k in range(26):
                L_2k = sp.gen_laguerre(2 * k, alpha)
                dL_2k = sp.diff(L_2k, x)
                row.append(dL_2k.subs(x, z[i]))
            M.append(row)
            b.append(sp.Integer(0))
        
        # Convert to sympy matrices
        M_mat = sp.Matrix(M)
        b_mat = sp.Matrix(b)
        
        # Check for singularity
        if M_mat.det() == 0:
            return None
        
        # Solve using LU decomposition
        try:
            c_sol = M_mat.LUsolve(b_mat)
        except Exception:
            return None
        
        if time_left() < 100:
            return None
        
        # Now g(x) is determined. Compute q(x) = g(x) / (x * prod_i (x - z_i)^2)
        # The denominator has degree 1 + 12*2 = 25
        # g(x) has degree at most 50 (from L_50)
        # So q(x) should have degree at most 50 - 25 = 25
        
        denom = x
        for i in range(12):
            denom *= (x - z[i]) ** 2
        
        g_full = sum(c_sol[k] * sp.gen_laguerre(2 * k, alpha) for k in range(26))
        
        # Polynomial division: q = g_full / denom
        try:
            q, rem = sp.div(g_full, denom, x)
        except Exception:
            return None
        
        if rem != 0:
            # If remainder is not zero, this is problematic
            # But per the task, it should be exact
            return None
        
        if time_left() < 60:
            return None
        
        # Find real roots of q
        try:
            real_roots = sp.real_roots(q)
        except Exception:
            return None
        
        # For each real root r, check if q has strictly opposite signs at r-1e-198 and r+1e-198
        # 1e-198 is extremely small, essentially checking sign change
        count_roots = []
        for r in real_roots:
            # r is a sympy expression
            r_val = float(r)
            if not np.isfinite(r_val) or r_val <= 0:
                continue
            
            eps = sp.Rational(1, 10) ** 198  # 1e-198 as rational
            
            q_at_minus = q.subs(x, r - eps)
            q_at_plus = q.subs(x, r + eps)
            
            # Check if signs are strictly opposite
            # Since these are exact rationals, we can check the sign
            try:
                sign_minus = -1 if q_at_minus < 0 else (1 if q_at_minus > 0 else 0)
                sign_plus = -1 if q_at_plus < 0 else (1 if q_at_plus > 0 else 0)
            except Exception:
                continue
            
            if sign_minus != 0 and sign_plus != 0 and sign_minus != sign_plus:
                count_roots.append(r_val)
        
        if not count_roots:
            B = 0.0
        else:
            B = max(count_roots)
        
        score = B / (2 * np.pi)
        return score
    
    # Initial candidates
    # The seed uses 4*(i+1) = 4, 8, 12, ..., 48
    # Let's try several initial configurations
    
    initial_configs = [
        [4.0 * (i + 1) for i in range(12)],  # Seed
        [2.0 * (i + 1) for i in range(12)],  # Compressed
        [6.0 * (i + 1) for i in range(12)],  # Expanded
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 17.0, 22.0, 28.0],
        [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 24.0, 32.0],
    ]
    
    best_score = float('inf')
    best_z = initial_configs[0]
    
    # Evaluate each initial config
    for config in initial_configs:
        if time_left() < 120:
            break
        
        if any(z <= 0 for z in config) or len(set([round(z, 10) for z in config])) != len(config):
            continue
        
        score = evaluate_z(config)
        if score is not None and score < best_score:
            best_score = score
            best_z = list(config)
        
        if time_left() < 120:
            break
    
    # Try local optimization from the best point
    if time_left() > 150 and best_z is not None:
        try:
            def objective(z_arr):
                if time_left() < 90:
                    return 1e10
                z_list = [float(z) for z in z_arr]
                if any(z <= 0.001 for z in z_list):
                    return 1e10
                if len(set([round(z, 8) for z in z_list])) != len(z_list):
                    return 1e10
                s = evaluate_z(z_list)
                return s if s is not None else 1e10
            
            # Nelder-Mead with limited iterations
            res = minimize(
                objective,
                np.array(best_z),
                method='Nelder-Mead',
                options={'maxiter': 8, 'xatol': 0.05, 'fatol': 1e-4, 'adaptive': True}
            )
            
            if time_left() > 60:
                if res.fun < best_score:
                    z_candidate = [float(z) for z in res.x]
                    # Validate
                    if all(z > 0 for z in z_candidate) and len(set([round(z, 8) for z in z_candidate])) == len(z_candidate):
                        best_score = res.fun
                        best_z = z_candidate
        except Exception:
            pass
    
    # Try a few random perturbations
    if time_left() > 120:
        np.random.seed(42)
        for _ in range(3):
            if time_left() < 100:
                break
            perturbed = [z * (1 + 0.15 * np.random.randn()) for z in best_z]
            perturbed = [max(0.1, z) for z in perturbed]
            perturbed.sort()
            # Ensure distinct
            for i in range(1, len(perturbed)):
                if perturbed[i] <= perturbed[i-1]:
                    perturbed[i] = perturbed[i-1] + 0.01
            
            score = evaluate_z(perturbed)
            if score is not None and score < best_score:
                best_score = score
                best_z = perturbed
    
    # Final validation
    if best_z is None:
        return [4.0 * (i + 1) for i in range(12)]
    
    final_z = [float(z) for z in best_z]
    for zi in final_z:
        if not np.isfinite(zi) or zi <= 0:
            return [4.0 * (i + 1) for i in range(12)]
    if len(set([round(z, 10) for z in final_z])) != len(final_z):
        return [4.0 * (i + 1) for i in range(12)]
    
    return final_z
# EVOLVE-BLOCK-END
