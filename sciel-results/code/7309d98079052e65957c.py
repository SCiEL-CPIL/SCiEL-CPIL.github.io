import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre
from scipy.optimize import minimize, differential_evolution

def construct():
    def objective(zs):
        # zs: array of 12 floats
        # Check validity
        if np.any(zs <= 1e-6):
            return 1e10
        # Check distinctness
        if np.any(np.diff(np.sort(zs)) < 1e-6):
            return 1e10
            
        alpha = -0.5
        n_max = 25 # k=0..25
        
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        for k in range(26):
            A[0, k] = eval_genlaguerre(2*k, alpha, 0)
        b[0] = 0
        
        # Row 1: g'(0) = 1
        for k in range(26):
            if k == 0:
                A[1, k] = 0
            else:
                A[1, k] = -eval_genlaguerre(2*k-1, alpha+1, 0)
        b[1] = 1
        
        # Rows for z_i
        for i, z in enumerate(zs):
            row_g = 2 + i
            row_gp = 14 + i
            
            for k in range(26):
                A[row_g, k] = eval_genlaguerre(2*k, alpha, z)
                if k == 0:
                    A[row_gp, k] = 0
                else:
                    A[row_gp, k] = -eval_genlaguerre(2*k-1, alpha+1, z)
            
            b[row_g] = 0
            b[row_gp] = 0
            
        # Check condition number to avoid singular/near-singular systems
        cond = np.linalg.cond(A)
        if cond > 1e15:
            return 1e10
            
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e10
            
        # Construct g(x) polynomial coefficients
        g_coeffs = np.zeros(51) # Degree 50
        
        for k in range(26):
            p = genlaguerre(2*k, alpha, monic=False)
            coeffs = p.coeffs
            for j in range(len(coeffs)):
                power = 2*k - j
                idx = 50 - power
                g_coeffs[idx] += c[k] * coeffs[j]
                
        # Construct D(x) = x * prod (x - z_i)^2
        d_poly = np.poly1d([1.0, 0.0]) # x
        for z in zs:
            factor = np.poly1d([1.0, -2*z, z*z])
            d_poly = np.poly1d(d_poly * factor)
            
        g_poly = np.poly1d(g_coeffs)
        q_poly, rem = np.polydiv(g_poly, d_poly)
        
        # If remainder is significant, something is wrong (numerical error)
        # Theoretically it should be 0. If it's not, the division is inexact.
        # We ignore remainder for score but might penalize if it's huge.
        
        if len(q_poly.coeffs) <= 1:
            return 0.0
            
        # Find roots of q_poly
        # q_poly is degree 25 (50 - 25)
        roots = np.roots(q_poly.coeffs)
        
        # Filter real, positive roots
        # A real root r counts if q(r-eps) and q(r+eps) have opposite signs.
        # For a simple root of a polynomial, the sign change is a good indicator.
        # However, np.roots might return complex roots with tiny imaginary parts.
        # We check real roots.
        
        real_roots = []
        eps = 1e-9
        for r in roots:
            if abs(r.imag) < 1e-7 and r.real > 1e-9:
                # Check sign change
                # Evaluate q_poly at r - eps and r + eps
                # Note: q_poly is a poly1d object
                val1 = q_poly(r.real - eps)
                val2 = q_poly(r.real + eps)
                if val1 * val2 < 0:
                    real_roots.append(r.real)
                    
        if not real_roots:
            return 0.0
            
        return max(real_roots)

    # Define bounds for differential evolution
    # Roots must be > 0. Let's try a range [0.1, 50].
    # We can also try to sort them, but DE works on unsorted vectors.
    # To ensure distinctness, we rely on the penalty in objective.
    
    bounds = [(0.1, 50.0) for _ in range(12)]
    
    # Run Differential Evolution for global search
    # This is slow but robust.
    # We might need to reduce popsize or maxiter to fit time limits.
    # 12 dimensions, 26x26 solve per eval.
    # Let's try popsize=20, maxiter=100.
    
    try:
        res_de = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=15, tol=1e-6, polish=False)
        z_global = res_de.x
    except Exception:
        z_global = np.array([4.0 * (i + 1) for i in range(12)])
        
    # Refine with local optimization (Nelder-Mead or L-BFGS-B)
    # L-BFGS-B requires gradients or finite diff, which might be expensive.
    # Nelder-Mead is derivative-free and good for polishing.
    
    try:
        res_local = minimize(objective, z_global, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        z_opt = res_local.x
    except Exception:
        z_opt = z_global
        
    # Final check and cleanup
    if not np.all(z_opt > 0) or len(np.unique(np.round(z_opt, 6))) < 12:
        return [float(x) for x in [4.0 * (i + 1) for i in range(12)]]
        
    return [float(x) for x in z_opt]
