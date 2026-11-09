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
    
    # 1. Find initial points: Roots of L_12^(-1/2)
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    L12 = sp.Laguerre(12, alpha, x)
    
    try:
        roots_L12 = sp.nroots(L12, n=50, maxsteps=100)
        z_current = sorted([float(r) for r in roots_L12 if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0])
        if len(z_current) != 12:
             # Fallback if numerical roots fail
             z_current = [4.0 * (i + 1) for i in range(12)]
    except:
        z_current = [4.0 * (i + 1) for i in range(12)]
    
    best_B = evaluate_candidate(z_current)
    if best_B is None:
        return z_current
        
    best_z = list(z_current)
    
    # 2. Newton's Method Optimization
    # We want to minimize B = max(r_j).
    # We assume the maximum root is unique and smooth.
    # B(z) = r_max(z).
    # Gradient of B with respect to z_k:
    # dB/dz_k = (dr/dz_k) for the max root.
    # From implicit function theorem on q(r, z) = 0:
    # (dr/dz_k) = - (dq/dz_k) / (dq/dr)
    
    # We will approximate gradients by finite differences if analytic is too slow,
    # but let's try to compute the derivative of the root with respect to z_k
    # using the implicit function theorem numerically or symbolically if possible.
    # Given the cost of sympy, numerical differentiation of the root finder might be safer.
    # However, we can use the property that q(r) = 0.
    
    # Let's implement a simple coordinate descent / Newton step using numerical gradients
    # of the objective function B(z).
    
    step_size = 0.1
    max_iter = 20 # Limit iterations due to time cost
    
    for iter_idx in range(max_iter):
        if time.time() > deadline - 60: # Keep 60s buffer
            break
            
        # Compute gradient numerically
        grad = np.zeros(12)
        for k in range(12):
            z_plus = list(z_current)
            z_plus[k] += step_size
            z_minus = list(z_current)
            z_minus[k] -= step_size
            
            # Ensure positivity
            if z_minus[k] <= 0:
                z_minus[k] = 1e-6
            
            B_plus = evaluate_candidate(z_plus)
            B_minus = evaluate_candidate(z_minus)
            
            if B_plus is not None and B_minus is not None:
                grad[k] = (B_plus - B_minus) / (2 * step_size)
            else:
                # If invalid, push towards valid region (heuristic: increase z if it was too small?)
                # Or just skip this dimension for this step
                grad[k] = 0.0
            
            if time.time() > deadline - 60:
                break
        
        # Check convergence
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-4:
            break
            
        # Newton step: z_new = z - alpha * grad
        # We need to choose alpha. Let's try a small step first.
        alpha = 0.5
        z_new = [z_current[i] - alpha * grad[i] for i in range(12)]
        
        # Enforce positivity and distinctness
        valid = True
        for i in range(12):
            if z_new[i] <= 0:
                z_new[i] = 1e-6
            # Ensure distinctness
            for j in range(i+1, 12):
                if abs(z_new[i] - z_new[j]) < 1e-6:
                    z_new[j] += 1e-5
        
        B_new = evaluate_candidate(z_new)
        
        if B_new is not None and B_new < best_B:
            best_B = B_new
            best_z = z_new
            z_current = z_new
            # Adaptive step size: if we improved, keep or increase step slightly
            # alpha = min(alpha * 1.1, 1.0)
        else:
            # If no improvement, reduce step size
            alpha *= 0.5
            if alpha < 0.01:
                break
            # Try again with smaller step? To save time, just break or try one more time
            # Let's just break to save time, as we are constrained.
            break
            
    return best_z

if __name__ == '__main__':
    # Just for local testing
    start = time.time()
    res = construct()
    print(f"Time: {time.time() - start}")
    print(f"Result: {res}")
