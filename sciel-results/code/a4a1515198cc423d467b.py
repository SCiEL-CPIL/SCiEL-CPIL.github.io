"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def find_root(f, x0):
    """
    Find a root of f by minimizing g(x) = f(x)^2 using a 1D Nelder-Mead 
    (simplex) method, followed by a high-precision refinement.
    """
    x0 = float(x0)
    
    eval_count = 0
    max_evals = 500
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            raise RuntimeError("Evaluation limit exceeded")
        eval_count += 1
        return float(f(x))
    
    def eval_g(x):
        val = eval_f(x)
        return val * val

    # Initial points
    # Scale the initial step based on x0
    scale = max(1.0, abs(x0))
    delta = 1e-4 * scale
    
    x1 = x0
    x2 = x0 + delta
    
    g1 = eval_g(x1)
    g2 = eval_g(x2)
    
    # Ensure g1 <= g2
    if g1 > g2:
        x1, x2 = x2, x1
        g1, g2 = g2, g1
    
    # If we already have a root
    if abs(np.sqrt(g1)) <= 1e-8:
        return x1

    # Nelder-Mead 1D loop
    # Parameters
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5   # Contraction
    sigma = 0.5 # Shrinkage
    
    for _ in range(max_evals - eval_count - 10): # Reserve 10 evals for refinement
        if eval_count >= max_evals:
            break
            
        # Current simplex: x1 (best), x2 (worst)
        # Check convergence
        if abs(x2 - x1) < 1e-12 * scale:
            break
        if g1 < 1e-16: # Very small residual
             break

        # Reflection
        x_r = x1 - alpha * (x2 - x1)
        g_r = eval_g(x_r)
        
        if g_r < g1:
            # Expansion
            x_e = x1 - gamma * (x2 - x1)
            g_e = eval_g(x_e)
            
            if g_e < g_r:
                x1, g1 = x_e, g_e
                x2, g2 = x1, g1 # Wait, x2 should be the other point in the new simplex
                # In 1D Nelder-Mead, if we expand, the new simplex is [x_e, x1] or [x1, x_r]?
                # Standard NM: if g_e < g_r, replace x2 with x_e. Simplex becomes [x1, x_e]?
                # No, x1 was the best. If x_e is better than x_r, and x_r was better than x1?
                # No, g_r < g1 was the condition. So x_r is better than x1.
                # If g_e < g_r, x_e is the best.
                # The new simplex should contain the best point and the next best.
                # The points are x1, x2, x_r, x_e.
                # Best is x_e. Next best is x_r (since g_r < g1).
                # So new simplex is [x_e, x_r].
                x1, g1 = x_e, g_e
                x2, g2 = x_r, g_r
            else:
                # g_r < g1 <= g_e
                # New simplex is [x1, x_r]
                x2, g2 = x_r, g_r
        elif g_r < g2:
            # g1 <= g_r < g2
            # New simplex is [x1, x_r]
            x2, g2 = x_r, g_r
        else:
            # g_r >= g2
            # Contraction
            x_c = x1 + rho * (x2 - x1)
            g_c = eval_g(x_c)
            
            if g_c < g2:
                # New simplex is [x1, x_c]
                x2, g2 = x_c, g_c
            else:
                # Shrinkage
                x_s = x1 + sigma * (x2 - x1)
                g_s = eval_g(x_s)
                x2, g2 = x_s, g_s
                
                # If shrinkage doesn't help, we might be stuck.
                # But in 1D, this is rare if the function is smooth.
                
    # Refinement Phase
    # We have a small interval [x1, x2] containing the minimum of g.
    # We want |f(x)| <= 1e-8.
    # Check if we have a sign change in f in [x1, x2].
    # If yes, use Bisection/Illinois.
    # If no, use a local minimization step (Golden Section or just check midpoint).
    
    # Get f values for x1, x2
    # Note: we have g1, g2. We need f1, f2.
    # We might have to re-evaluate or store them. 
    # To save evals, let's just re-evaluate f for the current x1, x2 if we are close.
    # Actually, we can just check if sqrt(g1) or sqrt(g2) is good enough.
    
    if np.sqrt(g1) <= 1e-8:
        return x1
    if np.sqrt(g2) <= 1e-8:
        return x2

    # Try to find a bracket with sign change for Illinois/Bisection
    # We need f(x1) and f(x2)
    f1 = eval_f(x1)
    f2 = eval_f(x2)
    
    # If signs are different, use Illinois
    if f1 * f2 < 0:
        a, b = x1, x2
        fa, fb = f1, f2
        
        # Illinois Method
        for _ in range(max_evals - eval_count):
            if eval_count >= max_evals:
                break
            if abs(fa) <= 1e-8: return a
            if abs(fb) <= 1e-8: return b
            
            # Illinois step
            if abs(fa) > abs(fb):
                fa = fa * 0.5
                a = (a * fb - b * fa) / (fb - fa)
                fa_new = eval_f(a)
                if abs(fa_new) <= 1e-8: return a
                if fa_new * fb < 0:
                    b, fb = a, fa_new
                else:
                    # This case shouldn't happen if logic is right, but handle it
                    # Actually, Illinois updates the side with larger |f|
                    # If we updated a, we check if a is between old a and b?
                    # Standard Illinois:
                    # If |fa| > |fb|: fa = fa/2; a = (a*fb - b*fa)/(fb-fa)
                    # Then evaluate f(a).
                    # If f(a)*f(b) < 0, new bracket is [a, b].
                    # Else, new bracket is [a, old_a]? No.
                    # Let's stick to Bisection for safety in refinement if Illinois gets tricky.
                    # But Illinois is faster.
                    # Let's implement standard Illinois carefully.
                    pass
            else:
                fb = fb * 0.5
                b = (a * fb - b * fa) / (fb - fa)
                fb_new = eval_f(b)
                if abs(fb_new) <= 1e-8: return b
                if fa * fb_new < 0:
                    a, fa = b, fb_new
                else:
                    pass
            
            # Simplified Illinois implementation below to avoid bugs
            # Let's just use Bisection for the final 10-20 steps, it's robust.
            # Or a simple Regula Falsi with Illinois modification.
            
            # Let's restart the refinement loop properly.
            break # Exit this loop to use a cleaner implementation below

    # Clean Refinement: Bisection on [x1, x2] if sign change, else Golden Section on g
    a, b = x1, x2
    fa = eval_f(a)
    fb = eval_f(b)
    
    if fa * fb < 0:
        # Bisection
        for _ in range(max_evals - eval_count):
            if eval_count >= max_evals: break
            if abs(fa) <= 1e-8: return a
            if abs(fb) <= 1e-8: return b
            
            mid = (a + b) / 2.0
            fmid = eval_f(mid)
            
            if abs(fmid) <= 1e-8: return mid
            
            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
        return a if abs(fa) < abs(fb) else b
    else:
        # No sign change. Minimize g locally.
        # Use Golden Section Search on [x1, x2]
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = eval_g(c)
        fd = eval_g(d)
        
        for _ in range(max_evals - eval_count):
            if eval_count >= max_evals: break
            
            if abs(np.sqrt(fc)) <= 1e-8: return c
            if abs(np.sqrt(fd)) <= 1e-8: return d
            
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = eval_g(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = eval_g(d)
                
        return c if fc < fd else d

    return x1 if g1 < g2 else x2
# EVOLVE-BLOCK-END
