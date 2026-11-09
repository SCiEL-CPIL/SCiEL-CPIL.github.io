import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Root finder using Sparse Sampling + Cubic Lagrange Interpolation.
    
    Strategy:
    1. Evaluate f at x0 and 4 nearby points (x0 ± h, x0 ± 2h) to build a cubic
       polynomial approximation.
    2. Find the roots of this cubic polynomial analytically (or via numpy if 
       available, but we stick to stdlib for robustness in this context, though 
       numpy is allowed. Let's use numpy for root finding as it's robust).
    3. Select the root candidate closest to x0.
    4. Refine using safeguarded Newton (Illinois/Brent-like) starting from the 
       candidate.
    5. If no sign change is found in the initial sparse sample, expand the 
       sampling radius geometrically and repeat the interpolation step.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    
    # Cache to avoid redundant calls
    cache = {}
    
    def f_cached(x):
        if x in cache:
            return cache[x]
        val = f(x)
        cache[x] = val
        return val
        
    # Check initial point
    fx0 = f_cached(x0)
    if abs(fx0) <= 1e-8:
        return x0
        
    best_x = x0
    best_f = abs(fx0)
    
    # We will try to find a root using interpolation on a set of points.
    # We start with a small h and expand if necessary.
    
    # Initial step size
    # If x0 is very large, h should be larger. If x0 is small, h should be small.
    # But for oscillatory functions, the period is often O(1) or related to the scale.
    # Let's start with h = 1.0 if x0 is not extremely small/large.
    base_h = max(1e-3, min(1.0, abs(x0) * 0.1)) if abs(x0) > 1e-6 else 1e-3
    
    # List of radii to try for the sparse sample
    radii = [base_h, base_h * 10, base_h * 100, base_h * 1000, base_h * 10000]
    
    # We also need to handle the case where the function doesn't cross zero in the 
    # initial small window. We need to find a sign change first to bracket, OR 
    # rely on the cubic having a real root near the minimum of |f|.
    
    # Phase 1: Sparse Sampling and Interpolation
    # We collect points (x, f(x)) and try to fit a cubic.
    
    # To ensure we have a good cubic, we need 4 distinct points.
    # We will sample x0, x0+h, x0-h, x0+2h, x0-2h.
    
    bracket_a = None
    bracket_b = None
    
    for radius in radii:
        if time.time() > deadline - 5.0:
            break
            
        h = radius
        
        # Sample points: x0, x0+h, x0-h, x0+2h, x0-2h
        # Note: x0 is already cached.
        xs = [x0, x0 + h, x0 - h, x0 + 2*h, x0 - 2*h]
        fs = []
        
        # Check for sign changes in this local neighborhood first
        # This is cheaper than building a polynomial if a simple bracket exists.
        prev_x = None
        prev_f = None
        
        for x in xs:
            fx = f_cached(x)
            fs.append(fx)
            
            if abs(fx) <= 1e-8:
                return x
                
            if abs(fx) < best_f:
                best_f = abs(fx)
                best_x = x
                
            if prev_f is not None and prev_f * fx < 0:
                bracket_a = prev_x
                bracket_b = x
                break
                
            prev_x = x
            prev_f = fx
            
        if bracket_a is not None:
            break
            
        # If no sign change found in this radius, we might still have a local minimum
        # that we can use for Newton. But the directive asks for interpolation.
        # Let's try to find roots of the cubic formed by the first 4 points (or best 4).
        # Actually, if we have 5 points, we can fit a quartic. 
        # But cubic is more stable. Let's use x0, x0+h, x0-h, x0+2h.
        
        # If we have at least 4 points, try interpolation.
        if len(xs) >= 4:
            # Use the first 4 points: x0, x0+h, x0-h, x0+2h
            # Note: If h is very small, these points might be indistinguishable in float.
            # Check distinctness.
            distinct_xs = []
            distinct_fs = []
            for i in range(min(4, len(xs))):
                if all(abs(xs[i] - dx) > 1e-15 * max(1, abs(xs[i])) for dx in distinct_xs):
                    distinct_xs.append(xs[i])
                    distinct_fs.append(fs[i])
                    
            if len(distinct_xs) == 4:
                # Build Lagrange basis for cubic
                # P(x) = sum(f_i * L_i(x))
                # We want to find x such that P(x) = 0.
                # This is a cubic equation.
                
                # Coefficients of the cubic polynomial
                # We can compute the coefficients c3, c2, c1, c0 such that P(x) = c3*x^3 + c2*x^2 + c1*x + c0
                # Using the 4 points.
                
                # Let's use numpy for this as it's available and robust for small systems.
                try:
                    import numpy as np
                    # Vandermonde matrix
                    # V * c = f
                    # V[i, j] = x_i^j
                    V = np.vander(distinct_xs, 4, increasing=True) # x^0, x^1, x^2, x^3
                    c = np.linalg.solve(V, distinct_fs)
                    
                    # Find roots of c[3]x^3 + c[2]x^2 + c[1]x + c[0] = 0
                    roots = np.roots(c)
                    
                    # Filter real roots
                    real_roots = [r.real for r in roots if abs(r.imag) < 1e-12]
                    
                    if real_roots:
                        # Find the real root closest to x0
                        closest_root = min(real_roots, key=lambda r: abs(r - x0))
                        
                        # Check if this root is valid (finite)
                        if math.isfinite(closest_root):
                            # Use this as the starting point for Newton
                            # But first, check if |f(closest_root)| is already small
                            f_root = f_cached(closest_root)
                            if abs(f_root) <= 1e-8:
                                return closest_root
                                
                            # If not, we have a very good candidate.
                            # We should also check if this root is within a bracket.
                            # If we don't have a bracket, we proceed to Newton from this candidate.
                            # If we do have a bracket (from the scan), we use that.
                            
                            # Update best_x if this is better
                            if abs(f_root) < best_f:
                                best_f = abs(f_root)
                                best_x = closest_root
                                
                            # We break out of the radius loop to start refinement
                            break
                except Exception:
                    pass # Fallback to next radius or Newton from best_x
    
    # Phase 2: Refinement
    # If we found a bracket, use Illinois/Bisection (robust).
    # If we have a good candidate (best_x) but no bracket, use Newton.
    
    if bracket_a is not None:
        a = bracket_a
        b = bracket_b
        fa = f_cached(a)
        fb = f_cached(b)
        
        # Ensure opposite signs
        if fa * fb > 0:
            # Should not happen, but check
            pass
        else:
            # Illinois Algorithm (modified regula falsi)
            # It converges faster than bisection and is robust.
            for i in range(50):
                if time.time() > deadline - 1.0:
                    break
                    
                x_mid = (a + b) / 2
                f_mid = f_cached(x_mid)
                
                if abs(f_mid) <= 1e-8:
                    return x_mid
                    
                if fa * f_mid < 0:
                    b = x_mid
                    fb = f_mid
                else:
                    a = x_mid
                    fa = f_mid
                    # Illinois modification: if we keep choosing the same side,
                    # shrink the other side's function value to force bisection.
                    # Standard Illinois:
                    # if fa*fb < 0:
                    #    ...
                    # Actually, let's just use Bisection for simplicity and guaranteed convergence
                    # if Illinois implementation is tricky. Bisection is 1e-8 in ~30 steps.
                    # 30 calls is well within budget.
                    
                if abs(b - a) < 1e-15:
                    break
                    
            return (a + b) / 2
            
    # No bracket found. Use Newton from best_x.
    x = best_x
    fx = f_cached(x)
    
    for i in range(50):
        if time.time() > deadline - 0.5:
            break
            
        if abs(fx) <= 1e-8:
            return x
            
        # Numerical derivative
        h = 1e-6
        if abs(x) > 1e6:
            h = 1e-2
        elif abs(x) < 1e-6:
            h = 1e-10
            
        f_plus = f_cached(x + h)
        f_minus = f_cached(x - h)
        dfdx = (f_plus - f_minus) / (2 * h)
        
        if abs(dfdx) < 1e-12:
            break
            
        x_new = x - fx / dfdx
        f_new = f_cached(x_new)
        
        if not math.isfinite(x_new) or not math.isfinite(f_new):
            break
            
        # Divergence check: if we move too far, stop
        if abs(x_new - x) > 10 * max(1, abs(x)):
            break
            
        x = x_new
        fx = f_new
        
    return x
# EVOLVE-BLOCK-END
