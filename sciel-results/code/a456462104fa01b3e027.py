"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Derivative-free root finder using inverse quadratic interpolation (IQI)
    with safeguarding, targeting smooth polynomial equations.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if abs(fx) <= 1e-12:
        return x
    
    # We'll maintain a history of (x, f(x)) points for IQI
    # Start with 2 points to bootstrap
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = f(x1)
    
    if abs(f1) <= 1e-12:
        return x1
    
    # Now we have 2 points: (x, fx), (x1, f1)
    # For IQI we need 3 points. Get a third point.
    # If signs differ, we have a bracket. Use secant first.
    # If signs same, try a different step.
    
    history = [(x, fx), (x1, f1)]
    best_x = x if abs(fx) < abs(f1) else x1
    best_fx = min(abs(fx), abs(f1))
    
    # Try to get a third point that ideally brackets
    if fx * f1 < 0:
        # We have a bracket [x, x1]. Good.
        # Do one secant step to get a third point inside.
        x_new = x - fx * (x1 - x) / (f1 - fx)
        # Safeguard: keep within bracket
        if not (min(x, x1) < x_new < max(x, x1)):
            x_new = (x + x1) / 2.0
        f_new = f(x_new)
        if abs(f_new) <= 1e-12:
            return x_new
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
        # Update bracket
        if fx * f_new < 0:
            x1 = x_new
            f1 = f_new
        else:
            x = x_new
            fx = f_new
        history.append((x_new, f_new))
    else:
        # No bracket. Try a larger step.
        h2 = 1e-2 * max(1.0, abs(x))
        x2 = x + h2
        f2 = f(x2)
        if abs(f2) <= 1e-12:
            return x2
        if abs(f2) < best_fx:
            best_fx = abs(f2)
            best_x = x2
        # Pick the best 3 points for history
        # Use (x, fx), (x1, f1), (x2, f2)
        history = [(x, fx), (x1, f1), (x2, f2)]
    
    # Main iteration loop
    # We'll use a combination of IQI and safeguarding
    # For polynomials, IQI should converge very fast (cubic or better)
    
    for it in range(50):
        if time.time() > deadline:
            break
        if len(history) < 3:
            # Need more points
            # Use secant on best 2
            (xa, fa), (xb, fb) = history[-2], history[-1]
            if fa == fb:
                break
            x_new = xa - fa * (xb - xa) / (fb - fa)
            f_new = f(x_new)
            if abs(f_new) <= 1e-12:
                return x_new
            if abs(f_new) < best_fx:
                best_fx = abs(f_new)
                best_x = x_new
            history.append((x_new, f_new))
            if len(history) > 5:
                history.pop(0)
            continue
        
        # IQI: fit quadratic through last 3 points
        (x0v, f0v), (x1v, f1v), (x2v, f2v) = history[-3], history[-2], history[-1]
        
        # Inverse quadratic interpolation:
        # Fit p(f) = a*f^2 + b*f + c such that p(f_i) = x_i
        # Then find f such that p(f) = x_new, i.e., a*f^2 + b*f + (c - x_new) = 0
        # Actually, we want to find f_new such that the quadratic through (f_i, x_i)
        # gives x_new. Standard IQI formula:
        
        if f0v == f1v or f1v == f2v or f0v == f2v:
            # Degenerate, use secant
            (xa, fa), (xb, fb) = history[-2], history[-1]
            if fa == fb:
                break
            x_new = xa - fa * (xb - xa) / (fb - fa)
        else:
            # Standard IQI formula
            # x_new = x2 + (x2 - x1) * (x2 - x0) * f2 / ((f2 - f1) * (f2 - f0)) 
            #         + (x2 - x0) * (x1 - x0) * f2 / ((f2 - f0) * (f1 - f0))
            # Wait, let me use the standard form:
            # Given points (x0, f0), (x1, f1), (x2, f2)
            # x_new = x2 - f2 * (x2 - x1) * (x2 - x0) / ((f2 - f1) * (f2 - f0))
            #        - f2 * (x2 - x0) * (x1 - x0) / ((f2 - f0) * (f1 - f0))
            # Hmm, let me just use the formula from Numerical Recipes or similar.
            # Actually, the standard IQI step is:
            # x_new = x2 - f2 * (x2 - x1) * (x2 - x0) / ((f2 - f1) * (f2 - f0))
            #         + f2 * (x1 - x0) * (x2 - x0) / ((f1 - f0) * (f2 - f0))
            # Let me just compute it directly using Lagrange basis.
            
            # We want to find f_new such that the quadratic interpolant Q(f) = x
            # where Q is the quadratic through (f0, x0), (f1, x1), (f2, x2)
            # Q(f) = x0 * (f - f1)(f - f2) / ((f0 - f1)(f0 - f2))
            #       + x1 * (f - f0)(f - f2) / ((f1 - f0)(f1 - f2))
            #       + x2 * (f - f0)(f - f1) / ((f2 - f0)(f2 - f1))
            # Set Q(f) = x_new... but we don't know x_new.
            
            # Actually, the standard IQI for root finding fits a quadratic q(x) to f(x)
            # and finds its root. But the "inverse" version fits a quadratic p(f) to x
            # and finds f such that p(f) = current x estimate.
            
            # Let me use the simpler approach: fit quadratic q(x) through 3 points,
            # find its root.
            
            # q(x) = a*x^2 + b*x + c
            # Solve: a*x0^2 + b*x0 + c = f0
            #        a*x1^2 + b*x1 + c = f1
            #        a*x2^2 + b*x2 + c = f2
            
            # This is a 3x3 linear system.
            # Let's solve it.
            
            A = [[x0v*x0v, x0v, 1.0],
                 [x1v*x1v, x1v, 1.0],
                 [x2v*x2v, x2v, 1.0]]
            B = [f0v, f1v, f2v]
            
            # Solve using Cramer's rule or direct inversion
            # Determinant
            det = (A[0][0] * (A[1][1]*A[2][2] - A[1][2]*A[2][1])
                 - A[0][1] * (A[1][0]*A[2][2] - A[1][2]*A[2][0])
                 + A[0][2] * (A[1][0]*A[2][1] - A[1][1]*A[2][0]))
            
            if abs(det) < 1e-30:
                # Degenerate, use secant
                (xa, fa), (xb, fb) = history[-2], history[-1]
                if fa == fb:
                    break
                x_new = xa - fa * (xb - xa) / (fb - fa)
            else:
                # Cramer's rule
                det_a = (B[0] * (A[1][1]*A[2][2] - A[1][2]*A[2][1])
                       - A[0][1] * (B[1]*A[2][2] - A[1][2]*B[2])
                       + A[0][2] * (B[1]*A[2][1] - A[1][1]*B[2]))
                det_b = (A[0][0] * (B[1]*A[2][2] - A[1][2]*B[2])
                       - B[0] * (A[1][0]*A[2][2] - A[1][2]*A[2][0])
                       + A[0][2] * (A[1][0]*B[2] - B[1]*A[2][0]))
                det_c = (A[0][0] * (A[1][1]*B[2] - B[1]*A[2][1])
                       - A[0][1] * (A[1][0]*B[2] - B[1]*A[2][0])
                       + B[0] * (A[1][0]*A[2][1] - A[1][1]*A[2][0]))
                
                a = det_a / det
                b = det_b / det
                c = det_c / det
                
                # Find root of a*x^2 + b*x + c = 0
                if abs(a) < 1e-30:
                    # Linear
                    if abs(b) < 1e-30:
                        break
                    x_new = -c / b
                else:
                    disc = b*b - 4*a*c
                    if disc < 0:
                        # No real root, use secant
                        (xa, fa), (xb, fb) = history[-2], history[-1]
                        if fa == fb:
                            break
                        x_new = xa - fa * (xb - xa) / (fb - fa)
                    else:
                        sqrt_disc = math.sqrt(disc)
                        # Choose the root that is closer to the current best
                        r1 = (-b + sqrt_disc) / (2*a)
                        r2 = (-b - sqrt_disc) / (2*a)
                        # Pick the one that gives smaller |f|
                        # Evaluate both? That costs 2 calls.
                        # Just pick the one closer to current x
                        x_curr = history[-1][0]
                        if abs(r1 - x_curr) < abs(r2 - x_curr):
                            x_new = r1
                        else:
                            x_new = r2
        
        # Safeguard: don't go too far
        if not math.isfinite(x_new):
            break
        
        f_new = f(x_new)
        if abs(f_new) <= 1e-12:
            return x_new
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
        
        # Update history: keep the 3 most recent points
        history.append((x_new, f_new))
        if len(history) > 3:
            history.pop(0)
        
        # Check for convergence
        if abs(f_new) <= 1e-10:
            return x_new
    
    # Final high-precision polish: if we have a good candidate, try Newton-like
    # refinement using finite differences to get below 1e-8 more reliably
    if best_fx > 1e-8 and best_fx < 1e-6:
        # Try a few secant steps from the best point
        xp = best_x
        fp = f(xp)
        for _ in range(10):
            if time.time() > deadline:
                break
            if abs(fp) <= 1e-8:
                return xp
            # Estimate derivative
            dx = 1e-8 * max(1.0, abs(xp))
            fp2 = f(xp + dx)
            df = (fp2 - fp) / dx
            if abs(df) < 1e-30:
                break
            x_new = xp - fp / df
            if not math.isfinite(x_new):
                break
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < best_fx:
                best_fx = abs(f_new)
                best_x = x_new
            xp = x_new
            fp = f_new
    
    return best_x
# EVOLVE-BLOCK-END
