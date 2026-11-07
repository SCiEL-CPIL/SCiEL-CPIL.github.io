"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Use a multi-scale global grid search to find a sign-change bracket.
     - Evaluate f at exponentially spaced points across a wide domain.
     - Track the best (smallest |f|) point and any sign-change intervals.
  3. Once a bracket [a, b] with f(a)*f(b) < 0 is found, use the Illinois
     method (inverse linear interpolation with bisection safeguard) to
     converge to |f| <= 1e-10.
  4. If no bracket is found, use a damped Newton / secant local minimization
     of |f(x)| to drive f as close to zero as possible.
  5. Track the best (smallest |f|) point seen so far and return it if the
     deadline is approached.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 145.0  # 145s out of 150s wall-clock limit

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def _illinois_solve(f, a, fa, b, fb, max_iters=80):
    """
    Solve for root in bracket [a, b] using Illinois method.
    Returns (x, fx) with smallest |fx| found.
    """
    left = a
    fleft = fa
    right = b
    fright = fb
    
    # Ensure f(left) * f(right) < 0
    if fleft * fright > 0:
        return (left if abs(fleft) < abs(fright) else right, 
                fleft if abs(fleft) < abs(fright) else fright)
    
    best_x = left if abs(fleft) < abs(fright) else right
    best_f = fleft if abs(fleft) < abs(fright) else fright
    
    for _ in range(max_iters):
        if _elapsed() > _DEADLINE:
            break
        
        if abs(fleft) <= 1e-10:
            return left, fleft
        if abs(fright) <= 1e-10:
            return right, fright
        
        # Illinois interpolation: inverse linear interpolation
        denom = fright - fleft
        if abs(denom) > 1e-30:
            x_new = (left * fright - right * fleft) / denom
        else:
            x_new = (left + right) / 2.0
        
        # Safeguard: if x_new is not strictly between left and right, use bisection
        if not (left < x_new < right):
            x_new = (left + right) / 2.0
        
        try:
            f_new = f(x_new)
        except:
            f_new = None
        
        if f_new is None or not _is_finite(f_new):
            # Fallback to bisection
            x_new = (left + right) / 2.0
            try:
                f_new = f(x_new)
            except:
                f_new = None
            if f_new is None or not _is_finite(f_new):
                break
        
        if abs(f_new) < abs(best_f):
            best_x = x_new
            best_f = f_new
        
        if abs(f_new) <= 1e-8:
            return x_new, f_new
        
        if f_new * fleft < 0:
            right = x_new
            fright = f_new
        else:
            left = x_new
            fleft = f_new
        
        # Check convergence
        if abs(right - left) < 1e-14 * max(1.0, abs(left), abs(right)):
            break
    
    # Return the point with smallest |f|
    if abs(fleft) <= abs(fright):
        if abs(fleft) < abs(best_f):
            best_x = left
            best_f = fleft
        return best_x, best_f
    else:
        if abs(fright) < abs(best_f):
            best_x = right
            best_f = fright
        return best_x, best_f

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8, using a combination of
    global grid search, Illinois bracketing, and Newton-like methods.
    
    Returns a float x, or raises an exception if no root can be found.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0
    
    best_x = x0
    best_f = None
    try:
        best_f = f(x0)
    except:
        best_f = float('inf')
    
    if best_f is not None and _is_finite(best_f) and abs(best_f) <= 1e-8:
        return x0
    
    # Phase 1: Multi-scale global grid search to find sign-change brackets
    # Evaluate f at exponentially spaced points across a wide domain
    
    # Build a set of candidate points
    candidates = []
    
    # Points around x0 at various scales
    base_scale = max(1.0, abs(x0))
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 
                  50.0, 100.0, 200.0, 500.0, 1000.0]:
        for sign in [1, -1]:
            candidates.append(x0 + sign * scale * base_scale)
    
    # Also try absolute positions
    for pos in [1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0,
                50.0, -50.0, 100.0, -100.0, 200.0, -200.0, 500.0, -500.0]:
        candidates.append(pos)
    
    # Remove duplicates and sort
    seen = set()
    unique_candidates = []
    for c in candidates:
        if _is_finite(c) and c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    
    # Evaluate f at all candidates, tracking best and sign changes
    evals = {}  # x -> f(x)
    evals[x0] = best_f
    
    best_x = x0
    best_f = best_f if _is_finite(best_f) else float('inf')
    
    # Track sign changes: list of (x1, f1, x2, f2) where f1*f2 < 0
    sign_changes = []
    
    for c in unique_candidates:
        if _elapsed() > _DEADLINE:
            break
        try:
            fc = f(c)
        except:
            continue
        
        if not _is_finite(fc):
            continue
        
        evals[c] = fc
        
        if abs(fc) < abs(best_f):
            best_x = c
            best_f = fc
        
        if abs(fc) <= 1e-8:
            return c
        
        # Check for sign changes with all previously evaluated points
        for x_prev, f_prev in evals.items():
            if x_prev == c:
                continue
            if f_prev is None or not _is_finite(f_prev):
                continue
            if f_prev * fc < 0:
                sign_changes.append((x_prev, f_prev, c, fc))
    
    # If we found sign changes, use the one with smallest interval
    if sign_changes:
        # Sort by interval width
        sign_changes.sort(key=lambda t: abs(t[0] - t[2]))
        
        for a, fa, b, fb in sign_changes:
            if _elapsed() > _DEADLINE:
                break
            if fa * fb >= 0:
                continue
            
            x_sol, f_sol = _illinois_solve(f, a, fa, b, fb)
            
            if abs(f_sol) <= 1e-8:
                return x_sol
            
            if abs(f_sol) < abs(best_f):
                best_x = x_sol
                best_f = f_sol
            
            # If we got close enough, return
            if abs(f_sol) <= 1e-6:
                break
    
    # Phase 2: If no bracket found, try local Newton-like search
    # Try multiple restarts with different step sizes
    for base_x in [x0, x0 + 0.5, x0 - 0.5, x0 + 1.0, x0 - 1.0, 
                   x0 + 2.0, x0 - 2.0, 0.0, 1.0, -1.0]:
        if _elapsed() > _DEADLINE:
            break
        
        x = base_x
        try:
            fx = f(x)
        except:
            continue
        
        if fx is None or not _is_finite(fx):
            continue
        
        if abs(fx) < abs(best_f):
            best_x = x
            best_f = fx
        
        if abs(fx) <= 1e-8:
            return x
        
        for _ in range(40):
            if _elapsed() > _DEADLINE:
                break
            
            if abs(fx) <= 1e-8:
                return x
            
            # Finite difference derivative
            h = 1e-7 * max(1.0, abs(x))
            try:
                fx_plus = f(x + h)
            except:
                fx_plus = None
            
            if fx_plus is None or not _is_finite(fx_plus):
                break
            
            deriv = (fx_plus - fx) / h
            if deriv == 0 or not _is_finite(deriv):
                break
            
            x_new = x - fx / deriv
            if not _is_finite(x_new):
                break
            
            # Damped step to avoid divergence
            if abs(x_new - x) > max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x) * (1 if x_new > x else -1)
            
            try:
                fx_new = f(x_new)
            except:
                fx_new = None
            
            if fx_new is None or not _is_finite(fx_new):
                # Try a smaller step
                x_new = x - 0.1 * fx / deriv
                if not _is_finite(x_new):
                    break
                try:
                    fx_new = f(x_new)
                except:
                    fx_new = None
                if fx_new is None or not _is_finite(fx_new):
                    break
            
            if abs(fx_new) < abs(best_f):
                best_x = x_new
                best_f = fx_new
            
            if abs(fx_new) <= 1e-8:
                return x_new
            
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                # Step was bad, try smaller step
                x_new = x - 0.1 * fx / deriv
                if not _is_finite(x_new):
                    break
                try:
                    fx_new = f(x_new)
                except:
                    fx_new = None
                if fx_new is None or not _is_finite(fx_new):
                    break
                if abs(fx_new) < abs(best_f):
                    best_x = x_new
                    best_f = fx_new
                if abs(fx_new) <= 1e-8:
                    return x_new
                if abs(fx_new) < abs(fx):
                    x = x_new
                    fx = fx_new
    
    # Return best found
    if best_f is not None and _is_finite(best_f) and abs(best_f) <= 1e-8:
        return best_x
    
    return best_x

# EVOLVE-BLOCK-END
