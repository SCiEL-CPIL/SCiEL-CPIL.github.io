"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust hybrid approach:
    1. Try Newton-like methods with adaptive step control
    2. Secant method as primary
    3. Bisection fallback when bracket is found
    4. Multiple restart strategies for ill-conditioned problems
    """
    x0 = float(x0)
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    max_evals = 500
    eval_count = [0]
    
    def eval_f(x):
        """Evaluate f with counting and deadline check."""
        eval_count[0] += 1
        if eval_count[0] > max_evals:
            raise RuntimeError("Too many evaluations")
        if time.time() > deadline:
            raise RuntimeError("Deadline exceeded")
        val = f(x)
        return val
    
    # Try multiple strategies
    strategies = [
        _secant_newton_hybrid,
        _brent_like,
        _restart_search,
    ]
    
    best_x = x0
    best_f = None
    try:
        fx0 = eval_f(x0)
        best_f = fx0
        best_x = x0
    except:
        return x0
    
    if abs(best_f) <= 1e-8:
        return best_x
    
    for strategy in strategies:
        if eval_count[0] >= max_evals or time.time() > deadline:
            break
        try:
            x = strategy(f, x0, eval_f, eval_count, deadline, best_x, best_f)
            if x is not None:
                fx = eval_f(x)
                if abs(fx) <= 1e-8:
                    return x
                if best_f is None or abs(fx) < abs(best_f):
                    best_f = fx
                    best_x = x
        except:
            continue
    
    return best_x


def _secant_newton_hybrid(f, x0, eval_f, eval_count, deadline, best_x, best_f):
    """Secant method with Newton refinement and adaptive step control."""
    x = float(x0)
    fx = best_f
    
    # Try to get a second point for secant
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    try:
        fx1 = eval_f(x1)
    except:
        return None
    
    if abs(fx1) <= 1e-8:
        return x1
    
    # Secant iterations with occasional Newton steps
    for i in range(80):
        if eval_count[0] >= 500 or time.time() > deadline:
            break
        
        # Secant step
        if fx1 != fx and (x1 - x) != 0:
            x_new = x - fx * (x1 - x) / (fx1 - fx)
        else:
            # Fall back to Newton-like with finite difference
            h2 = 1e-6 * max(1.0, abs(x))
            try:
                fxh = eval_f(x + h2)
            except:
                break
            if fxh == fx:
                break
            d = (fxh - fx) / h2
            if d == 0:
                break
            x_new = x - fx / d
        
        # Check for convergence
        if abs(x_new - x) < 1e-12 * max(1.0, abs(x_new)):
            try:
                fx_new = eval_f(x_new)
            except:
                break
            if abs(fx_new) <= 1e-8:
                return x_new
            break
        
        # Adaptive step: if we're diverging, dampen
        if abs(x_new) > 1e10:
            x_new = x + 0.5 * (x_new - x)
        
        # Shift points
        x, fx = x1, fx1
        x1, fx1 = x_new, eval_f(x_new)
        
        if abs(fx1) <= 1e-8:
            return x1
        if abs(fx) < abs(fx1):
            # We moved away from the root, try bisection-like damping
            x_new = 0.5 * (x + x1)
            try:
                fx1 = eval_f(x_new)
            except:
                break
            if abs(fx1) <= 1e-8:
                return x_new
    
    return None


def _brent_like(f, x0, eval_f, eval_count, deadline, best_x, best_f):
    """Brent's method-like approach with bisection fallback."""
    x = float(x0)
    fx = best_f
    
    # Try to find a bracket
    a, b = None, None
    fa, fb = None, None
    
    # Expand to find sign change
    step = 1e-2 * max(1.0, abs(x))
    for direction in [1, -1]:
        for i in range(20):
            if eval_count[0] >= 500 or time.time() > deadline:
                return None
            x_try = x + direction * step
            try:
                fx_try = eval_f(x_try)
            except:
                break
            if abs(fx_try) <= 1e-8:
                return x_try
            if fx * fx_try < 0:
                if direction > 0:
                    a, b = x, x_try
                    fa, fb = fx, fx_try
                else:
                    a, b = x_try, x
                    fa, fb = fx_try, fx
                break
            step *= 2
        if a is not None:
            break
    
    if a is None:
        # No bracket found, try secant-like approach
        return _secant_newton_hybrid(f, x0, eval_f, eval_count, deadline, best_x, best_f)
    
    # Brent's method
    c, fc = a, fa
    d, e = b - a, b - a
    tol = 1e-8
    
    for i in range(60):
        if eval_count[0] >= 500 or time.time() > deadline:
            break
        
        # Invariant: fa != fb, fb != fc, a != b
        if (fa * fb > 0) or (fb * fc > 0):
            c, fc = a, fa
            d, e = b - a, b - a
        
        if abs(d) < abs(e):
            c, fc = b, fb
        
        # Close to convergence
        m = 0.5 * (c - b)
        if abs(m) * tol < 1e-12 or fc == 0:
            break
        
        if abs(e) * tol < 1e-12 or fa == fb:
            # Bisection
            e = m
        else:
            # Inverse quadratic interpolation
            A = fb / fc
            B = fa / fc
            C = fa / fb
            if A == B:
                e = -m * 2 * C / (A + C)
            else:
                e = -m * 2 * (A - 1) * B * C / ((A - B) * (A - C))
            # Check if e is within (b, c) and not too small
            if ((b < e and e < c) or (c < e and e < b)) and abs(e) < abs(m):
                e = m if e * m > 0 else m * tol / abs(m)
        
        a, fa = b, fb
        d, e = b - a, e
        b = b + e
        try:
            fb = eval_f(b)
        except:
            break
        
        if abs(fb) <= 1e-8:
            return b
    
    return b if abs(fb) <= 1e-8 else None


def _restart_search(f, x0, eval_f, eval_count, deadline, best_x, best_f):
    """Try multiple restart points to find a root."""
    x0 = float(x0)
    candidates = [x0]
    
    # Generate candidate points around x0
    scale = max(1.0, abs(x0))
    for factor in [0.1, 0.5, 2.0, 5.0, 10.0]:
        candidates.append(x0 + factor * scale)
        candidates.append(x0 - factor * scale)
    
    # Also try points based on the function's behavior
    try:
        fx0 = eval_f(x0)
        if fx0 != 0:
            # Try moving in direction opposite to function value
            step = 0.1 * scale
            candidates.append(x0 - math.copysign(step, fx0))
            candidates.append(x0 + math.copysign(step, fx0))
    except:
        pass
    
    best_x_local = best_x
    best_f_local = best_f
    
    for xc in candidates:
        if eval_count[0] >= 500 or time.time() > deadline:
            break
        try:
            fxc = eval_f(xc)
        except:
            continue
        if abs(fxc) <= 1e-8:
            return xc
        if best_f_local is None or abs(fxc) < abs(best_f_local):
            best_f_local = fxc
            best_x_local = xc
    
    # If we found a better point, try to refine it
    if best_x_local != x0 and best_f_local is not None:
        result = _secant_newton_hybrid(f, best_x_local, eval_f, eval_count, deadline, best_x_local, best_f_local)
        if result is not None:
            try:
                fr = eval_f(result)
                if abs(fr) <= 1e-8:
                    return result
            except:
                pass
    
    return None
# EVOLVE-BLOCK-END
