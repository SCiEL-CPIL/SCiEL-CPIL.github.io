"""Root finding: Ill-conditioned - Adaptive multi-strategy with derivative estimation."""
import math
import time

# EVOLVE-BLOG-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    
    Strategy:
    1. Quick check at x0
    2. Derivative estimation + Newton's method (fast if derivative is good)
    3. Secant method (superlinear convergence, no derivative needed)
    4. Illinois/Regula Falsi (robust bracketing)
    5. Brent's method (hybrid, most robust)
    6. Local grid search if all else fails
    """
    start_time = time.time()
    MAX_WALL = 110.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TOL = 1e-8
    X_EPS = 1e-14

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]
    history = []  # Store (x, f(x)) pairs for reuse

    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        if time.time() - start_time > MAX_WALL:
            raise RuntimeError("Time limit exceeded")
        eval_count[0] += 1
        try:
            val = f(x)
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        history.append((x, val))
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        return val

    def ok():
        return eval_count[0] < MAX_EVALS and (time.time() - start_time < MAX_WALL)

    def is_good(x, fx):
        return abs(fx) <= TOL

    # --- Newton's method with derivative estimation ---
    def newton(x_start, max_iter=50):
        """
        Newton's method with finite difference derivative estimation.
        Uses central difference for better accuracy.
        """
        if not ok():
            return None
        
        x = float(x_start)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        
        for i in range(max_iter):
            if not ok():
                break
            
            # Estimate derivative using central difference
            h = max(1e-8 * max(1.0, abs(x)), 1e-12)
            x_plus = x + h
            x_minus = x - h
            
            try:
                fp = eval_f(x_plus)
                fm = eval_f(x_minus)
            except RuntimeError:
                break
            
            if not math.isfinite(fp) or not math.isfinite(fm):
                break
            
            df = (fp - fm) / (2 * h)
            
            if abs(df) < 1e-15 or not math.isfinite(df):
                # Derivative too small, try secant instead
                break
            
            x_new = x - fx / df
            
            if not math.isfinite(x_new):
                break
            
            # Check for convergence
            if abs(x_new - x) <= X_EPS * max(1.0, abs(x_new)):
                fx_new = eval_f(x_new)
                if is_good(x_new, fx_new):
                    return x_new
                break
            
            x = x_new
            fx = eval_f(x)
            
            if is_good(x, fx):
                return x
        
        if is_good(x, fx):
            return x
        return None

    # --- Secant method ---
    def secant(xa, fa, xb, fb, max_iter=60):
        """
        Secant method: superlinear convergence, no derivative needed.
        """
        for _ in range(max_iter):
            if not ok():
                break
            if abs(fa) <= TOL:
                return xa
            if abs(fb) <= TOL:
                return xb
            if fa == fb:
                break
            if abs(xa - xb) < 1e-15 * max(1.0, abs(xa), abs(xb)):
                break
            
            d = (fa - fb) / (xa - xb)
            if d == 0 or not math.isfinite(d):
                break
            
            xc = xa - fa / d
            if not math.isfinite(xc):
                break
            
            fc = eval_f(xc)
            if is_good(xc, fc):
                return xc
            
            xa, fa = xb, fb
            xb, fb = xc, fc
        
        if abs(fa) <= TOL:
            return xa
        if abs(fb) <= TOL:
            return xb
        return None

    # --- Illinois method (modified regula falsi) ---
    def illinois(a, fa, b, fb, max_iter=100):
        """
        Illinois method: modified regula falsi that shrinks the endpoint
        with larger |f| to avoid slow convergence.
        """
        if fa * fb > 0:
            return None
        
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
        
        for _ in range(max_iter):
            if not ok():
                break
            
            # Standard regula falsi step
            if fa == fb:
                break
            
            c = a - fa * (b - a) / (fb - fa)
            
            if not math.isfinite(c):
                break
            
            if abs(c - a) <= X_EPS * max(1.0, abs(c)):
                fc = eval_f(c)
                if is_good(c, fc):
                    return c
                break
            
            fc = eval_f(c)
            
            if is_good(c, fc):
                return c
            
            # Update bracket
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
                # Illinois modification: shrink the other endpoint
                fb = fb * 0.5
            
            if abs(b - a) <= X_EPS * max(1.0, abs(b)):
                if abs(fa) <= TOL:
                    return a
                if abs(fb) <= TOL:
                    return b
                break
        
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
        return None

    # --- Brent's method ---
    def brent(a, fa, b, fb, max_iter=100):
        """
        Brent's method: combines bisection, secant, and ICI.
        Most robust bracketing method.
        """
        if fa * fb > 0:
            return None
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
        
        # Ensure |f(a)| <= |f(b)|
        if abs(fa) < abs(fb):
            a, fa = b, fb
            b, fb = a, fa
        
        c = a
        fc = fa
        d = b - a
        e = d
        
        for _ in range(max_iter):
            if not ok():
                break
            
            if abs(fb) <= TOL:
                return b
            if abs(b - a) <= X_EPS * max(1.0, abs(b)):
                return b
            
            # Try ICI, fall back to secant, then bisection
            x_new = None
            try:
                if c != a and c != b and fc != fa and fc != fb:
                    s = (b / (b - c)) * (fb / (fb - fc))
                    t = (b / (b - a)) * (fb / (fb - fa))
                    u = (c / (c - a)) * (fc / (fc - fa))
                    x_new = b + (s * t * u) / ((s + t) * (s + u))
                else:
                    if fa != fb:
                        x_new = b - fb * (b - a) / (fb - fa)
                    else:
                        x_new = 0.5 * (a + b)
            except (ZeroDivisionError, OverflowError):
                x_new = 0.5 * (a + b)
            
            if not math.isfinite(x_new):
                x_new = 0.5 * (a + b)
            
            # Check if x_new is acceptable
            cond1 = not ((min(a, b) < x_new < max(a, b)) or
                         (x_new < min(a, b)) != (x_new > max(a, b)))
            cond2 = abs(x_new - b) >= 0.5 * abs(b - c)
            cond3 = abs(x_new - b) >= X_EPS * max(1.0, abs(b))
            
            if cond1 or cond2 or cond3:
                e = d
                d = 0.5 * (b - a)
                x_new = b + d
            else:
                d = x_new - b
            
            fx_new = eval_f(x_new)
            
            if is_good(x_new, fx_new):
                return x_new
            
            if fa * fx_new < 0:
                b, fb = a, fa
            a, fa = b, fb
            b, fb = x_new, fx_new
            
            if abs(fa) < abs(fb):
                b, fb = a, fa
                a, fa = b, fb
            
            c, fc = b, fb
            if abs(fa) < abs(fb):
                c, fc = a, fa
            
            if abs(d) < X_EPS * max(1.0, abs(b)):
                break
        
        if abs(fb) <= TOL:
            return b
        if abs(fa) <= TOL:
            return a
        return None

    # --- Bracket finding ---
    def find_bracket(x_start, max_steps=30):
        """
        Find a bracket [a,b] with f(a)*f(b) < 0 using geometrically growing steps.
        """
        if not ok():
            return None
        
        fa = eval_f(x_start)
        if is_good(x_start, fa):
            return (x_start, fa, x_start, fa)
        
        base = max(1.0, abs(x_start))
        
        for i in range(max_steps):
            if not ok():
                break
            step = base * (10 ** (i - 3))
            if step < 1e-15:
                step = 1e-15
            
            # Try right
            xr = x_start + step
            fr = eval_f(xr)
            if is_good(xr, fr):
                return (xr, fr, xr, fr)
            if fa * fr < 0:
                return (x_start, fa, xr, fr)
            
            # Try left
            xl = x_start - step
            fl = eval_f(xl)
            if is_good(xl, fl):
                return (xl, fl, xl, fl)
            if fa * fl < 0:
                return (xl, fl, x_start, fa)
        
        return None

    # --- Local grid search ---
    def local_grid(x_center, radius, n_points=20):
        """
        Local grid search around x_center to find sign changes.
        """
        if not ok():
            return None
        
        if radius <= 0:
            radius = max(1e-6, 1e-3 * max(1.0, abs(x_center)))
        
        best_local = None
        best_local_f = float('inf')
        
        xs = []
        fs = []
        
        for i in range(n_points):
            if not ok():
                break
            x = x_center - radius + (2 * radius * i) / (n_points - 1)
            fx = eval_f(x)
            xs.append(x)
            fs.append(fx)
            if abs(fx) < best_local_f:
                best_local_f = abs(fx)
                best_local = x
        
        # Check for sign changes
        for i in range(len(xs) - 1):
            if fs[i] * fs[i+1] < 0:
                return (xs[i], fs[i], xs[i+1], fs[i+1])
            if abs(fs[i]) <= TOL:
                return (xs[i], fs[i], xs[i], fs[i])
            if abs(fs[i+1]) <= TOL:
                return (xs[i+1], fs[i+1], xs[i+1], fs[i+1])
        
        # No bracket found, return best point
        if best_local is not None:
            return (best_local, best_local_f, best_local, best_local_f)
        
        return None

    # === MAIN EXECUTION ===

    x0 = float(x0)
    
    # 1. Quick check at x0
    fx0 = eval_f(x0)
    if is_good(x0, fx0):
        return float(x0)
    
    # 2. Try Newton's method (fast if derivative is well-behaved)
    if ok():
        r = newton(x0, max_iter=40)
        if r is not None:
            return r
    
    # 3. Try secant from x0 with a nearby point
    if ok():
        h = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h
        fx1 = eval_f(x1)
        if is_good(x1, fx1):
            return x1
        r = secant(x0, fx0, x1, fx1, max_iter=50)
        if r is not None:
            return r
    
    # 4. Try secant with a slightly different offset
    if ok():
        h2 = 1e-3 * max(1.0, abs(x0))
        x2 = x0 + h2
        fx2 = eval_f(x2)
        if is_good(x2, fx2):
            return x2
        r = secant(x0, fx0, x2, fx2, max_iter=50)
        if r is not None:
            return r
    
    # 5. Find a bracket and use Illinois or Brent
    bracket = find_bracket(x0, max_steps=25)
    
    if bracket is not None:
        a, fa, b, fb = bracket
        if is_good(a, fa):
            return a
        if is_good(b, fb):
            return b
        if fa * fb < 0:
            # Try Illinois first (often faster for ill-conditioned problems)
            r = illinois(a, fa, b, fb, max_iter=80)
            if r is not None:
                return r
            # Fall back to Brent
            r = brent(a, fa, b, fb, max_iter=80)
            if r is not None:
                return r
        else:
            # Same sign, try secant
            r = secant(a, fa, b, fb, max_iter=50)
            if r is not None:
                return r
    
    # 6. Multi-start: try from nearby points
    if ok():
        base = max(1.0, abs(x0))
        offsets = [
            1e-5 * base, -1e-5 * base,
            1e-4 * base, -1e-4 * base,
            1e-3 * base, -1e-3 * base,
            1e-2 * base, -1e-2 * base,
        ]
        
        for off in offsets:
            if not ok():
                break
            x_try = x0 + off
            fx_try = eval_f(x_try)
            if is_good(x_try, fx_try):
                return x_try
            
            # Try Newton from this point
            r = newton(x_try, max_iter=30)
            if r is not None:
                return r
            
            # Try secant from this point
            h3 = 1e-4 * max(1.0, abs(x_try))
            x3 = x_try + h3
            fx3 = eval_f(x3)
            if is_good(x3, fx3):
                return x3
            r = secant(x_try, fx_try, x3, fx3, max_iter=40)
            if r is not None:
                return r
    
    # 7. Local grid search
    if ok():
        for radius_factor in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            if not ok():
                break
            radius = radius_factor * max(1.0, abs(x0))
            bracket = local_grid(x0, radius, n_points=15)
            if bracket is not None:
                a, fa, b, fb = bracket
                if is_good(a, fa):
                    return a
                if is_good(b, fb):
                    return b
                if fa * fb < 0:
                    r = illinois(a, fa, b, fb, max_iter=60)
                    if r is not None:
                        return r
                    r = brent(a, fa, b, fb, max_iter=60)
                    if r is not None:
                        return r
                else:
                    r = secant(a, fa, b, fb, max_iter=40)
                    if r is not None:
                        return r
    
    # 8. Final polish from best point
    if ok():
        h4 = 1e-6 * max(1.0, abs(best_x[0]))
        fx_near = eval_f(best_x[0] + h4)
        if is_good(best_x[0] + h4, fx_near):
            return best_x[0] + h4
        r = secant(best_x[0], best_f[0], best_x[0] + h4, fx_near, max_iter=30)
        if r is not None:
            return r
    
    # Final fallback
    return best_x[0]

# EVOLVE-BLOG-END
