import math
import time
import random


def find_root(f, x0):
    TOL = 1e-8
    TOL_POLISH = 1e-10  # tighter tolerance for polish pass
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 138.0  # safety margin before the 150s kill

    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = math.inf

    def call_f(x):
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() >= DEADLINE:
            return None
        evals += 1
        try:
            val = float(f(x))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError,
                AttributeError, ArithmeticError):
            return None
        if not math.isfinite(val):
            return None
        av = abs(val)
        if av < best_abs:
            best_abs = av
            best_x = float(x)
            best_f = val
        return val

    def is_good(x, fx):
        return fx is not None and abs(fx) <= TOL

    def is_close(x, fx):
        """Check if we're close enough to start a polish pass."""
        return fx is not None and abs(fx) <= 1e-5

    def safe_get_deriv(fx):
        try:
            d = f.deriv
            if callable(d):
                return d
        except Exception:
            pass
        try:
            d = f.derivative
            if callable(d):
                return d
        except Exception:
            pass
        return None

    def siqi(x0, x1, x2):
        """
        Safeguarded Inverse Quadratic Interpolation (SIQI) / Robust Secant.
        Uses 3 points to estimate the next step.
        If quadratic interpolation is unstable, falls back to secant.
        """
        x = float(x0)
        x1 = float(x1)
        x2 = float(x2)
        
        fx = call_f(x)
        fx1 = call_f(x1)
        fx2 = call_f(x2)
        
        # Handle failures
        if fx is None and fx1 is None and fx2 is None:
            return x, None
        if fx is None:
            x, fx = x1, fx1
            x1, fx1 = x2, fx2
        if fx1 is None:
            x1, fx1 = x2, fx2
            x2, fx2 = x, fx
        if fx2 is None:
            return x, fx
            
        if is_good(x, fx): return x, fx
        if is_good(x1, fx1): return x1, fx1
        if is_good(x2, fx2): return x2, fx2

        for _ in range(50):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            
            # Try Quadratic Interpolation
            h1 = x1 - x
            h2 = x2 - x
            valid_quad = (abs(h1) > 1e-15 and abs(h2) > 1e-15 and 
                          abs(h1 - h2) > 1e-15)
            
            step = None
            if valid_quad:
                # Coefficients for quadratic passing through (x,fx), (x1,fx1), (x2,fx2)
                # Using Lagrange basis or Newton form.
                # Let's use the form: f(x+h) = A*h^2 + B*h + fx
                # f(x1+h1) = A*h1^2 + B*h1 + fx = fx1  => A*h1^2 + B*h1 = fx1-fx
                # f(x2+h2) = A*h2^2 + B*h2 + fx = fx2  => A*h2^2 + B*h2 = fx2-fx
                # Divide by h1, h2:
                # A*h1 + B = (fx1-fx)/h1
                # A*h2 + B = (fx2-fx)/h2
                # Subtract: A*(h1-h2) = (fx1-fx)/h1 - (fx2-fx)/h2
                denom = h1 - h2
                if abs(denom) > 1e-15:
                    A = ((fx1-fx)/h1 - (fx2-fx)/h2) / denom
                    B = (fx1-fx)/h1 - A*h1
                    # Solve A*h^2 + B*h + fx = 0
                    disc = B*B - 4*A*fx
                    if disc >= 0:
                        sqrt_disc = math.sqrt(disc)
                        # Two roots: (-B +/- sqrt_disc) / (2*A)
                        # We want the root that is smallest in magnitude or closest to previous?
                        # Usually SIQI picks the one that keeps us in the bracket or is smallest.
                        h1_root = (-B + sqrt_disc) / (2*A)
                        h2_root = (-B - sqrt_disc) / (2*A)
                        
                        # Heuristic: pick the step with smaller magnitude to ensure convergence
                        # Or pick the one that doesn't jump too far.
                        if abs(h1_root) < abs(h2_root):
                            step = h1_root
                        else:
                            step = h2_root
                        
                        if not math.isfinite(step):
                            step = None
            else:
                step = None
            
            # Fallback to Secant if Quadratic failed or is too large
            if step is None:
                # Use secant on the two closest points to x? Or just x and x1?
                # Let's use secant on x and x1 (most recent)
                if fx == fx1:
                    break
                step = -fx * (x - x1) / (fx - fx1)
                if not math.isfinite(step):
                    break
            
            # Damp large steps
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            xn = x + step
            if not math.isfinite(xn):
                break
                
            fxn = call_f(xn)
            if fxn is None:
                # If evaluation failed, maybe shrink step? Or break?
                # Let's break to avoid wasting evals on bad regions
                break
                
            if is_good(xn, fxn):
                return xn, fxn
                
            # Update points: shift x0 -> x1 -> x2 -> xn
            x, fx = xn, fxn
            x1, fx1 = x2, fx2
            x2, fx2 = xn, fxn
            
        # Return best of the 3
        candidates = [(x, fx), (x1, fx1), (x2, fx2)]
        valid_cands = [c for c in candidates if c[1] is not None]
        if valid_cands:
            best_c = min(valid_cands, key=lambda c: abs(c[1]))
            return best_c
        return x, fx

    x0 = float(x0)
    
    # Initial evaluation
    fx0 = call_f(x0)
    if is_good(x0, fx0):
        return x0

    # Multi-start strategy
    # We will try a few random restarts.
    # Scale of exploration: log-uniform from 1e-8 to 100 * |x0| (or 1 if x0=0)
    base_scale = max(1.0, abs(x0))
    
    # Number of restarts limited by budget
    # Each restart uses ~20-30 evals.
    # We have 500 evals. So ~10-20 restarts.
    # But we also want to prioritize local convergence first.
    
    # 1. Try local convergence from x0 with SIQI
    # We need 3 points for SIQI.
    # Use x0, x0 + small, x0 - small
    h_init = 1e-6 * base_scale
    x_sol, fx_sol = siqi(x0, x0 + h_init, x0 - h_init)
    if is_good(x_sol, fx_sol):
        return x_sol

    # If not good, but close, try Newton polish? 
    # The directive suggests SIQI. Let's stick to SIQI but maybe with tighter start.
    if is_close(x_sol, fx_sol):
        # Try a very tight SIQI
        h_tight = 1e-10 * base_scale
        x_sol2, fx_sol2 = siqi(x_sol, x_sol + h_tight, x_sol - h_tight)
        if is_good(x_sol2, fx_sol2):
            return x_sol2

    # 2. Random Restarts
    # Use random module seeded deterministically? 
    # The task implies deterministic behavior is preferred for reproducibility, 
    # but "stochastic global optimization" implies randomness.
    # However, using a fixed seed ensures the "random" choices are consistent 
    # across runs, which is good for evaluation stability.
    rng = random.Random(42)
    
    max_restarts = 20
    for i in range(max_restarts):
        if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 10:
            break
            
        # Choose a random offset
        # Log-uniform distribution for the offset magnitude
        # Range: 1e-8 to 1e2 * base_scale
        log_min = -8
        log_max = 2
        log_off = rng.uniform(log_min, log_max)
        off_mag = (10 ** log_off) * base_scale
        
        # Random sign
        sign = 1 if rng.random() < 0.5 else -1
        x_start = x0 + sign * off_mag
        
        # Run SIQI from x_start
        # Need 3 points.
        h_local = 1e-4 * max(1.0, abs(x_start))
        # Ensure distinct points
        pts = [x_start, x_start + h_local, x_start - h_local]
        
        x_r, fx_r = siqi(pts[0], pts[1], pts[2])
        
        if is_good(x_r, fx_r):
            return x_r
            
        # If close, try polish
        if is_close(x_r, fx_r):
            h_tight = 1e-10 * max(1.0, abs(x_r))
            x_r2, fx_r2 = siqi(x_r, x_r + h_tight, x_r - h_tight)
            if is_good(x_r2, fx_r2):
                return x_r2

    # Final fallback: return best found so far
    if best_f is not None:
        return best_x
        
    return x0
