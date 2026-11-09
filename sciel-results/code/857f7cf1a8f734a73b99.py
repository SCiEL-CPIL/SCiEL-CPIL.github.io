"""
Global root finder using Nelder-Mead minimization of squared residual.
Strategy: Minimize g(x) = f(x)^2. Nelder-Mead is derivative-free and robust,
handling multi-root polynomials by exploring the landscape broadly.
"""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # Safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using Nelder-Mead minimization of f(x)^2.
    Returns the x with minimum |f(x)| found within the call budget.
    """
    tol = 1e-8
    max_calls = 480  # Stay well under 500
    
    call_count = [0]
    
    def counted_eval(x):
        call_count[0] += 1
        if call_count[0] > max_calls:
            raise RuntimeError("call budget exceeded")
        val = f(x)
        if not _is_finite(val):
            raise ValueError("non-finite value")
        return val

    best_x = float(x0)
    best_fval = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fval
        if _is_finite(x) and abs(fx) < best_fval:
            best_x = x
            best_fval = abs(fx)

    try:
        # Objective function for Nelder-Mead: minimize f(x)^2
        # We track |f(x)| for the "best" update, but Nelder-Mead needs the squared value
        # to ensure it's minimizing towards zero from either side.
        
        # Initialize simplex
        # x0 is the starting point.
        # We need a second point to form a simplex in 1D.
        # A small step size is standard.
        h = 1e-4 * max(1.0, abs(x0))
        
        # Simplex points: x0, x1
        x_points = [float(x0), float(x0) + h]
        
        # Evaluate initial points
        f_vals = []
        for xp in x_points:
            try:
                fx = counted_eval(xp)
                f_vals.append(fx * fx) # Squared residual
                update_best(xp, fx)
                if abs(fx) <= tol:
                    return xp
            except (RuntimeError, ValueError):
                # If evaluation fails, assign high penalty
                f_vals.append(float('inf'))
                if call_count[0] > max_calls:
                    break
        
        if call_count[0] > max_calls:
            return best_x

        # Nelder-Mead Parameters
        alpha = 1.0   # Reflection coefficient
        gamma = 2.0   # Expansion coefficient
        rho = 0.5     # Contraction coefficient
        sigma = 0.5   # Shrinkage coefficient
        
        # Max iterations for Nelder-Mead
        # Each iteration can use 1-3 function evaluations.
        # Budget is ~480 calls. We already used 2.
        # Let's allow up to 150 iterations to be safe with time.
        max_iter = 150
        
        for _ in range(max_iter):
            if _time_left() < 2.0:
                break
            if call_count[0] >= max_calls - 5:
                break
            
            # Sort simplex points by objective value (f(x)^2)
            # In 1D, we have 2 points.
            # x[0] should have smaller f^2 than x[1]
            if f_vals[0] > f_vals[1]:
                x_points[0], x_points[1] = x_points[1], x_points[0]
                f_vals[0], f_vals[1] = f_vals[1], f_vals[0]
            
            x_best = x_points[0]
            x_worst = x_points[1]
            f_best = f_vals[0]
            f_worst = f_vals[1]
            
            # Check convergence
            if math.sqrt(f_best) <= tol:
                return x_best
            
            # Calculate centroid of best points (excluding worst)
            # In 1D, centroid is just the best point
            x_centroid = x_best
            
            # 1. Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            if not _is_finite(x_r):
                break
                
            try:
                f_r = counted_eval(x_r)
                f_r_sq = f_r * f_r
                update_best(x_r, f_r)
                if abs(f_r) <= tol:
                    return x_r
            except (RuntimeError, ValueError):
                f_r_sq = float('inf')
            
            if f_best <= f_r_sq < f_worst:
                # Accept reflection
                x_points[1] = x_r
                f_vals[1] = f_r_sq
            elif f_r_sq < f_best:
                # Reflection is better than best, try Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                if not _is_finite(x_e):
                    # Accept reflection if expansion is invalid
                    x_points[1] = x_r
                    f_vals[1] = f_r_sq
                    continue
                    
                try:
                    f_e = counted_eval(x_e)
                    f_e_sq = f_e * f_e
                    update_best(x_e, f_e)
                    if abs(f_e) <= tol:
                        return x_e
                except (RuntimeError, ValueError):
                    f_e_sq = float('inf')
                
                if f_e_sq < f_r_sq:
                    # Accept expansion
                    x_points[1] = x_e
                    f_vals[1] = f_e_sq
                else:
                    # Accept reflection
                    x_points[1] = x_r
                    f_vals[1] = f_r_sq
            else:
                # Reflection is worse than worst, try Contraction
                x_c = x_centroid + rho * (x_worst - x_centroid)
                if not _isfinite(x_c):
                    break
                    
                try:
                    f_c = counted_eval(x_c)
                    f_c_sq = f_c * f_c
                    update_best(x_c, f_c)
                    if abs(f_c) <= tol:
                        return x_c
                except (RuntimeError, ValueError):
                    f_c_sq = float('inf')
                
                if f_c_sq < f_worst:
                    # Accept contraction
                    x_points[1] = x_c
                    f_vals[1] = f_c_sq
                else:
                    # Shrinkage
                    # Shrink all points except the best towards the best
                    for i in range(1, 2):
                        x_points[i] = x_best + sigma * (x_points[i] - x_best)
                        try:
                            f_s = counted_eval(x_points[i])
                            f_vals[i] = f_s * f_s
                            update_best(x_points[i], f_s)
                            if abs(f_s) <= tol:
                                return x_points[i]
                        except (RuntimeError, ValueError):
                            f_vals[i] = float('inf')
                            
            # Check if simplex is too small
            if abs(x_points[1] - x_points[0]) < 1e-12:
                break

        return best_x

    except (RuntimeError, ValueError, OverflowError):
        if best_fval < float('inf') and _is_finite(best_x):
            return best_x
        return float(x0)
