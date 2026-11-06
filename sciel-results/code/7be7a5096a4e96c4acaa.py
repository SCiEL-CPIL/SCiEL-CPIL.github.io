"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using trust-region minimization of |f(x)|.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    def obj(x):
        """Objective: minimize |f(x)|"""
        try:
            val = call_f(x)
            return abs(val)
        except Exception:
            return float('inf')
    
    # Track best point found
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except Exception:
        return best_x
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is not None and (best_fx is None or abs(fx) < abs(best_fx)):
            best_x = x
            best_fx = fx
    
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x
    
    x = float(x0)
    
    # Phase 1: Try to find a bracket using expansion (quick win if possible)
    a, b, fa, fb = None, None, None, None
    if best_fx is not None:
        a, fa = x, best_fx
        step = max(1e-3, abs(x) * 1e-3)
        for i in range(20):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            direction = 1 if (i % 2 == 0) else -1
            b = a + direction * step * (2 ** i)
            try:
                fb = call_f(b)
                update_best(b, fb)
                if abs(fb) <= 1e-8:
                    return b
                if fa * fb < 0:
                    break
                a, fa = b, fb
            except Exception:
                break
        
        if fa is not None and fb is not None and fa * fb < 0:
            # Bisection to tighten bracket
            for _ in range(80):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                mid = (a + b) / 2.0
                try:
                    fm = call_f(mid)
                    update_best(mid, fm)
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                    if abs(b - a) < 1e-12 * max(1.0, abs(a)):
                        break
                except Exception:
                    break
            # Return the better endpoint
            if abs(fa) <= abs(fb):
                return a
            else:
                return b
    
    # Phase 2: Trust-region minimization of |f(x)|
    # Start with initial simplex (2 points in 1D)
    scale = max(1.0, abs(x))
    simplex = [x, x + 1e-3 * scale]
    
    try:
        simplex_vals = [call_f(simplex[0]), call_f(simplex[1])]
        for i, v in enumerate(simplex_vals):
            update_best(simplex[i], v)
    except Exception:
        # Fallback: return best found so far
        return best_x
    
    # Trust region radius
    trust_radius = 1e-2 * scale
    
    for iteration in range(300):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        if best_fx is not None and abs(best_fx) <= 1e-8:
            return best_x
        
        # Find the point with smallest |f(x)| in the simplex
        min_idx = 0
        min_val = abs(simplex_vals[0])
        for i in range(1, len(simplex_vals)):
            if abs(simplex_vals[i]) < min_val:
                min_val = abs(simplex_vals[i])
                min_idx = i
        
        best_point = simplex[min_idx]
        best_val = min_val
        
        # Generate trial point: move away from the worst point toward the best
        # In 1D with 2 points, this is straightforward
        if len(simplex) == 2:
            # The "other" point is the one that's not best
            other_idx = 1 - min_idx
            other_point = simplex[other_idx]
            other_val = simplex_vals[other_idx]
            
            # Direction: from worst to best
            direction = best_point - other_point
            
            # Try reflection: extend beyond best point
            # trial = best + alpha * direction
            # Choose alpha based on trust region
            trial = best_point + direction * 0.5
            
            # Check if trial is within trust region
            if abs(trial - best_point) > trust_radius:
                trial = best_point + math.copysign(trust_radius, direction)
            
            # Evaluate trial point
            try:
                trial_val = call_f(trial)
                update_best(trial, trial_val)
            except Exception:
                break
            
            if abs(trial_val) <= 1e-8:
                return trial
            
            # Decide whether to accept
            if abs(trial_val) < best_val:
                # Accept: move best point to trial
                simplex[min_idx] = trial
                simplex_vals[min_idx] = trial_val
                # Shrink trust region slightly on success
                trust_radius *= 0.7
            else:
                # Reject: try opposite direction
                trial2 = best_point - direction * 0.5
                if abs(trial2 - best_point) > trust_radius:
                    trial2 = best_point - math.copysign(trust_radius, abs(direction))
                
                try:
                    trial2_val = call_f(trial2)
                    update_best(trial2, trial2_val)
                except Exception:
                    break
                
                if abs(trial2_val) <= 1e-8:
                    return trial2
                
                if abs(trial2_val) < best_val:
                    simplex[min_idx] = trial2
                    simplex_vals[min_idx] = trial2_val
                    trust_radius *= 0.7
                else:
                    # Both failed: shrink trust region and try smaller step
                    trust_radius *= 0.5
                    if trust_radius < 1e-12 * scale:
                        # Very small trust region: try direct small steps
                        h = trust_radius
                        improved = False
                        for d in [1, -1]:
                            x_try = best_point + d * h
                            try:
                                fx_try = call_f(x_try)
                                update_best(x_try, fx_try)
                                if abs(fx_try) <= 1e-8:
                                    return x_try
                                if abs(fx_try) < best_val:
                                    simplex[min_idx] = x_try
                                    simplex_vals[min_idx] = fx_try
                                    improved = True
                            except Exception:
                                pass
                        if not improved:
                            h *= 0.5
                            if h < 1e-14 * scale:
                                break
                    # Update the other simplex point to be closer
                    simplex[other_idx] = best_point + direction * 0.25
                    try:
                        simplex_vals[other_idx] = call_f(simplex[other_idx])
                        update_best(simplex[other_idx], simplex_vals[other_idx])
                    except Exception:
                        break
        
        # Check convergence
        if len(simplex) == 2:
            if abs(simplex[0] - simplex[1]) < 1e-12 * max(1.0, abs(simplex[0])):
                break
    
    # Phase 3: Final polish with secant if we're close
    if best_fx is not None and abs(best_fx) < 1e-4:
        x_local = best_x
        fx_local = best_fx
        x_local_prev = x_local - 1e-7 * max(1.0, abs(x_local))
        try:
            fx_local_prev = call_f(x_local_prev)
            update_best(x_local_prev, fx_local_prev)
        except Exception:
            fx_local_prev = None
        
        for _ in range(60):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            if abs(fx_local) <= 1e-8:
                return x_local
            
            # Secant step
            if fx_local_prev is not None and abs(fx_local - fx_local_prev) > 1e-30:
                secant_x = x_local - fx_local * (x_local - x_local_prev) / (fx_local - fx_local_prev)
                if math.isfinite(secant_x):
                    try:
                        fx_try = call_f(secant_x)
                        update_best(secant_x, fx_try)
                        if abs(fx_try) <= 1e-8:
                            return secant_x
                        if abs(fx_try) < abs(fx_local) * 0.5:
                            x_local_prev = x_local
                            fx_local_prev = fx_local
                            x_local = secant_x
                            fx_local = fx_try
                            continue
                    except Exception:
                        pass
            
            # Small steps
            h = 1e-8 * max(1.0, abs(x_local))
            improved = False
            for direction in [1, -1]:
                x_try = x_local + direction * h
                try:
                    fx_try = call_f(x_try)
                    update_best(x_try, fx_try)
                    if abs(fx_try) <= 1e-8:
                        return x_try
                    if abs(fx_try) < abs(fx_local) * 0.5:
                        x_local = x_try
                        fx_local = fx_try
                        improved = True
                except Exception:
                    pass
            
            if not improved:
                h *= 0.5
            if h < 1e-14:
                break
    
    return best_x
# EVOLVE-BLOCK-END
