"""Root finding for ill-conditioned problems using adaptive global search."""
import time
import math
import random

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using an adaptive global search strategy.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    
    Strategy:
    1. Maintain a set of candidate points.
    2. Score candidates by a mix of |f(x)| and distance to other points (diversity).
    3. Evaluate the most promising candidate.
    4. If local progress stalls, inject random perturbations to escape local minima/plateaus.
    5. If a very close point is found, switch to high-precision secant polishing.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
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

    # State
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except:
        best_fx = float('inf')
    
    if abs(best_fx) <= 1e-8:
        return best_x

    # Candidate pool: list of (x, fx)
    # We keep a history of evaluated points to avoid re-evaluating nearby points
    candidates = [(best_x, best_fx)]
    
    # Parameters for adaptive search
    initial_scale = max(1.0, abs(best_x)) * 0.1
    min_scale = 1e-12
    max_scale = max(1.0, abs(best_x)) * 10.0
    
    # Track recent progress to detect stalling
    prev_best = abs(best_fx)
    stall_count = 0
    last_improvement_time = start_time
    
    # Polishing mode flag
    polishing = False
    
    while eval_count < MAX_EVALS and check_time():
        current_best = abs(best_fx)
        
        # Check for success
        if current_best <= 1e-8:
            return best_x
            
        # Check for stalling
        if current_best < prev_best * 0.99:  # At least 1% improvement
            prev_best = current_best
            stall_count = 0
            last_improvement_time = time.time()
        else:
            stall_count += 1
            
        # If we are close, try to polish with secant
        if current_best < 1e-4 and not polishing:
            polishing = True
            
        # If polishing, use secant method with the two best distinct points
        if polishing:
            # Find two best distinct points
            points = sorted(candidates, key=lambda p: abs(p[1]))
            x1, fx1 = points[0]
            x2, fx2 = points[1] if len(points) > 1 else (x1 + 1e-6, None)
            
            if fx2 is None:
                # Need a second point
                h = 1e-6 * max(1.0, abs(x1))
                x2 = x1 + h
                try:
                    fx2 = call_f(x2)
                    candidates.append((x2, fx2))
                except:
                    break
                
            if math.isfinite(fx2) and abs(fx1 - fx2) > 1e-15:
                # Secant step
                x_new = x1 - fx1 * (x1 - x2) / (fx1 - fx2)
                try:
                    fx_new = call_f(x_new)
                except:
                    break
                    
                if math.isfinite(fx_new):
                    candidates.append((x_new, fx_new))
                    if abs(fx_new) < abs(best_fx):
                        best_x = x_new
                        best_fx = fx_new
                    # If secant is not improving, break polishing mode
                    if abs(fx_new) >= abs(best_fx) * 0.9:
                        polishing = False
                        stall_count = 0
                else:
                    break
            else:
                polishing = False
            continue

        # Global/Local Adaptive Search
        # Generate new candidate points based on current best and scale
        # Scale shrinks as we get closer, or if stalling, we might expand or randomize
        
        # Determine scale for new candidates
        if stall_count > 5:
            # Stalling: try larger jumps or random restarts near best
            scale = initial_scale * (1 + stall_count * 0.5)
            # Occasionally try a very small step to refine
            if random.random() < 0.2:
                scale = min_scale * 10
        else:
            # Normal progress: shrink scale
            scale = initial_scale * (0.5 ** stall_count)
            scale = max(min_scale, min(scale, max_scale))

        # Generate a few new candidates around best_x
        new_candidates = []
        for _ in range(3):
            # Random direction and magnitude
            dir = random.choice([-1, 1])
            mag = scale * random.uniform(0.5, 2.0)
            x_new = best_x + dir * mag
            
            # Check if this point is too close to existing candidates
            too_close = False
            for cx, _ in candidates:
                if abs(x_new - cx) < scale * 0.1:
                    too_close = True
                    break
            
            if not too_close:
                new_candidates.append(x_new)
        
        # If no new candidates generated (all too close), force a small perturbation
        if not new_candidates:
            x_new = best_x + random.uniform(-scale, scale) * 0.01
            new_candidates.append(x_new)
            
        # Evaluate the best new candidate (closest to best_x or random?)
        # Let's pick the one with the largest potential for improvement (further away but not too far)
        # Actually, let's just evaluate the first one to keep it simple, or pick random
        x_to_eval = new_candidates[0]
        
        try:
            fx_to_eval = call_f(x_to_eval)
        except:
            break
            
        if not math.isfinite(fx_to_eval):
            continue
            
        candidates.append((x_to_eval, fx_to_eval))
        
        # Update best
        if abs(fx_to_eval) < abs(best_fx):
            best_x = x_to_eval
            best_fx = fx_to_eval
            
        # Prune candidate list to keep it manageable (keep best 20)
        if len(candidates) > 20:
            candidates = sorted(candidates, key=lambda p: abs(p[1]))[:20]
            
    return best_x
# EVOLVE-BLOCK-END
