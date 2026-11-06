"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def find_root(f, x0):
    """
    Find root of f near x0 using an adaptive global search strategy.
    Maintains a dynamic set of candidate points, prioritizes evaluation
    based on potential improvement, and switches to secant polishing
    when close to a root.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Objective function: minimize |f(x)|
    def objective(x):
        try:
            val = call_f(x)
            if not math.isfinite(val):
                return float('inf')
            return abs(val)
        except:
            return float('inf')
    
    x0 = float(x0)
    
    # Initial evaluation
    best_x = x0
    best_fx = objective(x0)
    
    if best_fx <= 1e-8:
        return best_x
    
    # Adaptive global search with diversity-based candidate selection
    # Maintain a set of candidate points with their function values
    candidates = [(x0, best_fx)]
    
    # Strategy:
    # 1. Start with a few exploratory points around x0
    # 2. Iteratively select the most promising next point to evaluate
    # 3. Use diversity and function value to guide selection
    # 4. When close to a root, switch to secant method for fast convergence
    
    # Initial exploration: try points at various scales
    scales = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    for scale in scales:
        if time_left() < 2 or eval_count >= MAX_EVALS - 5:
            break
        for sign in [1.0, -1.0]:
            if eval_count >= MAX_EVALS - 5:
                break
            x_try = x0 + sign * scale * max(1.0, abs(x0))
            if not math.isfinite(x_try):
                continue
            fx_try = objective(x_try)
            candidates.append((x_try, fx_try))
            if fx_try < best_fx:
                best_x = x_try
                best_fx = fx_try
                if best_fx <= 1e-8:
                    return best_x
    
    # If we're already very close, try secant polishing
    if best_fx < 1e-6 and eval_count < MAX_EVALS - 20 and time_left() > 2:
        result = secant_polish(f, best_x, call_f, objective, max_iter=30)
        if result is not None:
            x_sec, fx_sec = result
            if fx_sec < best_fx:
                best_x = x_sec
                best_fx = fx_sec
                if best_fx <= 1e-8:
                    return best_x
    
    # Main adaptive search loop
    # Use a combination of local refinement and global exploration
    for iteration in range(100):  # Cap iterations to avoid infinite loop
        if time_left() < 1.0 or eval_count >= MAX_EVALS - 5:
            break
        
        # If we found a good candidate, try to refine it
        if best_fx < 1e-4:
            # Try secant polishing from the best point
            if eval_count < MAX_EVALS - 10 and time_left() > 2:
                result = secant_polish(f, best_x, call_f, objective, max_iter=20)
                if result is not None:
                    x_sec, fx_sec = result
                    if fx_sec < best_fx:
                        best_x = x_sec
                        best_fx = fx_sec
                        if best_fx <= 1e-8:
                            return best_x
                        # Update candidates
                        candidates.append((x_sec, fx_sec))
                    # If secant converged well, we can stop
                    if fx_sec < 1e-10:
                        break
            else:
                break
        
        # Generate new candidate points
        new_candidates = []
        
        # 1. Local refinement around the best point
        h_local = 1e-4 * max(1.0, abs(best_x))
        for sign in [1.0, -1.0]:
            x_new = best_x + sign * h_local
            if math.isfinite(x_new):
                new_candidates.append((x_new, 'local'))
        
        # 2. Interpolation-based candidates (if we have at least 2 points)
        if len(candidates) >= 2:
            # Sort by function value
            sorted_cands = sorted(candidates, key=lambda c: c[1])
            # Try points between the two best
            if sorted_cands[0][1] < sorted_cands[1][1]:
                x1, f1 = sorted_cands[0]
                x2, f2 = sorted_cands[1]
                if x1 != x2:
                    # Linear interpolation
                    t = f1 / (f1 - f2) if f1 != f2 else 0.5
                    x_interp = x1 + t * (x2 - x1)
                    if math.isfinite(x_interp):
                        new_candidates.append((x_interp, 'interp'))
        
        # 3. Global exploration: random points in a reasonable range
        range_scale = max(1.0, abs(x0)) * 10
        for _ in range(2):
            x_rand = x0 + random.uniform(-range_scale, range_scale)
            if math.isfinite(x_rand):
                new_candidates.append((x_rand, 'global'))
        
        # 4. Diversity-based selection: pick points far from existing candidates
        # Score each new candidate by a combination of novelty and proximity to best
        scored_candidates = []
        for x_cand, cand_type in new_candidates:
            # Compute minimum distance to existing candidates
            min_dist = float('inf')
            for x_exist, _ in candidates:
                dist = abs(x_cand - x_exist)
                if dist < min_dist:
                    min_dist = dist
            
            # Novelty score (higher is better)
            novelty = min_dist / max(1.0, abs(x_cand))
            
            # Proximity score (lower is better, so invert)
            prox = 1.0 / (1.0 + abs(x_cand - best_x))
            
            # Combined score
            score = 0.3 * novelty + 0.7 * prox
            scored_candidates.append((x_cand, score, cand_type))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda c: c[1], reverse=True)
        
        # Evaluate the top candidate
        if scored_candidates:
            x_next = scored_candidates[0][0]
            fx_next = objective(x_next)
            candidates.append((x_next, fx_next))
            
            if fx_next < best_fx:
                best_x = x_next
                best_fx = fx_next
                if best_fx <= 1e-8:
                    return best_x
            
            # If we found a sign change, we can bracket the root
            # Check for sign changes among recent candidates
            recent = candidates[-10:] if len(candidates) >= 10 else candidates
            for i in range(len(recent) - 1):
                x1, f1 = recent[i]
                x2, f2 = recent[i + 1]
                if f1 * f2 < 0:  # Sign change
                    # Use bisection to narrow down
                    if eval_count < MAX_EVALS - 10 and time_left() > 2:
                        result = bisection_bracket(f, x1, x2, call_f, objective, max_iter=20)
                        if result is not None:
                            x_bis, fx_bis = result
                            if fx_bis < best_fx:
                                best_x = x_bis
                                best_fx = fx_bis
                                if best_fx <= 1e-8:
                                    return best_x
                    break
    
    # Final polish: try secant from the best point
    if best_fx < 1e-4 and eval_count < MAX_EVALS - 10 and time_left() > 2:
        result = secant_polish(f, best_x, call_f, objective, max_iter=30)
        if result is not None:
            x_sec, fx_sec = result
            if fx_sec < best_fx:
                best_x = x_sec
                best_fx = fx_sec
    
    return best_x


def secant_polish(f, x0, call_f, objective, max_iter=50):
    """
    Secant method polishing. Returns (x, |f(x)|) or None if it fails.
    """
    try:
        x1 = x0
        h = 1e-6 * max(1.0, abs(x1))
        x2 = x1 + h
        
        f1 = call_f(x1)
        f2 = call_f(x2)
        
        if not math.isfinite(f1) or not math.isfinite(f2):
            return None
        
        best_x = x1 if abs(f1) < abs(f2) else x2
        best_fx = min(abs(f1), abs(f2))
        
        if best_fx <= 1e-8:
            return (best_x, best_fx)
        
        for _ in range(max_iter):
            if abs(f1 - f2) < 1e-15:
                break
            
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            
            if not math.isfinite(x3):
                break
            
            f3 = call_f(x3)
            
            if not math.isfinite(f3):
                break
            
            if abs(f3) < best_fx:
                best_x = x3
                best_fx = abs(f3)
                if best_fx <= 1e-8:
                    return (best_x, best_fx)
            
            # Update points
            x1 = x2
            f1 = f2
            x2 = x3
            f2 = f3
            
            # Check for stagnation
            if abs(x3 - x2) < 1e-12 * max(1.0, abs(x3)):
                break
        
        return (best_x, best_fx)
    
    except:
        return None


def bisection_bracket(f, a, b, call_f, objective, max_iter=50):
    """
    Bisection method for a bracketed root. Returns (x, |f(x)|) or None if it fails.
    """
    try:
        fa = call_f(a)
        fb = call_f(b)
        
        if not math.isfinite(fa) or not math.isfinite(fb):
            return None
        
        if fa * fb > 0:
            return None  # No sign change
        
        best_x = a if abs(fa) < abs(fb) else b
        best_fx = min(abs(fa), abs(fb))
        
        for _ in range(max_iter):
            c = (a + b) / 2.0
            fc = call_f(c)
            
            if not math.isfinite(fc):
                break
            
            if abs(fc) < best_fx:
                best_x = c
                best_fx = abs(fc)
                if best_fx <= 1e-8:
                    return (best_x, best_fx)
            
            if fa * fc <= 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            if abs(b - a) < 1e-12 * max(1.0, abs(c)):
                break
        
        return (best_x, best_fx)
    
    except:
        return None

# EVOLVE-BLOCK-END
