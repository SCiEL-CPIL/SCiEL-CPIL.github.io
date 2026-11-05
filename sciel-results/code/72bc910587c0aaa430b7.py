"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def find_root(f, x0):
    """
    Robust root finder using a Global-Local Hybrid strategy with Bracketing.
    1. Establish a bracket [a, b] containing a root via expansion from x0.
    2. Iterate using a hybrid method:
       - Try Newton/Secant step.
       - If the step stays within the bracket and improves the solution, accept it.
       - Otherwise, fall back to Bisection (midpoint).
    3. Stop when |f(x)| <= 1e-8 or max evaluations reached.
    """
    max_evals = 500
    evals = 0
    
    def eval_f(x):
        nonlocal evals
        val = f(x)
        evals += 1
        return val

    x = float(x0)
    fx = eval_f(x)
    
    if abs(fx) <= 1e-8:
        return x

    # Phase 1: Find a bracket [a, b] such that f(a) * f(b) < 0
    # Start with a small interval around x0 and expand
    a = x
    b = x
    fa = fx
    fb = fx
    
    # We need to find two points with opposite signs.
    # Try expanding to the left and right.
    step = 1e-6 * max(1.0, abs(x))
    
    # Initialize a and b to x0, but we need distinct points.
    # Let's try x0 - step and x0 + step first.
    a = x - step
    b = x + step
    fa = eval_f(a)
    fb = eval_f(b)
    
    if not np.isfinite(fa) or not np.isfinite(fb):
        # If NaN/Inf, try different steps or directions
        # For now, if we hit non-finite, we might fail, but let's try to recover
        # If x0 itself is finite, maybe the neighborhood is bad.
        pass

    # Expand bracket if no sign change
    # We keep the interval [a, b] and expand it if f(a) and f(b) have the same sign.
    # We must ensure we don't use too many evals here.
    expansion_limit = 100
    for _ in range(expansion_limit):
        if evals > max_evals - 20: # Reserve some evals for refinement
            break
            
        if fa == 0: return a
        if fb == 0: return b
        
        if fa * fb < 0:
            break
            
        # Expand
        # If fa and fb have same sign, we need to go further.
        # Which side? Usually, the side where |f| is smaller might be closer to root,
        # but for bracketing, we just need to cross.
        # Let's expand the side with smaller |f|? Or just expand both?
        # Expanding both is safer to ensure we cover the root if it's far.
        # But if the function is monotonic, expanding the "wrong" side might be wasteful.
        # Let's expand the interval [a, b] by doubling its width and shifting.
        # Actually, a standard approach: 
        # If f(a)*f(b) > 0:
        #   If |fa| < |fb|, expand a to a - (b-a)
        #   Else expand b to b + (b-a)
        
        width = b - a
        if width == 0:
            width = 1.0
            
        if abs(fa) < abs(fb):
            new_a = a - width
            new_fa = eval_f(new_a)
            if not np.isfinite(new_fa):
                # Try other side if this fails?
                new_b = b + width
                new_fb = eval_f(new_b)
                a, b = new_a, new_b
                fa, fb = new_fa, new_fb
            else:
                a, fa = new_a, new_fa
        else:
            new_b = b + width
            new_fb = eval_f(new_b)
            if not np.isfinite(new_fb):
                new_a = a - width
                new_fa = eval_f(new_a)
                a, b = new_a, new_b
                fa, fb = new_fa, new_fb
            else:
                b, fb = new_b, new_fb
        
        if fa * fb < 0:
            break

    if fa * fb >= 0:
        # Failed to bracket. Return best guess so far.
        # We have x, a, b.
        # Check which is smallest |f|
        candidates = [(x, fx), (a, fa), (b, fb)]
        # If we have more points from expansion? We only kept current a,b.
        # Let's just return the one with min |f|
        best_x, best_f = min(candidates, key=lambda p: abs(p[1]))
        return best_x

    # Phase 2: Hybrid Refinement
    # We have [a, b] with f(a)*f(b) < 0.
    # We will iterate until |f| <= 1e-8 or max evals.
    
    # We can use Secant/Newton if we have a derivative estimate, or just Secant.
    # Secant requires two points. We have a, b.
    # However, Secant can jump out of bracket.
    # Strategy:
    # 1. Calculate Secant/Newton estimate x_new.
    # 2. If x_new is in (a, b) and |f(x_new)| < min(|fa|, |fb|), accept x_new.
    #    Update [a, b] to contain the root (i.e., one endpoint becomes x_new, the other is the one with opposite sign).
    # 3. Else, use Midpoint x_new = (a+b)/2.
    #    Update [a, b] based on sign of f(x_new).
    
    # To improve speed, we can try to use Newton if we can estimate derivative.
    # But Secant is cheaper (1 eval per step vs 1 eval for Newton if we reuse previous f, but we need f at new point anyway).
    # Actually, Secant uses f(a), f(b) to predict x_new. Then we evaluate f(x_new).
    # Newton uses f(x), f'(x). If we use finite diff for f', that's 2 extra evals.
    # Secant is generally better for this "black box" scenario unless we cache derivatives.
    # Let's stick to a robust Secant-with-Bracketing (Illinois-like but with explicit bracket check).
    
    # Current points
    x1, f1 = a, fa
    x2, f2 = b, fb
    
    # Ensure x1, x2 are ordered for clarity, though not strictly needed for math
    # We maintain invariant: f(x1) * f(x2) < 0
    
    while evals < max_evals:
        # Check convergence
        if abs(f1) <= 1e-8: return x1
        if abs(f2) <= 1e-8: return x2
        
        # Try Secant step
        # x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        denom = f2 - f1
        if abs(denom) < 1e-300:
            # Degenerate, use midpoint
            x_new = (x1 + x2) / 2
        else:
            x_new = x2 - f2 * (x2 - x1) / denom
        
        # Check if x_new is within bracket
        # Note: floating point errors might make x_new slightly outside.
        # We accept if it's "close enough" to the bracket or strictly inside.
        # For safety, let's require it to be strictly inside (x1, x2) or (x2, x1).
        
        # Determine bracket bounds
        lo = min(x1, x2)
        hi = max(x1, x2)
        
        use_secant = False
        if lo < x_new < hi:
            # It's inside. Check if it improves the "best" f value?
            # Actually, if it's inside, it will split the bracket.
            # We should evaluate it.
            use_secant = True
        else:
            # Outside. We could still evaluate it if it reduces |f| significantly?
            # No, if it's outside, it might diverge. Stick to bisection for stability.
            # However, if the function is very flat, secant might be far but correct.
            # But for ill-conditioned, stability is key.
            pass
        
        if use_secant:
            f_new = eval_f(x_new)
        else:
            x_new = (x1 + x2) / 2
            f_new = eval_f(x_new)
            
        if not np.isfinite(f_new):
            # If we hit NaN/Inf, shrink bracket aggressively?
            # Or just break and return best.
            # Let's try to recover by using midpoint of the *other* half?
            # Actually, if f_new is non-finite, the point is bad.
            # We should discard it and maybe shrink the bracket around the finite points.
            # But we don't have other finite points.
            # Let's just break.
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update bracket
        # We need to replace x1 or x2 with x_new such that sign change is preserved.
        # If f1 * f_new < 0, then root is in [x1, x_new]. So x2 becomes x_new.
        # Else, root is in [x_new, x2]. So x1 becomes x_new.
        
        if f1 * f_new < 0:
            x2, f2 = x_new, f_new
        else:
            x1, f1 = x_new, f_new
            
        # Optional: Illinois method modification
        # If we used bisection (or secant failed to improve), and we keep bisecting on the same side,
        # shrink the interval on that side to accelerate convergence.
        # Standard Illinois: if new point is on same side as previous new point, shrink that side.
        # This is complex to track simply. The simple bracketing is robust enough.
        
        # Check if interval is too small
        if abs(x2 - x1) < 1e-12:
            break

    # Return best candidate
    # We have x1, f1 and x2, f2.
    if abs(f1) <= abs(f2):
        return x1
    else:
        return x2

# EVOLVE-BLOCK-END
