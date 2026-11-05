"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    # The evaluator kills after 150s. We use 145s to be safe.
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isfinite(val) and abs(val) < best_abs:
            best_abs = abs(val)
            best_x = x
        return val
    
    def is_good(x, fx):
        return abs(fx) <= TOL
    
    def illinois_bracket(x0):
        """
        Attempts to find a bracket [a, b] such that f(a)*f(b) < 0 
        using the Illinois algorithm (a variant of bisection/secant).
        Returns (a, b, fa, fb) or None.
        """
        nonlocal evals
        
        x = float(x0)
        h = 1e-4 * max(1.0, abs(x))
        
        # Try to find an initial bracket by expanding
        # We check x, x+h, x-h
        candidates = [x, x + h, x - h]
        
        # Evaluate candidates
        vals = []
        for c in candidates:
            v = call_f(c)
            vals.append((c, v))
            
        # Check for immediate root
        for c, v in vals:
            if v is not None and is_good(c, v):
                return c, c, v, v # Degenerate bracket, root found
        
        # Look for a sign change among the three points
        # Order them: x-h, x, x+h
        # If we have finite values, check pairs
        finite_points = [(c, v) for c, v in vals if v is not None and math.isfinite(v)]
        
        if len(finite_points) < 2:
            return None
            
        # Sort by x to ensure proper ordering for bracketing
        finite_points.sort(key=lambda p: p[0])
        
        # Check adjacent points for sign change
        for i in range(len(finite_points) - 1):
            x1, f1 = finite_points[i]
            x2, f2 = finite_points[i+1]
            if f1 * f2 < 0:
                return x1, x2, f1, f2
                
        # If no sign change found in the local neighborhood, 
        # try expanding the bracket
        # Use the point with the smallest |f| as the center? 
        # Or just expand from x0.
        
        # Let's try expanding from x0
        a = x0
        b = x0 + h
        fa = call_f(a)
        fb = call_f(b)
        
        if fa is None or fb is None:
             # Try other direction
             a = x0 - h
             fa = call_f(a)
             b = x0
             fb = call_f(b)
             if fa is None or fb is None:
                 return None

        if fa * fb < 0:
            return a, b, fa, fb
            
        # If we still don't have a bracket, we can't use Illinois effectively 
        # without a sign change. 
        # However, for ill-conditioned functions that don't cross zero 
        # but touch it (e.g., (x-1)^2), bisection/bracketing fails.
        # The task says "root finding", usually implying f(x)=0.
        # If the function doesn't change sign, we are in trouble for bracketing methods.
        # But "Ill-conditioned" often refers to steep gradients or narrow roots, 
        # which usually still cross zero.
        
        # If no bracket is found, return None to let fallback handle it.
        return None

    def illinois_solve(a, b, fa, fb):
        """
        Solves for root in [a, b] given f(a)*f(b) < 0 using Illinois algorithm.
        Returns x if found, else None.
        """
        nonlocal evals, best_x, best_abs
        
        # Illinois algorithm:
        # 1. If f(a)*f(b) > 0, error (shouldn't happen if called correctly)
        # 2. If |f(a)| < |f(b)|, swap a and b (so b is the "better" side, closer to root)
        # 3. Compute secant point c = b - f(b)*(b-a)/(f(b)-f(a))
        # 4. If c is not in (a, b), use midpoint c = (a+b)/2
        # 5. Evaluate f(c)
        # 6. If f(a)*f(c) < 0, new interval is [a, c]
        #    Else if f(c)*f(b) < 0, new interval is [c, b]
        # 7. If we shrunk the interval containing b (the "good" side), halve f(b)
        #    Specifically, if the new interval is [c, b], then f(b) = f(b)/2
        # 8. Repeat until |b-a| < tol or |f(b)| < tol
        
        max_iter = 100
        tol_x = 1e-12
        tol_f = TOL
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS:
                break
                
            # Ensure b is the side with smaller absolute value
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
                
            # Check convergence
            if abs(fb) <= tol_f:
                return b
            if abs(b - a) <= tol_x:
                # Return the point with smaller f
                return b if abs(fb) < abs(fa) else a
                
            # Compute secant point
            denom = fb - fa
            if denom == 0:
                c = (a + b) / 2.0
            else:
                c = b - fb * (b - a) / denom
                
            # If secant point is outside [a, b], use midpoint
            if not (a < c < b):
                c = (a + b) / 2.0
                
            fc = call_f(c)
            if fc is None:
                break
                
            if is_good(c, fc):
                return c
                
            # Update bracket
            if fa * fc < 0:
                # Root is in [a, c]
                b = c
                fb = fc
                # Since we kept 'a' (the "bad" side) and moved 'b', 
                # and 'b' was the "good" side, we are moving away from the good side?
                # No, Illinois rule:
                # "If the new interval is [a, c] (i.e., we discarded b), 
                #  then we do NOT halve f(b) because b is gone. 
                #  We keep fa and fb=fc. 
                #  Wait, the standard Illinois rule says:
                #  If the new interval is [c, b] (discarding a), then f(b) = f(b)/2.
                #  If the new interval is [a, c] (discarding b), then f(a) = f(a)/2? 
                #  Let's check standard definition.
                #  Standard: 
                #  1. If |f(a)| < |f(b)|, swap. So |f(a)| >= |f(b)|.
                #  2. c = secant.
                #  3. If f(a)f(c) < 0: b = c, fb = fc. (Interval [a, c])
                #     In this case, we discarded the 'good' side (old b). 
                #     The new 'good' side is c. The 'bad' side is a.
                #     We do NOT halve anything here in some variants, 
                #     but to maintain the property that we shrink the 'bad' side's influence,
                #     actually the rule is: 
                #     "If the new interval is [c, b], halve f(b). 
                #      If the new interval is [a, c], halve f(a)."
                #     Let's verify. 
                #     If we discard a (interval becomes [c, b]), a was the 'bad' side (large f). 
                #     We want to prevent the secant from being pulled too far by the large f(a).
                #     So we halve f(a) for the NEXT iteration? 
                #     No, a is gone. 
                #     The rule is: 
                #     "If the new interval is [c, b] (so a is discarded), set f(a) = f(a)/2? 
                #      No, a is not in the interval. 
                #      The rule is: 
                #      If the new interval is [c, b], then the point b is still there. 
                #      If the new interval is [a, c], then the point a is still there.
                #      
                #      Correct Illinois Algorithm:
                #      1. Ensure |f(a)| >= |f(b)|.
                #      2. Compute c.
                #      3. If f(a)*f(c) < 0: 
                #           b = c, fb = fc.
                #           a = a, fa = fa / 2.  (Halve the 'bad' side f(a))
                #      4. Else (f(c)*f(b) < 0):
                #           a = c, fa = fc.
                #           b = b, fb = fb / 2. (Halve the 'good' side f(b)?? No.)
                #      
                #      Let's re-read a reliable source mentally.
                #      Illinois method:
                #      - Start with [a,b].
                #      - If |f(a)| < |f(b)|, swap a,b. (So a is 'bad', b is 'good').
                #      - c = b - f(b)(b-a)/(f(b)-f(a)).
                #      - If c not in (a,b), c = (a+b)/2.
                #      - If f(a)f(c) < 0:
                #          b = c, fb = fc.
                #          fa = fa / 2.
                #      - Else:
                #          a = c, fa = fc.
                #          fb = fb / 2.
                #      
                #      Yes, this is the standard formulation. It halves the function value 
                #      of the point that remains in the interval from the previous "bad" side? 
                #      Actually, it halves the value of the endpoint that is NOT the new secant point?
                #      No, it halves the value of the endpoint that was the "bad" one in the 
                #      previous step if it is still there?
                #      
                #      Let's stick to:
                #      If new interval is [a, c] (b discarded): fa = fa / 2.
                #      If new interval is [c, b] (a discarded): fb = fb / 2.
                #      This ensures that the "bad" side's value is reduced if it persists,
                #      or the "good" side's value is reduced if the other is discarded?
                #      Actually, if a is discarded, b remains. b was the "good" side. 
                #      Why halve f(b)? 
                #      It prevents the secant from getting stuck if the function is flat 
                #      on the good side. 
                #      
                #      Let's implement this standard rule.
                pass 
            else:
                # Root is in [c, b]
                a = c
                fa = fc
                fb = fb / 2.0
                
        return None

    # 1. Try to find a bracket
    bracket = illinois_bracket(x0)
    
    if bracket:
        a, b, fa, fb = bracket
        # Check if we already have a root
        if is_good(a, fa):
            return a
        if is_good(b, fb):
            return b
            
        # Solve using Illinois
        result = illinois_solve(a, b, fa, fb)
        if result is not None:
            return result
            
    # Fallback: If bracketing failed (e.g., no sign change), 
    # try a simple local minimization or just return best_x.
    # For ill-conditioned roots that touch zero without crossing, 
    # we might need a different approach, but the task implies 
    # standard root finding. 
    # Let's try a few random perturbations to find a sign change if we didn't.
    
    # Quick retry with different step sizes
    for h_mult in [1e-3, 1e-2, 1e-1, 1.0]:
        h = h_mult * max(1.0, abs(x0))
        x1 = x0 + h
        x2 = x0 - h
        f1 = call_f(x1)
        f2 = call_f(x2)
        
        if f1 is None or f2 is None:
            continue
            
        if is_good(x1, f1):
            return x1
        if is_good(x2, f2):
            return x2
            
        # Check sign change with x0
        f0 = call_f(x0)
        if f0 is None:
            continue
            
        if f0 * f1 < 0:
            res = illinois_solve(x0, x1, f0, f1)
            if res is not None:
                return res
        if f0 * f2 < 0:
            res = illinois_solve(x2, x0, f2, f0)
            if res is not None:
                return res

    return best_x
# EVOLVE-BLOCK-END
