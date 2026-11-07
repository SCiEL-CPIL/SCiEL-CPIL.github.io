import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline for 120s limit, safety margin
    
    x = float(x0)
    calls = 0
    
    def eval_f(x_val):
        nonlocal calls
        calls += 1
        val = f(x_val)
        if not math.isfinite(val):
            return float('nan')
        return val

    # Initial evaluation
    fx0 = eval_f(x)
    if math.isnan(fx0):
        return x
    if abs(fx0) <= 1e-8:
        return x
    
    best_x = x
    best_fx = abs(fx0)
    
    # Bracketing search
    # We will search in both directions
    # State: (left_bound, left_val, right_bound, right_val)
    # We maintain a window [L, R] around x0 that we are expanding
    
    # Initial window
    L = x
    R = x
    fL = fx0
    fR = fx0
    
    # We will try to find a sign change
    # Strategy: expand step size
    h = 0.1
    max_h = 1e6
    
    # Keep track of previous points to check sign changes in intervals
    # prev_left, prev_right
    prev_L = x
    prev_R = x
    prev_fL = fx0
    prev_fR = fx0
    
    bracket_found = False
    a, b = 0.0, 0.0
    fa, fb = 0.0, 0.0
    
    # We need to be careful not to use too many calls.
    # Each expansion step uses 2 calls (one left, one right).
    # Max 100 steps -> 200 calls.
    
    for i in range(100):
        if time.time() > deadline:
            break
            
        new_L = L - h
        new_R = R + h
        
        f_new_L = eval_f(new_L)
        f_new_R = eval_f(new_R)
        
        if not math.isnan(f_new_L):
            if abs(f_new_L) < best_fx:
                best_fx = abs(f_new_L)
                best_x = new_L
            if abs(f_new_L) <= 1e-8:
                return new_L
        if not math.isnan(f_new_R):
            if abs(f_new_R) < best_fx:
                best_fx = abs(f_new_R)
                best_x = new_R
            if abs(f_new_R) <= 1e-8:
                return new_R
            
        # Check sign change on left side: [new_L, L]
        if not math.isnan(f_new_L) and not math.isnan(fL):
            if f_new_L * fL < 0:
                a, b = new_L, L
                fa, fb = f_new_L, fL
                bracket_found = True
                break
            elif f_new_L == 0:
                return new_L
            elif fL == 0:
                return L
                
        # Check sign change on right side: [R, new_R]
        if not math.isnan(f_new_R) and not math.isnan(fR):
            if f_new_R * fR < 0:
                a, b = R, new_R
                fa, fb = fR, f_new_R
                bracket_found = True
                break
            elif f_new_R == 0:
                return new_R
            elif fR == 0:
                return R
            
        # Update bounds
        L, R = new_L, new_R
        fL, fR = f_new_L, f_new_R
        
        # Expand step
        h *= 2.0
        if h > max_h:
            break

    if bracket_found:
        # Refine using Illinois method
        # a, b are the bracket endpoints
        # fa, fb are the function values
        for j in range(100):
            if time.time() > deadline:
                break
                
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
                
            # Illinois step
            # c = (a*|fb| + b*|fa|) / (|fa| + |fb|)
            denom = abs(fa) + abs(fb)
            if denom == 0:
                c = (a + b) / 2.0
            else:
                c = (a * abs(fb) + b * abs(fa)) / denom
                
            fc = eval_f(c)
            if math.isnan(fc):
                # Fall back to bisection
                c = (a + b) / 2.0
                fc = eval_f(c)
                if math.isnan(fc):
                    break
                    
            if abs(fc) <= 1e-8:
                return c
                
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
                
            # If interval is very small, break
            if abs(b - a) < 1e-12:
                break
                
        # Return the one with smaller absolute value
        if abs(fa) < abs(fb):
            return a
        else:
            return b
    else:
        # No bracket found, return best guess
        return best_x

# EVOLVE-BLOCK-START
# (The above code is the implementation)
# EVOLVE-BLOCK-END
