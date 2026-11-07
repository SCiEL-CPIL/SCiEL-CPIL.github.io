import math
import time

def find_root(f, x0):
    deadline = time.time() + 115.0
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Parameters for finite difference
    h = 1e-5 * max(1.0, abs(x))
    
    # We will use Newton's method with forward difference derivative.
    # If it fails or diverges, we switch to Secant.
    
    # To switch to Secant, we need two points.
    # We have (x, fx).
    # Let's get a second point (x1, fx1) to allow Secant fallback.
    # We can use x1 = x + h, which we already evaluated for derivative?
    # No, for forward diff, we evaluate f(x+h).
    # So we have (x, fx) and (x+h, fx_h).
    
    x1 = x + h
    fx1 = f(x1)
    
    if abs(fx1) <= 1e-8:
        return x1
    
    # Now we have two points: (x, fx) and (x1, fx1).
    # We can use Secant directly, or Newton.
    # Newton step from x using derivative at x:
    # dfx = (fx1 - fx) / h
    # x_new = x - fx / dfx
    
    # Let's try Newton first.
    # But Newton uses only one point's value and derivative.
    # Secant uses two points.
    
    # Actually, if we already have two points, Secant is a natural next step.
    # And Secant is 1 call/iter.
    # Newton is 2 calls/iter.
    # But Newton converges faster.
    
    # Let's stick to Secant but with a better initial second point.
    # The current code uses h=1e-6.
    # If we use a larger h, the secant line is more representative of the curve,
    # potentially leading to better convergence in early steps.
    
    # Let's try h = 1e-3 or 1e-4.
    # For polynomials, the error in secant step is related to the second derivative.
    # A larger h might overshoot less if the function is flat, or more if it's steep.
    
    # Let's try a hybrid:
    # 1. Start with Secant using a moderate step.
    # 2. If it converges, great.
    # 3. If it stagnates or diverges, restart with a different step or use Newton.
    
    # Given the "Inspiration 2" says pure secant is better, and the current score is decent,
    # maybe the bottleneck is the "polish" phase or the initial step size.
    
    # Let's optimize the Secant method:
    # 1. Choose h adaptively.
    # 2. Use the two most recent points.
    # 3. Strict convergence check.
    
    # Let's try to reduce calls by checking convergence more frequently?
    # No, we must call f to know the value.
    
    # What if we use the **Illinois Algorithm**?
    # It's a bisection variant that uses the secant line but adjusts the interval asymmetrically.
    # It guarantees convergence but might be slower than pure Secant if Secant works.
    
    # Let's stick to Secant but improve the initial point selection.
    # If x0 is far from the root, a small h might be bad.
    # If x0 is close, a small h is good.
    
    # Let's try:
    # x1 = x0 + 1e-3 * max(1, |x0|)
    
    def secant_step(xa, fa, xb, fb):
        if fa == fb:
            return (xa + xb) / 2.0
        return xb - fb * (xb - xa) / (fb - fa)
    
    # Initial points
    x_a = x
    f_a = fx
    x_b = x + 1e-3 * max(1.0, abs(x))
    f_b = f(x_b)
    
    if abs(f_b) <= 1e-8:
        return x_b
    
    best_x = x_a if abs(f_a) < abs(f_b) else x_b
    best_f = min(abs(f_a), abs(f_b))
    
    # Main loop
    for _ in range(50):
        if time.time() > deadline:
            break
        
        x_new = secant_step(x_a, f_a, x_b, f_b)
        
        if not math.isfinite(x_new):
            # Fallback to bisection
            x_new = (x_a + x_b) / 2.0
            
        f_new = f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        if abs(f_new) < best_f:
            best_f = abs(f_new)
            best_x = x_new
            
        # Update points
        # Keep the point with smaller |f|? Or just shift?
        # Standard Secant shifts: (x_a, f_a) <- (x_b, f_b), (x_b, f_b) <- (x_new, f_new)
        # This is fast but can diverge.
        # "Memory" Secant (keeping the best point) is more robust but might be slower?
        # For smooth polynomials, standard Secant is usually fine.
        
        x_a, f_a = x_b, f_b
        x_b, f_b = x_new, f_new
        
    return best_x
