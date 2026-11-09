import time

def find_root(f, x0):
    start_time = time.time()
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_fx = fx
    
    # Use a hybrid approach: Newton with fallback to smaller steps
    for _ in range(100):
        if time.time() - start_time > 110:
            return best_x
            
        # Central difference for derivative (more accurate, 2 calls)
        h = 1e-5 * max(1.0, abs(x))
        fx_plus = f(x + h)
        fx_minus = f(x - h)
        df = (fx_plus - fx_minus) / (2 * h)
        
        if df == 0 or not (df == df): # Check for zero or NaN
            # Fallback: try a small random step or just break
            break
            
        x_new = x - fx / df
        
        if not (x_new == x_new): # NaN check
            break
            
        fx_new = f(x_new)
        
        if abs(fx_new) < abs(fx):
            x, fx = x_new, fx_new
            if abs(fx) <= 1e-8:
                return x
        else:
            # Newton step increased error. Try a smaller step (backtracking)
            x_new = x - 0.5 * fx / df
            fx_new = f(x_new)
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
                if abs(fx) <= 1e-8:
                    return x
            else:
                # Still diverging. Try a very small step or break
                # If we keep failing, we might be at a local extremum or bad derivative
                # For polynomials, this is rare unless x0 is very far.
                break
                
    # If we exited the loop, return the best x found
    return best_x
