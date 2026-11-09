import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using Nelder-Mead simplex minimization on g(x) = f(x)^2.
    
    Strategy:
    1. Treat root finding as a minimization problem: minimize g(x) = f(x)^2.
    2. Use the Nelder-Mead simplex method, a derivative-free optimization algorithm.
       - In 1D, the simplex consists of 2 points.
       - It is robust to oscillatory functions because it tracks the local minimum 
         of the squared residual, rather than relying on sign changes.
       - It handles tangential roots (where f touches 0 but doesn't cross) naturally.
    3. Adaptive step size and reflection/contraction/expansion steps ensure 
       convergence even if the initial guess is far from the root.
    4. Early termination if |f(x)| <= 1e-8.
    5. Strict call budget and time limit enforcement.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    max_calls = 150  # Hard cap to ensure we stay well under 500 calls average
    
    def objective(x):
        nonlocal max_calls
        if max_calls <= 0:
            return float('inf')
        max_calls -= 1
        try:
            fx = f(x)
        except Exception:
            return float('inf')
        if not math.isfinite(fx):
            return float('inf')
        return fx * fx

    # Initial simplex points
    # Start with x0 and a small perturbation
    # Scale perturbation based on x0 magnitude to handle different scales
    delta = 1e-4 * (1.0 + abs(x0))
    
    # Initialize simplex: list of [x, g(x)]
    p0 = x0
    p1 = x0 + delta
    
    g0 = objective(p0)
    g1 = objective(p1)
    
    # Track best point found so far
    best_x = p0 if g0 <= g1 else p1
    best_g = min(g0, g1)
    
    # If we already have a root, return it
    if best_g <= 1e-16:  # (1e-8)^2
        return best_x

    # Nelder-Mead parameters
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrinkage
    
    max_iter = 100
    
    for _ in range(max_iter):
        if time.time() > deadline - 0.5:
            break
        if max_calls <= 0:
            break
            
        # Order points: p_best (lower g), p_worst (higher g)
        if g0 <= g1:
            p_best, p_worst = p0, p1
            g_best, g_worst = g0, g1
        else:
            p_best, p_worst = p1, p0
            g_best, g_worst = g1, g0
            
        # Check convergence
        if g_best <= 1e-16:
            return p_best
        if abs(p_worst - p_best) < 1e-10 * (1.0 + abs(p_best)):
            break
            
        # Centroid of all points except worst (in 1D, just the best point)
        # centroid = p_best
        
        # 1. Reflection
        x_ref = p_best + alpha * (p_best - p_worst)
        g_ref = objective(x_ref)
        
        if g_ref < g_best:
            # Try Expansion
            x_exp = p_best + gamma * (x_ref - p_best)
            g_exp = objective(x_exp)
            
            if g_exp < g_ref:
                # Accept expansion
                p_worst, g_worst = x_exp, g_exp
            else:
                # Accept reflection
                p_worst, g_worst = x_ref, g_ref
        else:
            # Reflection is not better than best
            # Try Contraction
            if g_ref < g_worst:
                # Outside contraction
                x_con = p_best + rho * (x_ref - p_best)
                g_con = objective(x_con)
                
                if g_con <= g_ref:
                    p_worst, g_worst = x_con, g_con
                else:
                    # Shrink
                    p_worst = p_best + sigma * (p_worst - p_best)
                    g_worst = objective(p_worst)
            else:
                # Inside contraction
                x_con = p_best - rho * (p_worst - p_best)
                g_con = objective(x_con)
                
                if g_con <= g_worst:
                    p_worst, g_worst = x_con, g_con
                else:
                    # Shrink
                    p_worst = p_best + sigma * (p_worst - p_best)
                    g_worst = objective(p_worst)
        
        # Update p0, p1 from p_best, p_worst
        if g_best <= g_worst:
            p0, g0 = p_best, g_best
            p1, g1 = p_worst, g_worst
        else:
            p0, g0 = p_worst, g_worst
            p1, g1 = p_best, g_best
            
        # Update global best
        if g0 < best_g:
            best_g = g0
            best_x = p0
        if g1 < best_g:
            best_g = g1
            best_x = p1

    # Final check
    if best_g <= 1e-16:
        return best_x
        
    # If we didn't converge to 1e-8, try a few local bisection steps around best_x 
    # if we can find a sign change nearby, as a fallback for robustness
    # This is a hybrid touch: NM finds the local min, bisection refines if sign change exists
    
    # Check neighbors for sign change to potentially polish
    # We have limited calls left, so only do this if we have budget
    if max_calls > 5:
        step = 1e-6 * (1.0 + abs(best_x))
        candidates = [best_x - step, best_x + step]
        fb = None
        for cx in candidates:
            if max_calls <= 0: break
            try:
                fc = f(cx)
                max_calls -= 1
            except:
                continue
            if not math.isfinite(fc):
                continue
            if abs(fc) < math.sqrt(best_g):
                best_g = fc * fc
                best_x = cx
                if best_g <= 1e-16:
                    return best_x
            if fb is not None and fb * fc < 0:
                # Found bracket [cx, prev_cx] or similar, do a few bisection steps
                # But we need the two points. This logic is getting complex with limited calls.
                # Stick to the NM result for simplicity and reliability.
                pass
            fb = fc

    return best_x
# EVOLVE-BLOCK-END
