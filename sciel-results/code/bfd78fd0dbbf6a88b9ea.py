"""Root finding via derivative-free minimization of f(x)^2 using Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses Nelder-Mead simplex method to minimize g(x) = f(x)^2.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    max_evals = 490  # Safety margin for 500 limit
    
    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]
    
    def g(x):
        nonlocal eval_count
        if eval_count[0] >= max_evals:
            return float('inf')
        if time.time() > deadline:
            return float('inf')
        eval_count[0] += 1
        try:
            fx = f(x)
        except Exception:
            return float('inf')
        if not math.isfinite(fx):
            return float('inf')
        val = fx * fx
        if val < best_f[0]:
            best_f[0] = val
            best_x[0] = float(x)
        return val
    
    # Initial simplex
    x0 = float(x0)
    h = 1e-4 * max(1.0, abs(x0))
    
    # Create initial simplex points
    simplex = [
        x0,
        x0 + h,
        x0 - h,
    ]
    
    # Evaluate initial simplex
    values = [g(x) for x in simplex]
    
    # Sort simplex by function value
    def sort_simplex(simplex, values):
        pairs = list(zip(values, simplex))
        pairs.sort(key=lambda p: p[0])
        return [p[1] for p in pairs], [p[0] for p in pairs]
    
    simplex, values = sort_simplex(simplex, values)
    
    # Nelder-Mead parameters
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    # Main iteration loop
    max_iters = 300
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if eval_count[0] >= max_evals:
            break
        
        # Check convergence
        if abs(math.sqrt(values[0])) <= 1e-8:
            break
        
        # Centroid of all points except the worst
        centroid = sum(simplex[:-1]) / (len(simplex) - 1)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = g(x_r)
        
        if f_r < values[0]:
            # Expansion
            x_e = centroid + gamma * (centroid - simplex[-1])
            f_e = g(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            # Reflection accepted
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            if f_r < values[-1]:
                # Outside contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
            else:
                # Inside contraction
                x_c = centroid - rho * (simplex[-1] - centroid)
            f_c = g(x_c)
            
            if f_c < min(f_r, values[-1]):
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = g(simplex[i])
        
        # Re-sort
        simplex, values = sort_simplex(simplex, values)
        
        # Check if we've converged on the best value
        if values[0] <= 1e-16:
            break
    
    # Return the best x found
    return best_x[0]
# EVOLVE-BLOCK-END
