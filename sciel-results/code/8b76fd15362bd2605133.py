"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Powell's Method with Nelder-Mead fallback for BBOB-style f13 (Sharp Ridge).
    
    Structural Change:
    Instead of (1+1)-ES, we use Powell's Method, which adapts search directions
    to the local geometry of the function. This is particularly effective for
    sharp ridges where the function is anisotropic. If Powell stalls, we fall
    back to Nelder-Mead for local refinement or escape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, aim for 285s to be safe
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization
        # Sample a few points to find a good starting basin
        n_init = min(10, budget)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = timed_call(x)
            if fx is None:
                break
        
        if best_x is None:
            # Fallback: center
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            best_f = fx
            best_x = x.copy()
        
        x = best_x.copy()
        f = best_f

        # Phase 2: Powell's Method
        # Directions matrix
        dirs = np.eye(dim)
        
        # Parameters for 1D minimization (Golden Section)
        # We will perform a limited number of function evaluations per line search
        max_line_evals = 15
        
        def minimize_line(x_start, direction):
            """
            Perform 1D minimization along 'direction' starting from 'x_start'.
            Uses Golden Section Search.
            Returns (x_min, f_min, n_evals_used).
            """
            nonlocal calls_made, best_x, best_f
            n_evals = 0
            
            # Initial bracket
            # Try expanding step size to find a valley
            step = 0.1
            x0 = x_start.copy()
            f0 = timed_call(x0)
            n_evals += 1
            if f0 is None:
                return x_start, f, n_evals
            
            # Expand in positive direction
            x_plus = np.clip(x_start + step * direction, -5, 5)
            f_plus = timed_call(x_plus)
            n_evals += 1
            if f_plus is None:
                return x_start, f, n_evals
            
            if f_plus < f0:
                # Expand further
                a, b = x_start, x_start + step * direction
                fa, fb = f0, f_plus
                fa_old = f0
                while n_evals < max_line_evals and time.time() < deadline - 2:
                    x_new = np.clip(b + step * direction, -5, 5)
                    f_new = timed_call(x_new)
                    n_evals += 1
                    if f_new is None:
                        break
                    if f_new > fb:
                        # We have a bracket [a, b, x_new]
                        break
                    a = b
                    b = x_new
                    fa = fb
                    fb = f_new
                    step *= 1.5
                # Now bracket is [a, b] with f(b) < f(a)
                # But we might have gone too far. Let's just use the best point found so far
                # Simple approach: return the best point found in this expansion
                # However, Golden Section is better. Let's implement a simple Golden Section
                # on the interval [a, b] if we found a valley, or just return best.
                
                # To keep it simple and robust with limited budget, we will use the best point
                # found during expansion. This is a "hill climbing" step.
                # A more rigorous GS search would be better but consumes more calls.
                # Let's try a few GS steps if we have a bracket.
                
                # Actually, let's restart with a simpler strategy:
                # Try a few fixed step sizes around the current best.
                pass
            
            # Try negative direction
            x_minus = np.clip(x_start - step * direction, -5, 5)
            f_minus = timed_call(x_minus)
            n_evals += 1
            if f_minus is None:
                return x_start, f, n_evals
            
            # Determine best direction
            if f_minus < f_plus and f_minus < f0:
                # Go negative
                a, b = x_start - step * direction, x_start
                fa, fb = f_minus, f0
                # Expand
                while n_evals < max_line_evals and time.time() < deadline - 2:
                    x_new = np.clip(a - step * direction, -5, 5)
                    f_new = timed_call(x_new)
                    n_evals += 1
                    if f_new is None:
                        break
                    if f_new > fa:
                        break
                    b = a
                    a = x_new
                    fb = fa
                    fa = f_new
                    step *= 1.5
            elif f_plus < f_minus and f_plus < f0:
                # Go positive
                a, b = x_start, x_start + step * direction
                fa, fb = f0, f_plus
                while n_evals < max_line_evals and time.time() < deadline - 2:
                    x_new = np.clip(b + step * direction, -5, 5)
                    f_new = timed_call(x_new)
                    n_evals += 1
                    if f_new is None:
                        break
                    if f_new > fb:
                        break
                    a = b
                    b = x_new
                    fa = fb
                    fb = f_new
                    step *= 1.5
            else:
                # No improvement, return current
                return x_start, f0, n_evals
            
            # We have a bracket [a, b] where f(a) and f(b) are lower than center.
            # Let's perform a few Golden Section iterations to refine.
            gr = (np.sqrt(5) - 1) / 2
            x1 = np.clip(b - gr * (b - a), -5, 5)
            x2 = np.clip(a + gr * (b - a), -5, 5)
            
            f1 = timed_call(x1)
            f2 = timed_call(x2)
            n_evals += 2
            if f1 is None or f2 is None:
                # Return best found so far
                if f1 is not None and f1 < f2:
                    return x1, f1, n_evals
                else:
                    return x2, f2, n_evals
            
            for _ in range(3): # Few GS iterations
                if time.time() > deadline - 2 or calls_made >= budget:
                    break
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = np.clip(b - gr * (b - a), -5, 5)
                    f1 = timed_call(x1)
                    n_evals += 1
                    if f1 is None: break
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = np.clip(a + gr * (b - a), -5, 5)
                    f2 = timed_call(x2)
                    n_evals += 1
                    if f2 is None: break
            
            # Return the best of x1, x2
            if f1 < f2:
                return x1, f1, n_evals
            else:
                return x2, f2, n_evals

        # Main Powell Loop
        stall_count = 0
        max_stall = 5
        
        while calls_made < budget and time.time() < deadline - 2:
            x_old = x.copy()
            f_old = f
            min_improvement_dir = -1
            min_improvement_val = np.inf
            
            for i in range(dim):
                if time.time() > deadline - 2 or calls_made >= budget:
                    break
                x_new, f_new, _ = minimize_line(x, dirs[i])
                if f_new is None:
                    f_new = f
                    x_new = x
                if f_new < f_old:
                    x = x_new
                    f = f_new
                    improvement = f_old - f
                    if improvement < min_improvement_val:
                        min_improvement_val = improvement
                        min_improvement_dir = i
            
            # Update best
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            # Check for stall
            if f >= f_old - 1e-10:
                stall_count += 1
            else:
                stall_count = 0
            
            if stall_count >= max_stall:
                # Stall detected, break out of Powell
                break
            
            # Update directions
            if min_improvement_dir != -1 and f < f_old - 1e-10:
                # Replace the direction that gave the least improvement with the new direction
                d_new = x - x_old
                if np.linalg.norm(d_new) > 1e-10:
                    dirs[min_improvement_dir] = d_new

        # Phase 3: Nelder-Mead Polish
        # If we have budget left, use NM to refine the solution
        if calls_made < budget and time.time() < deadline - 2:
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = best_x
            for i in range(n):
                simplex[i + 1] = best_x.copy()
                step_init = 0.1
                simplex[i + 1][i] += step_init if best_x[i] < 4.5 else -step_init
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
            
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if calls_made >= budget or time.time() > deadline - 2:
                    break
                fvals[i] = timed_call(simplex[i])
                if fvals[i] is None:
                    fvals[i] = np.inf
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            no_improvement = 0
            
            while calls_made < budget and time.time() < deadline - 2:
                if no_improvement > 10:
                    break
                
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = timed_call(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = timed_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    no_improvement = 0
                elif fr < fvals[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    fvals[-1] = fr
                    no_improvement = 0
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = timed_call(xc)
                    if fc is None:
                        break

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                        no_improvement = 0
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = timed_call(simplex[i])
                            if fvals[i] is None:
                                fvals[i] = np.inf
                        no_improvement += 1
                
                idx = np.argmin(fvals)
                if fvals[idx] < best_f:
                    best_f = fvals[idx]
                    best_x = simplex[idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
