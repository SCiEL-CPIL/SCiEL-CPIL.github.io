import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        total_budget = self.budget
        
        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > total_budget:
                raise ValueError("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # 1. Initial Exploration: Random sampling to get a rough sense of the landscape
        # Use 20% of budget for initial random search to find a good starting point
        init_budget = int(total_budget * 0.2)
        for _ in range(init_budget):
            if time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, n)
            try:
                wrapped_func(x)
            except (TimeoutError, ValueError):
                break

        # 2. Coordinate Descent Phase
        # Coordinate descent is efficient for ridges if we can adapt step sizes.
        # We will use a simple adaptive coordinate descent.
        # We allocate the remaining 80% of the budget to this phase.
        
        # Current best point
        x_current = best_x.copy()
        f_current = best_f
        
        # Step sizes for each dimension
        # Start with a large step size
        steps = np.ones(n) * 2.0
        
        # We'll iterate until budget or time is exhausted
        # To avoid getting stuck in local minima on a ridge, we might need to occasionally jump.
        # But f13 is a sharp ridge, so coordinate descent should work well if we find the ridge.
        
        while calls_made[0] < total_budget:
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                if calls_made[0] >= total_budget:
                    break
                    
                # Try positive step
                x_try = x_current.copy()
                x_try[i] += steps[i]
                x_try = np.clip(x_try, -5, 5)
                
                try:
                    f_try = wrapped_func(x_try)
                except (TimeoutError, ValueError):
                    break
                    
                if f_try < f_current:
                    x_current = x_try
                    f_current = f_try
                    improved = True
                else:
                    # Try negative step
                    x_try = x_current.copy()
                    x_try[i] -= steps[i]
                    x_try = np.clip(x_try, -5, 5)
                    
                    try:
                        f_try = wrapped_func(x_try)
                    except (TimeoutError, ValueError):
                        break
                        
                    if f_try < f_current:
                        x_current = x_try
                        f_current = f_try
                        improved = True
                    else:
                        # No improvement in this dimension, reduce step size
                        steps[i] *= 0.5
                        # If step size becomes too small, we might be stuck.
                        # For now, just keep reducing.
                        
            if not improved:
                # If no dimension improved, we might be stuck or at a local min.
                # For a sharp ridge, if we are on the ridge, we should be able to move.
                # If we are off the ridge, we might need a larger jump.
                # However, reducing steps is the standard CD behavior.
                # Let's add a small random perturbation if we've been stuck for a while?
                # Or just break if steps are very small.
                if np.all(steps < 1e-4):
                    break
                    
        # 3. Final Polish with Nelder-Mead
        # Use remaining budget for Nelder-Mead to handle non-orthogonal ridges better.
        if calls_made[0] < total_budget:
            try:
                # Simple Nelder-Mead implementation
                # Initialize simplex
                simplex = []
                f_simplex = []
                
                # Start with current best
                x0 = x_current.copy()
                f0 = f_current
                simplex.append(x0)
                f_simplex.append(f0)
                
                # Generate other vertices
                for i in range(n):
                    xi = x0.copy()
                    # Perturb one dimension
                    if abs(xi[i]) < 1e-10:
                        xi[i] = 0.05
                    else:
                        xi[i] += 0.05 * np.sign(xi[i])
                    xi = np.clip(xi, -5, 5)
                    try:
                        fi = wrapped_func(xi)
                    except (TimeoutError, ValueError):
                        break
                    simplex.append(xi)
                    f_simplex.append(fi)
                    
                if len(simplex) == n + 1:
                    # Nelder-Mead iterations
                    while calls_made[0] < total_budget:
                        if time.time() > deadline:
                            break
                        
                        # Sort simplex
                        idx = np.argsort(f_simplex)
                        simplex = [simplex[i] for i in idx]
                        f_simplex = [f_simplex[i] for i in idx]
                        
                        # Check convergence
                        if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                            break
                            
                        # Centroid of all but worst
                        centroid = np.mean(simplex[:-1], axis=0)
                        
                        # Reflection
                        alpha = 1.0
                        xr = centroid + alpha * (centroid - simplex[-1])
                        xr = np.clip(xr, -5, 5)
                        
                        try:
                            fr = wrapped_func(xr)
                        except (TimeoutError, ValueError):
                            break
                            
                        if fr < f_simplex[0]:
                            # Expansion
                            gamma = 2.0
                            xe = centroid + gamma * (centroid - simplex[-1])
                            xe = np.clip(xe, -5, 5)
                            try:
                                fe = wrapped_func(xe)
                            except (TimeoutError, ValueError):
                                break
                                
                            if fe < fr:
                                simplex[-1] = xe
                                f_simplex[-1] = fe
                            else:
                                simplex[-1] = xr
                                f_simplex[-1] = fr
                        elif fr < f_simplex[-2]:
                            # Reflection
                            simplex[-1] = xr
                            f_simplex[-1] = fr
                        else:
                            # Contraction
                            rho = 0.5
                            if fr < f_simplex[-1]:
                                # Outside contraction
                                xc = centroid + rho * (xr - centroid)
                            else:
                                # Inside contraction
                                xc = centroid + rho * (simplex[-1] - centroid)
                                
                            xc = np.clip(xc, -5, 5)
                            try:
                                fc = wrapped_func(xc)
                            except (TimeoutError, ValueError):
                                break
                                
                            if fc < min(fr, f_simplex[-1]):
                                simplex[-1] = xc
                                f_simplex[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, n + 1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], -5, 5)
                                    try:
                                        f_simplex[i] = wrapped_func(simplex[i])
                                    except (TimeoutError, ValueError):
                                        break
                                        
                        # Update best
                        if f_simplex[0] < best_f:
                            best_f = f_simplex[0]
                            best_x = simplex[0].copy()
                            
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        return best_x, best_f
