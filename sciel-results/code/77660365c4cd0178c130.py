"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Multi-start Nelder-Mead Simplex Method.
    
    The Bent Cigar function (BBOB f12) is highly anisotropic, with one easy direction
    and several very narrow directions. Isotropic methods (like CMA-ES or DE) struggle
    to make progress along the narrow axes within a tight budget because they waste
    evaluations exploring the wide axis. Nelder-Mead, being a local method that adapts
    its shape to the local landscape, can align with the narrow directions and exploit
    them efficiently once it finds a good region.
    
    To handle the global aspect (hidden shift/rotation), we use a multi-start strategy:
    1. Perform a few random initial evaluations to find a promising region.
    2. Run Nelder-Mead from the best found point.
    3. If Nelder-Mead stagnates or the budget allows, restart from other good points.
    
    This approach typically achieves much better early progress (AOCC) on ill-conditioned
    functions compared to population-based methods that require many generations to adapt.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # Nelder-Mead parameters
        # Initial simplex: centered at a start point, with a small initial step size
        # We'll use a standard Nelder-Mead implementation inside __call__
        
        def nelder_mead(start_x, max_iter):
            """
            Perform Nelder-Mead optimization starting from start_x.
            Returns the best x found.
            """
            n = dim
            # Initial simplex: start_x + small perturbations in each coordinate
            # Use a small initial step size to ensure we are in a local region
            # but large enough to not be stuck immediately.
            # For [-5, 5] box, 0.1 is a reasonable initial step for local search.
            init_step = 0.1
            
            # Create initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = start_x
            for i in range(n):
                simplex[i + 1] = start_x
                simplex[i + 1, i] += init_step
            
            # Evaluate initial simplex
            f_values = np.zeros(n + 1)
            for i in range(n + 1):
                if time.time() > deadline or self.calls >= budget:
                    break
                try:
                    f_values[i] = wrapped_func(simplex[i])
                except (TimeoutError, ValueError):
                    return start_x  # Return current best if we run out of resources
            
            # Standard Nelder-Mead parameters
            alpha = 1.0    # Reflection
            gamma = 2.0    # Expansion
            rho = 0.5      # Contraction
            sigma = 0.5    # Shrink
            
            for _ in range(max_iter):
                if time.time() > deadline or self.calls >= budget:
                    break
                
                # Sort simplex by function values
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                # Check for convergence: if the spread of f_values is very small
                if np.max(f_values) - np.min(f_values) < 1e-10:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                if time.time() > deadline or self.calls >= budget:
                    break
                try:
                    f_r = wrapped_func(x_r)
                except (TimeoutError, ValueError):
                    break
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    if time.time() > deadline or self.calls >= budget:
                        break
                    try:
                        f_e = wrapped_func(x_e)
                    except (TimeoutError, ValueError):
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Reflection is good enough
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    
                    if time.time() > deadline or self.calls >= budget:
                        break
                    try:
                        f_c = wrapped_func(x_c)
                    except (TimeoutError, ValueError):
                        break
                    
                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if time.time() > deadline or self.calls >= budget:
                                break
                            try:
                                f_values[i] = wrapped_func(simplex[i])
                            except (TimeoutError, ValueError):
                                break
                        if time.time() > deadline or self.calls >= budget:
                            break

            # Return the best point in the final simplex
            order = np.argsort(f_values)
            return simplex[order[0]]

        # Multi-start strategy
        # 1. Initial random sampling to find a good starting point
        # Use 10% of budget for initial exploration
        init_budget = max(10, budget // 10)
        
        # Generate initial random points
        # We want to cover the space, but focus on finding a good local minimum
        # Random uniform in [-5, 5]
        candidates = []
        for _ in range(init_budget):
            if time.time() > deadline or self.calls >= budget:
                break
            x = np.random.uniform(-5, 5, dim)
            try:
                fx = wrapped_func(x)
                candidates.append((x, fx))
            except (TimeoutError, ValueError):
                break
        
        if not candidates:
            # If we couldn't even do the first call, return zero
            return np.zeros(dim), 0.0

        # Sort candidates by function value
        candidates.sort(key=lambda x: x[1])
        
        # Run Nelder-Mead from the best candidate
        # We can also try restarting from the second best if the first doesn't improve much
        # But for now, let's just run NM from the best and use remaining budget for NM iterations
        
        # Estimate max iterations for NM
        # Each NM iteration uses ~1-2 function evaluations on average
        # Let's allocate the rest of the budget to NM
        remaining_budget = budget - self.calls
        max_iter = remaining_budget
        
        best_x = nelder_mead(candidates[0][0], max_iter)
        
        # If we have time and budget, we could try a second restart from the second best candidate
        # But NM is usually fast. Let's check if we have significant budget left.
        # For now, return the best found.
        
        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
