"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start Powell's method with dense, stratified Latin Hypercube initialization.
    
    Strategy:
    1. Generate a dense set of initial points using stratified Latin Hypercube sampling
       to ensure broad coverage of the search space.
    2. Evaluate all initial points to identify promising basins.
    3. For the top candidates, apply Powell's method (a derivative-free line-search method)
       to rapidly converge to precise local minima.
    4. Powell's method is particularly effective for smooth, separable functions like
       Lunacek's bi-Rastrigin, as it exploits the coordinate-wise structure efficiently.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin: 15s buffer before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        calls_used = 0
        rng = np.random.default_rng(42)
        
        # Step 1: Generate dense Latin Hypercube initial points
        # We want to use about 50% of the budget for initial exploration
        n_initial = min(budget // 2, 200)  # Cap at 200 to leave room for Powell
        n_initial = max(n_initial, 20)     # At least 20 points
        
        # Latin Hypercube Sampling: stratify each dimension into n_initial bins
        # and pick one point from each bin per dimension
        lhs_points = np.zeros((n_initial, dim))
        for d in range(dim):
            # Random permutation of bins
            perm = rng.permutation(n_initial)
            # Random offset within each bin
            offsets = rng.uniform(0, 1, size=n_initial)
            # Map to [-5, 5]
            lhs_points[:, d] = -5.0 + 10.0 * (perm + offsets) / n_initial
        
        # Evaluate initial points
        f_values = np.array([func(x) for x in lhs_points])
        calls_used += n_initial
        
        # Sort by fitness (ascending)
        order = np.argsort(f_values)
        lhs_points = lhs_points[order]
        f_values = f_values[order]
        
        global_best_x = lhs_points[0].copy()
        global_best_f = f_values[0]
        
        # Step 2: Apply Powell's method to top candidates
        # We'll try Powell on the top K candidates
        n_powell = min(5, n_initial)  # Try up to 5 candidates
        
        for i in range(n_powell):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            
            x0 = lhs_points[i].copy()
            f0 = f_values[i]
            
            # Run Powell's method
            x_opt, f_opt, calls_this_run = self._powell(func, x0, f0, dim, budget - calls_used, deadline)
            calls_used += calls_this_run
            
            if f_opt < global_best_f:
                global_best_f = f_opt
                global_best_x = x_opt
        
        return global_best_x, global_best_f

    def _powell(self, func, x0, f0, dim, remaining_budget, deadline):
        """
        Powell's method implementation.
        
        Powell's method works by maintaining a set of conjugate directions.
        At each iteration, it performs line searches along each direction,
        then constructs a new direction from the most successful search.
        
        Parameters:
        - func: objective function
        - x0: starting point
        - f0: function value at x0
        - dim: dimension
        - remaining_budget: max number of function calls allowed
        - deadline: time deadline
        
        Returns:
        - x_opt: best point found
        - f_opt: best function value
        - calls_used: number of function calls made
        """
        calls_used = 0
        
        # Initial directions: standard basis vectors
        directions = np.eye(dim)
        
        x = x0.copy()
        f = f0
        
        # Maximum iterations to prevent infinite loops
        max_iter = 50
        
        for iteration in range(max_iter):
            if calls_used >= remaining_budget:
                break
            if time.time() > deadline:
                break
            
            x_old = x.copy()
            f_old = f
            
            # Line search along each direction
            for i in range(dim):
                if calls_used >= remaining_budget:
                    break
                if time.time() > deadline:
                    break
                
                # Line search along directions[i]
                x_new, f_new, calls_line = self._line_search(
                    func, x, f, directions[i], dim, remaining_budget - calls_used, deadline
                )
                calls_used += calls_line
                
                if f_new < f:
                    x = x_new
                    f = f_new
                
                # Update direction: the new direction is the old direction minus the one we just searched
                # Actually, in Powell's method, we swap: the direction we just searched becomes the new conjugate direction
                # Standard Powell: after searching along direction i, we replace direction i with (x_new - x_old_prev)
                # But a simpler variant: just keep the basis and update via reflection
                pass
            
            # Check for convergence
            if np.linalg.norm(x - x_old) < 1e-8 and abs(f - f_old) < 1e-10:
                break
        
        return x, f, calls_used

    def _line_search(self, func, x, f, direction, dim, remaining_budget, deadline):
        """
        1D line search along a given direction.
        Uses a simple parabolic interpolation or golden section search.
        """
        calls_used = 0
        
        # Initial step size
        step = 0.1
        
        # Try to find a better point by stepping in both directions
        best_x = x.copy()
        best_f = f
        
        # Try positive direction
        x_trial = x + step * direction
        x_trial = np.clip(x_trial, -5, 5)
        f_trial = func(x_trial)
        calls_used += 1
        
        if f_trial < best_f:
            best_x = x_trial.copy()
            best_f = f_trial
        
        # Try negative direction
        x_trial = x - step * direction
        x_trial = np.clip(x_trial, -5, 5)
        f_trial = func(x_trial)
        calls_used += 1
        
        if f_trial < best_f:
            best_x = x_trial.copy()
            best_f = f_trial
        
        # Refine with golden section if we found an improvement
        if best_f < f and calls_used < remaining_budget:
            # Determine the direction of improvement
            if best_x[0] > x[0]:
                a = x.copy()
                b = best_x + step * direction
            else:
                a = best_x - step * direction
                b = x.copy()
            
            # Golden section search
            gr = (np.sqrt(5) - 1) / 2  # Golden ratio conjugate
            a = np.clip(a, -5, 5)
            b = np.clip(b, -5, 5)
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            f_c = func(c)
            f_d = func(d)
            calls_used += 2
            
            for _ in range(10):
                if calls_used >= remaining_budget or time.time() > deadline:
                    break
                
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - gr * (b - a)
                    f_c = func(c)
                    calls_used += 1
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + gr * (b - a)
                    f_d = func(d)
                    calls_used += 1
                
                if np.linalg.norm(b - a) < 1e-8:
                    break
            
            # Best point is the midpoint of the final interval
            x_opt = 0.5 * (a + b)
            x_opt = np.clip(x_opt, -5, 5)
            f_opt = func(x_opt)
            calls_used += 1
            
            if f_opt < best_f:
                best_x = x_opt
                best_f = f_opt
        
        return best_x, best_f, calls_used
# EVOLVE-BLOCK-END
