"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Nelder-Mead Optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # Split budget: 25% for global search, 75% for local refinement
        # This is a structural change from PSO's constant population size.
        global_budget = int(budget * 0.25)
        local_budget = budget - global_budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # --- Phase 1: Global Search via Multi-Start Nelder-Mead ---
        # We will run multiple short Nelder-Mead runs.
        # Each run consumes a portion of the global budget.
        # Let's aim for ~10 restarts in the global phase.
        num_global_restarts = 10
        budget_per_restart = max(5, global_budget // num_global_restarts)
        
        calls_made = 0
        
        for r in range(num_global_restarts):
            if calls_made >= global_budget or time.time() > deadline - 10.0:
                break
            
            # Random starting point
            x0 = rng.uniform(-5, 5, dim)
            
            # Run a short Nelder-Mead
            current_best_x, current_best_f, calls_used = self._nelder_mead(
                func, x0, dim, budget_per_restart, calls_made, deadline, rng
            )
            calls_made += calls_used
            
            if current_best_f < best_f:
                best_f = current_best_f
                best_x = current_best_x
        
        # --- Phase 2: Local Refinement ---
        # Use the remaining budget to refine the best point found.
        if best_x is not None and calls_made < budget:
            # Run a longer Nelder-Mead from the best point
            # We can also try a few small perturbations if budget allows
            final_best_x, final_best_f, calls_used = self._nelder_mead(
                func, best_x, dim, local_budget, calls_made, deadline, rng
            )
            calls_made += calls_used
            
            if final_best_f < best_f:
                best_f = final_best_f
                best_x = final_best_x
            
            # If we have spare budget, do one more refinement from the new best
            if calls_made < budget and time.time() < deadline - 5.0:
                # Use half of remaining for a final polish
                remaining = budget - calls_made
                if remaining > 10:
                    final2_best_x, final2_best_f, calls_used = self._nelder_mead(
                        func, best_x, dim, remaining, calls_made, deadline, rng
                    )
                    if final2_best_f < best_f:
                        best_f = final2_best_f
                        best_x = final2_best_x

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f

    def _nelder_mead(self, func, x0, dim, max_iter, current_calls, deadline, rng):
        """
        Run Nelder-Mead optimization.
        Returns: (best_x, best_f, calls_made_in_this_run)
        """
        # Initialize simplex
        # Standard Nelder-Mead simplex has dim+1 points
        # We start with x0 and dim other points slightly perturbed
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        
        # Initial step size
        step = 0.5
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i-1] += step
        
        # Evaluate initial simplex
        fvals = np.array([func(x) for x in simplex])
        calls_made = dim + 1
        
        # Sort simplex by fitness
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        best_x = simplex[0].copy()
        best_f = fvals[0]
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        while calls_made < max_iter:
            if time.time() > deadline - 1.0:
                break
            
            # Check convergence (optional, but helps avoid useless iterations)
            # If spread is very small, stop
            if np.std(fvals) < 1e-8:
                break
                
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_made += 1
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_made += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                # Outside contraction if worst is worse than reflection
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_made += 1
                
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        if calls_made >= max_iter:
                            break
            
            # Re-sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
        
        return best_x, best_f, calls_made
# EVOLVE-BLOCK-END
