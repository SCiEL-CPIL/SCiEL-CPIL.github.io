"""
Hybrid Multi-Start Nelder-Mead Optimizer for BBOB f7 (Step Ellipsoidal).
Combines the structural framework of the current program with the 
Nelder-Mead refinement machinery from the high-scoring inspirations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer designed for non-smooth, unimodal functions
    like the BBOB Step Ellipsoidal (f7).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s, but task says 300s limit for result validity.
        # We use 290s to be safe.
        deadline = start_time + 290.0
        
        dim = self.dim
        total_budget = self.budget
        
        # Fixed seed for reproducibility if needed, but random starts are better for generalization.
        # However, the task implies deterministic instances might be used, but usually random seed is fine.
        # The prompt says "hidden seeded shift", implying the function is fixed per instance.
        # We use a fixed rng seed to ensure consistent behavior across runs if the evaluator re-runs.
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Helper to check budget and time
        def can_continue():
            nonlocal calls_made
            return calls_made < total_budget and time.time() < deadline

        # Strategy:
        # 1. Perform 3 multi-starts of Nelder-Mead.
        # 2. Each start gets ~30% of the budget.
        # 3. Final 10% is used for a high-precision polish of the best point found.
        
        num_starts = 3
        # Reserve some budget for the final polish
        polish_budget = max(20, int(total_budget * 0.1))
        remaining_for_starts = total_budget - polish_budget
        evals_per_start = max(10, remaining_for_starts // num_starts)

        # List of (x, f) candidates to potentially polish
        candidates = []

        for i in range(num_starts):
            if not can_continue():
                break
                
            # Random initialization in [-5, 5]^dim
            # For f7, the minimum is in [-4, 4]^5. Uniform [-5, 5] is safe.
            x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = func(x0)
            calls_made += 1
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            candidates.append((x0.copy(), f0))
            
            # Run Nelder-Mead from this start
            # Initial step size: 1.0 is a good default for box [-5,5]
            x_opt, f_opt, calls_used = self._nelder_mead(
                func, x0, f0, 
                budget=evals_per_start, 
                rng=rng,
                deadline=deadline,
                current_calls=calls_made,
                initial_step=1.0
            )
            
            calls_made += calls_used
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
            
            candidates.append((x_opt.copy(), f_opt))

        # Polish phase:
        # If we have remaining budget and a best_x, run a tighter NM to refine.
        if best_x is not None and can_continue():
            # Calculate remaining budget
            remaining = total_budget - calls_made
            if remaining > 10:
                # Use a smaller initial step for polishing to avoid jumping away
                # The minimum is likely close to best_x now.
                x_polish, f_polish, calls_used = self._nelder_mead(
                    func, best_x, best_f,
                    budget=remaining,
                    rng=rng,
                    deadline=deadline,
                    current_calls=calls_made,
                    initial_step=0.5 # Smaller step for local refinement
                )
                
                calls_made += calls_used
                
                if f_polish < best_f:
                    best_f = f_polish
                    best_x = x_polish.copy()

        # Ensure we return something even if best_x is None (shouldn't happen with budget>0)
        if best_x is None:
             # Fallback: return a random point if something went wrong
             best_x = np.zeros(dim)
             best_f = func(best_x)
             calls_made += 1

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, budget, rng, deadline, current_calls, initial_step=1.0):
        """
        Custom Nelder-Mead implementation with strict budget and time tracking.
        Returns (x_opt, f_opt, calls_used).
        """
        dim = len(x0)
        n = dim + 1  # Number of vertices in simplex
        
        # Initialize simplex
        simplex = np.zeros((n, dim))
        f_values = np.zeros(n)
        
        simplex[0] = x0
        f_values[0] = f0
        
        calls_used = 1  # Count the initial evaluation (f0)
        
        # Initialize other vertices
        for i in range(1, n):
            if time.time() > deadline or (current_calls + calls_used) >= budget:
                break
            simplex[i] = x0.copy()
            simplex[i][i-1] += initial_step
            # Clip to box to be safe, though NM usually stays close
            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
            f_values[i] = func(simplex[i])
            calls_used += 1
        
        # If we couldn't initialize the full simplex, return best so far
        if calls_used < n:
             best_idx = np.argmin(f_values[:calls_used])
             return simplex[best_idx].copy(), f_values[best_idx], calls_used

        # Order vertices by function value
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        
        best_f = f_values[0]
        best_x = simplex[0].copy()
        
        # Standard NM parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Main loop
        # We stop if budget is exhausted or time limit reached
        # Also stop if simplex has collapsed (spread is very small)
        
        while (current_calls + calls_used) < budget:
            if time.time() > deadline:
                break
            
            # Convergence check: if the spread of function values is tiny
            if np.max(f_values) - np.min(f_values) < 1e-10:
                break
                
            # Calculate centroid of all but the worst vertex
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)
            calls_used += 1
            
            if f_values[0] <= fr < f_values[-2]:
                # Accept reflection
                simplex[-1] = xr
                f_values[-1] = fr
            elif fr < f_values[0]:
                # 2. Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                calls_used += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            else:
                # 3. Contraction
                if fr < f_values[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                
                xc = np.clip(xc, -5.0, 5.0)
                fc = func(xc)
                calls_used += 1
                
                if fc < min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # 4. Shrinkage
                    # Shrink all points except the best towards the best
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        f_values[i] = func(simplex[i])
                        calls_used += 1
                        if (current_calls + calls_used) >= budget or time.time() > deadline:
                            break
                    
                    if (current_calls + calls_used) >= budget or time.time() > deadline:
                        break
            
            # Re-order simplex
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            # Update global best for this run
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
            
            # Stagnation check: If the best point hasn't changed in a while, 
            # and the simplex is small, we might be stuck. 
            # However, NM usually handles this by shrinking. 
            # We rely on the budget and convergence check.
            
        return best_x, best_f, calls_used
# EVOLVE-BLOCK-END
