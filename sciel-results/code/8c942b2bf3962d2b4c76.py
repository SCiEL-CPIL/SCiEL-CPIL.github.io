"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hierarchical Multi-Scale Stochastic Tunneling Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, aim for 300s to be safe
        deadline = start_time + 300
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def try_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return np.inf
            # Ensure x is a valid numpy array
            x = np.asarray(x, dtype=float)
            call_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_time():
            return time.time() < deadline and call_count < budget

        # 1. Coarse-Grained Exploration (Multi-Scale Tunneling)
        # Weierstrass is self-similar. We start with large steps to find basins,
        # then reduce steps to refine.
        
        # Initial random samples to get a rough idea of the landscape
        # We allocate ~20% of budget to coarse exploration
        coarse_budget = int(budget * 0.2)
        
        # Start with a few random points
        # The minimizer is in [-4, 4]^5, so sampling here is good
        current_x = rng.uniform(-4, 4, dim)
        current_f = try_func(current_x)
        
        # Scale factors for the random walk
        # We reduce scale by factor 10 at each level
        scales = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        
        # Tunneling parameters
        # If we don't improve for a while, we "tunnel" to a new random location
        # but keep the best found so far.
        
        for scale in scales:
            if not check_time():
                break
            if call_count >= budget:
                break
                
            # If we've used up the coarse budget, we switch to a different strategy
            # But let's just continue the walk for now, as it's cheap.
            
            # Number of steps at this scale
            # We want to spend roughly equal time/compute at each scale?
            # Or just a fixed number of steps?
            # Let's say 10 steps per scale for the first few, then more?
            # Actually, let's just do a random walk with step size 'scale'
            # and accept if better (Metropolis-like or just greedy with restarts)
            
            # Greedy random walk with occasional restarts (tunneling)
            # If stuck (no improvement in K steps), jump to a new random point
            # weighted by previous best? No, just random.
            
            stuck_count = 0
            max_stuck = 5  # If no improvement in 5 steps, tunnel
            
            while check_time() and call_count < budget:
                # Generate step
                step = rng.standard_normal(dim) * scale
                new_x = current_x + step
                # Clip to box
                new_x = np.clip(new_x, -5, 5)
                
                new_f = try_func(new_x)
                
                if new_f < current_f:
                    current_x = new_x
                    current_f = new_f
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count >= max_stuck:
                        # Tunneling: Jump to a new random location
                        # We can bias the jump towards the global best found so far
                        # Or just purely random.
                        # Let's try a mix: 50% random, 50% near best
                        if rng.random() < 0.5:
                            current_x = rng.uniform(-4, 4, dim)
                        else:
                            current_x = best_x + rng.standard_normal(dim) * 1.0
                            current_x = np.clip(current_x, -5, 5)
                        current_f = try_func(current_x)
                        stuck_count = 0
                        
        # 2. Fine-Grained Refinement (Nelder-Mead)
        # Use the remaining budget for local search around the best point found.
        # Nelder-Mead is good for low-dimensional, non-smooth-ish problems.
        
        if check_time() and call_count < budget:
            # We need to wrap func to count calls and check time
            nm_call_count = 0
            nm_best_f = best_f
            nm_best_x = best_x.copy()
            
            def nm_func(x):
                nonlocal nm_call_count, nm_best_f, nm_best_x
                if call_count >= budget:
                    return nm_best_f
                if time.time() >= deadline:
                    return nm_best_f
                call_count += 1
                nm_call_count += 1
                fx = func(x)
                if fx < nm_best_f:
                    nm_best_f = fx
                    nm_best_x = x.copy()
                return fx
            
            # Initial simplex
            # Use a small perturbation around the best point
            x0 = best_x
            # Simplex size
            # If we are close to the boundary, be careful
            # Standard NM simplex: x0, x0 + delta * e_i
            delta = 0.1  # Initial simplex size
            
            # Create initial simplex
            simplex = [x0]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # Evaluate initial simplex
            for i, x in enumerate(simplex):
                if not check_time():
                    break
                f_val = nm_func(x)
                # If we hit budget or time, break
                if call_count >= budget or time.time() >= deadline:
                    break
            
            # Run Nelder-Mead
            # We can't easily interrupt scipy.optimize.minimize mid-way if it's in a C loop,
            # but we can set a callback or just let it run and hope it's fast enough.
            # However, scipy NM might use many calls.
            # To be safe, we can implement a simple NM ourselves or use scipy with a tight maxiter.
            # Let's use scipy with maxiter=100 and a callback that checks time.
            
            # Actually, implementing a basic NM is safer for strict time limits.
            # But scipy is available. Let's try scipy with a custom callback.
            
            def callback(xk):
                if time.time() >= deadline or call_count >= budget:
                    raise RuntimeError("Time or Budget Exceeded")
            
            try:
                result = minimize(nm_func, x0, method='Nelder-Mead', 
                                  options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8},
                                  callback=callback)
            except RuntimeError:
                pass
            except Exception:
                pass
            
            # Update global best if NM found something better
            if result.success or hasattr(result, 'fun') and result.fun < best_f:
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            else:
                # If NM failed, we still have the best from the walk
                pass
                
        # 3. Final Polish (Optional)
        # If we have budget left, do a few more random perturbations around best_x
        # with very small steps.
        if check_time() and call_count < budget:
            for _ in range(10):
                if not check_time():
                    break
                step = rng.standard_normal(dim) * 0.01
                new_x = best_x + step
                new_x = np.clip(new_x, -5, 5)
                new_f = try_func(new_x)
                # No need to update best_x here as try_func does it, 
                # but we might want to move our search center if it improves?
                # try_func updates best_x. So the next step will be around the new best.
                # This is a simple (1+1)-ES around the best point.

        return best_x, best_f

# EVOLVE-BLOCK-END
