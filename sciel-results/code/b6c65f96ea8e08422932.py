# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead with adaptive restarts for anisotropic functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        tol = 1e-10  # Tolerance for stopping NM
        
        # Restart logic
        max_nm_iter = 50  # Max iterations for one NM run before checking restart
        stall_threshold = 20 # If no improvement in best_f for this many calls, restart NM
        
        current_best_f = np.inf
        last_improvement_call = 0
        
        # Initial point: center is a good prior for BBOB
        x0 = np.zeros(n)
        f0 = wrapped_func(x0)
        current_best_f = f0
        last_improvement_call = calls_made[0]
        
        # Initial simplex size
        step_size = 2.0
        
        while calls_made[0] < budget and time.time() < deadline:
            # Build initial simplex around current best or a new random point?
            # Strategy: Start NM from the current best_x.
            # If NM gets stuck (stagnation), we will restart NM from a random point.
            
            # 1. Construct Simplex
            simplex = [best_x.copy()]
            simplex_f = [best_f]
            
            for i in range(n):
                x_i = best_x.copy()
                # Perturb in direction i
                x_i[i] += step_size
                simplex.append(x_i)
                simplex_f.append(wrapped_func(x_i))
                
                if calls_made[0] >= budget or time.time() > deadline:
                    break
            
            if calls_made[0] >= budget or time.time() > deadline:
                break
            
            # 2. Run Nelder-Mead
            nm_iter = 0
            improved = False
            
            while nm_iter < max_nm_iter:
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                sorted_indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in sorted_indices]
                simplex_f = [simplex_f[i] for i in sorted_indices]
                
                # Check convergence
                if abs(simplex_f[-1] - simplex_f[0]) < tol:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = wrapped_func(xr)
                
                if simplex_f[0] <= fr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    improved = True
                elif fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = wrapped_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    improved = True
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = wrapped_func(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                            improved = True
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = wrapped_func(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = wrapped_func(xc)
                        if fc < simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                            improved = True
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = wrapped_func(simplex[i])
                
                nm_iter += 1
                
                # Update global best
                min_f_in_simplex = min(simplex_f)
                if min_f_in_simplex < best_f:
                    best_f = min_f_in_simplex
                    best_idx = np.argmin(simplex_f)
                    best_x = simplex[best_idx].copy()
                    last_improvement_call = calls_made[0]
                    improved = True
                elif improved:
                    # Even if not global best, local improvement suggests we are moving
                    pass

            # 3. Stagnation Check and Restart
            # If we haven't improved the global best in a while, or NM stopped early without converging
            if calls_made[0] - last_improvement_call > stall_threshold:
                # Restart: Pick a new random starting point
                # Bias towards center but allow exploration
                new_start = rng.uniform(-3, 3, n)
                f_new = wrapped_func(new_start)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = new_start.copy()
                    last_improvement_call = calls_made[0]
                else:
                    # If the random point is worse, keep best_x but maybe increase step size?
                    # Or just restart NM from best_x with a larger step size?
                    # Let's restart NM from best_x but with a larger step size to escape local minima
                    step_size *= 1.5
                    if step_size > 5.0:
                        step_size = 5.0
                
                # Also try a few random points if we are stuck
                for _ in range(5):
                    if calls_made[0] >= budget:
                        break
                    rand_x = rng.uniform(-5, 5, n)
                    f_rand = wrapped_func(rand_x)
                    if f_rand < best_f:
                        best_f = f_rand
                        best_x = rand_x.copy()
                        last_improvement_call = calls_made[0]
                        step_size = 2.0 # Reset step size on new best

        return best_x, best_f
# EVOLVE-BLOCK-END
