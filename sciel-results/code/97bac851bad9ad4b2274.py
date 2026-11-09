import numpy as np
import time

class Optimizer:
    """
    Sobol quasi-random global search + Nelder-Mead local refinement for BBOB f18.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        calls = 0
        best_x = None
        best_f = np.inf

        def eval_pt(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol quasi-random global exploration
        # Generate Sobol points for uniform coverage
        from scipy.stats import qmc

        # Allocate ~40% of budget to global exploration
        n_global = int(budget * 0.4)
        
        # Use Sobol sequence for low-discrepancy sampling
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate points in [0,1]^dim, then map to [-5,5]^dim
        sobol_points = sampler.random(n=n_global)
        sobol_points = 2.0 * sobol_points - 1.0  # Map to [-1, 1]^dim
        sobol_points = 5.0 * sobol_points  # Map to [-5, 5]^dim

        # Evaluate Sobol points
        for i in range(n_global):
            if calls >= budget or time.time() > deadline:
                break
            eval_pt(sobol_points[i])

        # If no valid point found (shouldn't happen), start from origin
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_pt(best_x)

        # Phase 2: Local Nelder-Mead refinement from best point
        # Use remaining budget for local optimization
        n_local = budget - calls
        
        if n_local <= 0 or time.time() > deadline:
            return best_x, best_f

        # Nelder-Mead implementation with adaptive step size
        def nelder_mead(func, x0, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
            """
            Nelder-Mead simplex optimization.
            """
            nonlocal calls, best_x, best_f
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                x_i = x0.copy()
                step = 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
                x_i[i] += step
                simplex.append(x_i)
            
            # Evaluate initial simplex
            fvals = [eval_pt(x) for x in simplex]
            
            # Track best from NM
            nm_best_x = simplex[0].copy()
            nm_best_f = min(fvals)
            
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Update best
                if fvals[0] < nm_best_f:
                    nm_best_f = fvals[0]
                    nm_best_x = simplex[0].copy()
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = eval_pt(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    f_e = eval_pt(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = eval_pt(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = eval_pt(simplex[i])
                        
                        # Check convergence
                        spread = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, dim + 1)])
                        if spread < 1e-6:
                            break
                
                # Check for convergence
                f_spread = max(fvals) - min(fvals)
                if f_spread < 1e-8:
                    break
            
            return nm_best_x, nm_best_f

        # Run Nelder-Mead from the best global point
        if calls < budget and time.time() < deadline:
            nm_x, nm_f = nelder_mead(func, best_x, max_iter=50)
            if nm_f < best_f:
                best_x = nm_x
                best_f = nm_f

        # Phase 3: Additional local refinements from other promising points
        # If we still have budget, try refining a few other good points
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            # Get the next few best points from Sobol phase (we didn't store them, so re-evaluate a few random points near best)
            # For simplicity, do a few more NM runs from slightly perturbed best
            for trial in range(min(3, remaining // 10)):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Perturb best point
                perturbation = 0.1 * np.random.randn(dim)
                x_trial = np.clip(best_x + perturbation, -5.0, 5.0)
                
                # Quick NM run with fewer iterations
                nm_x, nm_f = nelder_mead(func, x_trial, max_iter=20)
                if nm_f < best_f:
                    best_x = nm_x
                    best_f = nm_f

        return best_x, best_f
