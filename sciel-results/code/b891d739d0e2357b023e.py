# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer for RE34 using Sobol sampling and local stochastic refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        def is_deadline():
            return (deadline - time.time()) < 5.0

        def clip(x):
            return np.clip(x, lo, hi)

        def get_non_dominated(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            F_i = F[:, np.newaxis, :]
            F_j = F[np.newaxis, :, :]
            leq = np.all(F_i <= F_j, axis=2)
            lt = np.any(F_i < F_j, axis=2)
            dominates = leq & lt
            dominated = np.any(dominates, axis=0)
            return np.where(~dominated)[0]

        def is_dominated(f_new, F_front):
            if len(F_front) == 0:
                return False
            leq = np.all(F_front <= f_new, axis=1)
            lt = np.any(F_front < f_new, axis=1)
            return np.any(leq & lt)

        F_all = []
        X_all = []
        used = 0

        # Phase 1: Sobol quasi-random sampling
        n_sobol = 120
        sampler = qmc.Sobol(d=dim, scramble=True)
        n_points = 1
        while n_points < n_sobol:
            n_points *= 2
        sobol_samples = sampler.random(n_points)[:n_sobol]
        X_sobol = lo + sobol_samples * (hi - lo)

        for i in range(n_sobol):
            if is_deadline() or used >= budget:
                break
            x = X_sobol[i]
            f = problem(x)
            F_all.append(f)
            X_all.append(x)
            used += 1

        if used == 0:
            x = lo + np.random.random(dim) * (hi - lo)
            f = problem(x)
            return np.array([f])

        F_all = np.array(F_all)
        X_all = np.array(X_all)

        # Get initial non-dominated front
        nd_idx = get_non_dominated(F_all)
        front_F = F_all[nd_idx]
        front_X = X_all[nd_idx]

        # Phase 2: Local stochastic refinement around non-dominated points
        # We will try to refine the top 'k' points on the front
        # Sort front by hypervolume contribution or just take all if few
        # For simplicity, take up to 10 best points (randomly sampled from front if > 10)
        max_refine_points = 10
        if len(front_X) > max_refine_points:
            # Randomly select a subset to refine to ensure diversity
            indices = np.random.choice(len(front_X), max_refine_points, replace=False)
            refine_X = front_X[indices]
        else:
            refine_X = front_X

        # Calculate ideal point for potential future use, though we just do random steps
        ideal = np.min(F_all, axis=0)
        
        # Determine step size based on bound range
        range_size = hi - lo
        step_size = 0.1 * range_size

        n_refine_per_point = 5
        total_refine = len(refine_X) * n_refine_per_point
        remaining_budget = budget - used

        # Adjust number of refinements to fit budget
        if total_refine > remaining_budget:
            # Reduce steps per point or number of points
            # Simple approach: just loop until budget is hit
            pass

        for x_target in refine_X:
            if used >= budget or is_deadline():
                break
            
            for _ in range(n_refine_per_point):
                if used >= budget or is_deadline():
                    break
                
                # Gaussian perturbation
                x_new = x_target + np.random.normal(0, step_size, dim)
                x_new = clip(x_new)
                
                f_new = problem(x_new)
                F_all = np.vstack([F_all, [f_new]])
                X_all = np.vstack([X_all, [x_new]])
                used += 1

                # Update front dynamically to potentially target better points later?
                # No, we pre-selected targets. This is fine.

        # Phase 3: Fill remaining budget with random points if time/budget allows
        while used < budget and not is_deadline():
            x_new = lo + np.random.random(dim) * (hi - lo)
            f_new = problem(x_new)
            F_all = np.vstack([F_all, [f_new]])
            X_all = np.vstack([X_all, [x_new]])
            used += 1

        return F_all

# EVOLVE-BLOCK-END
