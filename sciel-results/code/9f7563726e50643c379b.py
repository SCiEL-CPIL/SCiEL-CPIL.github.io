import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 290.0  # Safety margin before 300s hard limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        evals = 0
        
        def is_dominated(f_new, archive_F):
            """Check if f_new is dominated by any point in archive_F"""
            for f_arch in archive_F:
                if np.all(f_arch <= f_new) and np.any(f_arch < f_new):
                    return True
            return False
        
        def dominates(f_new, f_arch):
            """Check if f_new dominates f_arch"""
            return np.all(f_new <= f_arch) and np.any(f_new < f_arch)
        
        def update_archive(x, f):
            """Update archive with new point, maintaining non-domination"""
            non_local = []
            for i in range(len(archive_F)):
                if dominates(f, archive_F[i]):
                    # f dominates archive point i, remove it
                    pass
                elif is_dominated(f, [archive_F[i]]):
                    # archive point i dominates f, skip adding f
                    return
                else:
                    non_local.append(i)
            
            # If we get here, f is not dominated by any existing point
            # Remove dominated points and add f
            for i in non_local:
                archive_X.append(archive_X[i])
                archive_F.append(archive_F[i])
            archive_X.clear()
            archive_F.clear()
            for i in non_local:
                archive_X.append(archive_X[i])
                archive_F.append(archive_F[i])
            archive_X.append(x)
            archive_F.append(f)
        
        # Simpler update: rebuild archive
        def update_archive_simple(x, f):
            """Simpler archive update"""
            # Check if f is dominated by any archive point
            for f_arch in archive_F:
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    return  # f is dominated, don't add
            
            # Remove archive points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not (np.all(f <= archive_F[i]) and np.any(f < archive_F[i])):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            
            # Add new point
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
        
        # Initialization: Use Sobol sampling for better coverage
        N_init = min(20, budget // 10)
        if N_init > 0:
            try:
                sampler = qmc.Sobol(d=dim, scramble=True)
                points = sampler.random(N_init)
                points = lo + points * (hi - lo)
            except Exception:
                # Fallback to random if qmc fails
                points = (lo + (hi - lo) * np.random.random((N_init, dim)))
            
            for x in points:
                if time.time() - start_time > timeout or evals >= budget:
                    break
                f = problem(x)
                evals += 1
                update_archive_simple(x, f)
        
        # Main loop: Local refinement using L-BFGS-B on scalarized objectives
        while evals < budget:
            if time.time() - start_time > timeout:
                break
            
            if not archive_X:
                # Fallback if archive is empty
                x = lo + (hi - lo) * np.random.random(dim)
                f = problem(x)
                evals += 1
                update_archive_simple(x, f)
                continue

            # Pick a random parent from archive
            idx = np.random.randint(len(archive_X))
            x_parent = archive_X[idx].copy()
            
            # Generate random weights for scalarization (Weighted Sum)
            weights = np.random.dirichlet(np.ones(n_obj))
            
            # Define scalar objective
            def scalar_obj(x):
                nonlocal evals
                if time.time() - start_time > timeout:
                    return np.inf
                fx = problem(x)
                evals += 1
                return np.dot(weights, fx)
            
            # Run L-BFGS-B
            remaining = budget - evals
            if remaining < 10:
                # Do a simple random step if budget is low
                x_new = x_parent + 0.1 * (hi - lo) * np.random.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                evals += 1
                update_archive_simple(x_new, f_new)
                continue

            # Limit evaluations per optimization step
            maxfev = min(remaining, 20)
            
            try:
                res = minimize(
                    scalar_obj, 
                    x_parent, 
                    method='L-BFGS-B', 
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 10, 'maxfev': maxfev}
                )
                x_new = res.x
                # Evaluate final point to ensure it's in archive
                f_new = problem(x_new)
                evals += 1
                update_archive_simple(x_new, f_new)
            except Exception:
                pass

        return archive_X if archive_X else [np.zeros(dim)]
