import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._start_time = None

    def __call__(self, problem, bounds):
        self._start_time = time.time()
        deadline = self._start_time + 320  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Phase 1: LHS initialization for broad coverage
        n_init = min(40, budget)
        X = self._lhs_sample(lo, hi, d, n_init, rng)
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Build archive
        archive_X = X.copy()
        archive_F = F.copy()
        
        # Phase 2: Local refinement around archive points
        # For each archive point, do a local search with decreasing step sizes
        n_refine = budget - used
        if n_refine > 0:
            # Sort archive by hypervolume contribution (approximate by sum of objectives)
            # Points with lower sum are closer to ideal, more likely to contribute to HV
            sum_obj = np.sum(archive_F, axis=1)
            # We want to refine points that are good
            # But we also want diversity, so let's just go through all archive points
            # and spend evaluations on local search around them
            
            # Step sizes for local search
            step_sizes = [0.15, 0.08, 0.04, 0.02, 0.01]
            
            # For each archive point, do local search
            # We'll iterate through archive points and spend some evals on each
            evals_per_point = max(1, n_refine // len(archive_X))
            
            for idx in range(len(archive_X)):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                center = archive_X[idx].copy()
                
                # Try multiple step sizes
                for step in step_sizes:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Generate local perturbations
                    n_local = min(evals_per_point, budget - used)
                    for _ in range(n_local):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Gaussian perturbation around center
                        x_new = center + step * (hi - lo) * rng.standard_normal(d)
                        x_new = np.clip(x_new, lo, hi)
                        
                        f_new = problem(x_new)
                        used += 1
                        
                        # Check if x_new is non-dominated w.r.t. archive
                        dominated = False
                        for j in range(len(archive_F)):
                            if np.all(archive_F[j] <= f_new) and np.any(archive_F[j] < f_new):
                                dominated = True
                                break
                        
                        if not dominated:
                            # Remove archive points dominated by x_new
                            new_archive_X = []
                            new_archive_F = []
                            for j in range(len(archive_F)):
                                if not (np.all(f_new <= archive_F[j]) and np.any(f_new < archive_F[j])):
                                    new_archive_X.append(archive_X[j])
                                    new_archive_F.append(archive_F[j])
                            
                            # Add x_new
                            new_archive_X.append(x_new)
                            new_archive_F.append(f_new)
                            
                            # Limit archive size
                            if len(new_archive_X) > 50:
                                # Keep best 50 by hypervolume contribution
                                # Approximate: sort by sum of objectives
                                sums = np.sum(new_archive_F, axis=1)
                                keep_idx = np.argsort(sums)[:50]
                                new_archive_X = [new_archive_X[i] for i in keep_idx]
                                new_archive_F = [new_archive_F[i] for i in keep_idx]
                            
                            archive_X = np.array(new_archive_X)
                            archive_F = np.array(new_archive_F)
                            
                            # Update center to the new point for next iterations
                            center = x_new.copy()
        
        # Return the archive (or a subset)
        # The evaluator records all returned objective vectors
        # So we should return the archive points
        return archive_X

    def _lhs_sample(self, lo, hi, d, n, rng):
        """Latin Hypercube Sampling"""
        X = np.zeros((n, d))
        for i in range(d):
            # Divide [0, 1] into n intervals, pick one point in each
            perm = rng.permutation(n)
            for j in range(n):
                X[j, i] = (perm[j] + rng.random()) / n
        
        # Scale to [lo, hi]
        X = lo + (hi - lo) * X
        return X
