import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo
        
        # Reference point for hypervolume
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Archive of non-dominated solutions
        archive = []
        evals_used = 0
        max_evals = self.budget
        
        # 1. Initialization: Latin Hypercube Sampling for diversity
        pop_size = min(20, max_evals)
        X_init = self._lhs_sample(rng, pop_size, self.dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        evals_used += pop_size
        
        for i in range(pop_size):
            self._add_to_archive(archive, X_init[i], F_init[i])
            
        # Population for DE
        X = X_init.copy()
        F = F_init.copy()
        
        # 2. Main Loop: Hypervolume-guided DE
        while evals_used < max_evals:
            if time.time() > deadline:
                break
                
            # Calculate hypervolume contributions for parent selection
            if len(archive) > 1:
                hv_contribs = self._calc_hv_contributions(archive, ref_point)
                # Select parent with probability proportional to hypervolume contribution
                probs = np.array(hv_contribs)
                probs = np.maximum(probs, 1e-10)
                probs /= probs.sum()
                idx1 = rng.choice(len(archive), p=probs)
                # For diversity, select other parents randomly from archive or pop
                idx2 = rng.choice(len(archive))
                idx3 = rng.choice(len(archive))
                # Ensure distinct
                attempts = 0
                while len(set([idx1, idx2, idx3])) < 3 and attempts < 10:
                    idx2 = rng.choice(len(archive))
                    idx3 = rng.choice(len(archive))
                    attempts += 1
                if len(set([idx1, idx2, idx3])) < 3:
                    # Fallback to random from population
                    idx1, idx2, idx3 = rng.choice(len(X), 3, replace=False)
                    parent1, parent2, parent3 = X[idx1], X[idx2], X[idx3]
                else:
                    parent1, parent2, parent3 = archive[idx1][0], archive[idx2][0], archive[idx3][0]
            else:
                idx1, idx2, idx3 = rng.choice(len(X), 3, replace=False)
                parent1, parent2, parent3 = X[idx1], X[idx2], X[idx3]
            
            # Adaptive F factor: start high for exploration, decrease for exploitation
            progress = evals_used / max_evals
            F_factor = 0.8 - 0.3 * progress  # 0.8 -> 0.5
            
            # DE Mutation
            mutant = parent1 + F_factor * (parent2 - parent3)
            mutant = np.clip(mutant, lo, hi)
            
            # Crossover: binomial with CR=0.5
            CR = 0.5
            jrand = rng.integers(self.dim)
            trial = mutant.copy()
            for j in range(self.dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = parent1[j]
            
            f_trial = problem(trial)
            evals_used += 1
            
            # Selection based on dominance
            dominated = False
            for i in range(len(X)):
                if np.all(F[i] <= f_trial) and np.any(F[i] < f_trial):
                    dominated = True
                    break
            
            if not dominated:
                # Remove points dominated by trial
                keep_indices = []
                for i in range(len(X)):
                    if not (np.all(f_trial <= F[i]) and np.any(f_trial < F[i])):
                        keep_indices.append(i)
                
                removed_indices = [i for i in range(len(X)) if i not in keep_indices]
                if removed_indices:
                    replace_idx = removed_indices[0]
                else:
                    # Replace worst by hypervolume contribution (approx: max sum)
                    replace_idx = np.argmax(np.sum(F, axis=1))
                
                X[replace_idx] = trial
                F[replace_idx] = f_trial
                self._add_to_archive(archive, trial, f_trial)
        
        # 3. Local Refinement using Augmented Epsilon-Constraint Method
        if archive and evals_used < max_evals:
            self._local_refinement(problem, archive, lo, hi, ref_point, 
                                  evals_used, max_evals, deadline, rng)
        
        return X

    def _lhs_sample(self, rng, n, dim, lo, hi):
        """Latin Hypercube Sampling"""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            for i in range(n):
                samples[i, d] = lo[d] + (hi[d] - lo[d]) * (perm[i] + rng.random()) / n
        return samples

    def _calc_hv_contributions(self, archive, ref_point):
        """Calculate hypervolume contribution of each point in archive"""
        n = len(archive)
        if n == 0:
            return []
        if n == 1:
            f = np.array(archive[0][1])
            if np.all(f < ref_point):
                vol = np.prod(np.maximum(ref_point - f, 0))
                return [vol]
            return [0.0]
        
        contributions = [0.0] * n
        for i in range(n):
            f_i = np.array(archive[i][1])
            if not np.all(f_i < ref_point):
                continue
            # Calculate hypervolume of region dominated by f_i but not by any other point
            # For 3 objectives, we can compute this exactly
            # Region: points p such that p <= f_i and p is not dominated by any f_j (j != i)
            # This is complex; use approximation: volume of box from f_i to ref,
            # minus volumes of boxes from other points that overlap
            
            # Simpler: compute the "unique" hypervolume contribution
            # For small archives, we can do exact 3D computation
            box = np.prod(np.maximum(ref_point - f_i, 0))
            
            # Subtract contributions from other points that dominate parts of this box
            # For each other point j, if f_j <= f_i in some objectives, 
            # the overlapping region is lost
            # This is an approximation - use the full box as upper bound
            # and let the selection be proportional to box volume
            contributions[i] = box
        
        # Normalize
        total = sum(contributions)
        if total > 0:
            contributions = [c / total for c in contributions]
        else:
            contributions = [1.0 / n] * n
        
        return contributions

    def _local_refinement(self, problem, archive, lo, hi, ref_point, 
                          evals_used, max_evals, deadline, rng):
        """
        Use Augmented Epsilon-Constraint Method to refine the Pareto front.
        For pairs of non-dominated points, minimize one objective subject to
        the other objectives being bounded by linear interpolation.
        """
        n = len(archive)
        if n < 2:
            return
        
        # Get all non-dominated points
        points = [p[0] for p in archive]
        values = [p[1] for p in archive]
        
        # Sort by first objective to create ordered front
        order = np.argsort(values[0])
        sorted_points = [points[i] for i in order]
        sorted_values = [values[i] for i in order]
        
        # For each consecutive pair, try to fill the gap
        num_refinements = min(10, max_evals - evals_used)
        evals_per_refinement = max(1, num_refinements // max(1, num_refinements))
        
        for k in range(min(n - 1, 5)):  # Try up to 5 gaps
            if evals_used >= max_evals or time.time() > deadline:
                break
            
            x1, x2 = sorted_points[k], sorted_points[k + 1]
            f1, f2 = np.array(sorted_values[k]), np.array(sorted_values[k + 1])
            
            # For each objective to minimize, set epsilons for the others
            for obj_min in range(self.n_obj):
                if evals_used >= max_evals or time.time() > deadline:
                    break
                
                # Epsilon bounds: interpolate between f1 and f2
                # We want to find a point that is better than both in some sense
                # Set epsilons slightly better than the worse of the two
                epsilons = np.zeros(self.n_obj)
                for j in range(self.n_obj):
                    if j == obj_min:
                        continue
                    # Set epsilon to be between f1[j] and f2[j], slightly tighter
                    epsilons[j] = min(f1[j], f2[j]) + 0.1 * abs(f2[j] - f1[j])
                    epsilons[j] = min(epsilons[j], max(f1[j], f2[j]))
                
                # Define the constrained optimization problem
                def objective(x, obj_min=obj_min, epsilons=epsilons):
                    f = problem(x)
                    # Penalize constraint violations
                    penalty = 0.0
                    for j in range(self.n_obj):
                        if j == obj_min:
                            continue
                        if f[j] > epsilons[j]:
                            penalty += 100.0 * (f[j] - epsilons[j])
                    return f[obj_min] + penalty
                
                # Try multiple starting points
                best_x = None
                best_f = None
                best_val = np.inf
                
                for start_idx in range(3):
                    if evals_used >= max_evals or time.time() > deadline:
                        break
                    
                    if start_idx == 0:
                        x0 = x1
                    elif start_idx == 1:
                        x0 = x2
                    else:
                        x0 = 0.5 * (x1 + x2) + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                        x0 = np.clip(x0, lo, hi)
                    
                    try:
                        result = minimize(objective, x0, method='L-BFGS-B', 
                                        bounds=list(zip(lo, hi)),
                                        options={'maxiter': 50, 'ftol': 1e-8})
                        evals_used += result.nfev
                        
                        if result.success or result.fun < best_val:
                            f_res = problem(result.x)
                            if np.all(f_res < ref_point):
                                best_x = result.x
                                best_f = f_res
                                best_val = result.fun
                    except Exception:
                        pass
                
                if best_x is not None:
                    self._add_to_archive(archive, best_x, best_f)
        
        # Additional local search around best points
        if archive and evals_used < max_evals:
            # Sort by sum of objectives
            sorted_arch = sorted(archive, key=lambda p: np.sum(p[1]))
            for point, f_val in sorted_arch[:3]:
                if evals_used >= max_evals or time.time() > deadline:
                    break
                
                # Random perturbations with decreasing step size
                for step_size in [0.1, 0.05, 0.01]:
                    if evals_used >= max_evals or time.time() > deadline:
                        break
                    for _ in range(3):
                        if evals_used >= max_evals or time.time() > deadline:
                            break
                        noise = step_size * (hi - lo)
                        new_point = point + rng.standard_normal(self.dim) * noise
                        new_point = np.clip(new_point, lo, hi)
                        new_f = problem(new_point)
                        evals_used += 1
                        self._add_to_archive(archive, new_point, new_f)

    def _add_to_archive(self, archive, x, f):
        """
        Add a point to the archive if it is non-dominated.
        Remove any points dominated by the new point.
        """
        f = np.array(f)
        # Check if new point is dominated by existing
        for ax, af in archive:
            af = np.array(af)
            if np.all(af <= f) and np.any(af < f):
                return  # Dominated, do not add
        
        # Remove points dominated by new point
        new_archive = []
        for ax, af in archive:
            af = np.array(af)
            if not (np.all(f <= af) and np.any(f < af)):
                new_archive.append((ax, af))
        
        new_archive.append((x, f))
        archive.clear()
        archive.extend(new_archive)
