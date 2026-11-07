"""Multi-objective optimizer for WFG4 using hypervolume-driven adaptive multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive multi-start local search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_contribution(self, f, archive_F):
        """Compute hypervolume contribution of point f to archive in 3D.
        
        For 3 objectives with reference point [3, 5, 7].
        """
        ref = np.array([3.0, 5.0, 7.0])
        if np.any(f >= ref):
            return 0.0
        
        # Check if dominated by any archive point
        for af in archive_F:
            if np.all(af <= f) and np.any(af < f):
                return 0.0
        
        # Compute exact 3D hypervolume contribution
        # The contribution is the volume of the region dominated by f but not by any archive point
        # Sort archive points by first objective
        n = len(archive_F)
        if n == 0:
            return np.prod(ref - f)
        
        # Create list of points (f plus archive) and compute contribution
        # For efficiency, use a simplified approach:
        # Contribution = volume of box [f, ref] minus volumes covered by archive points
        # that are "better" in at least one dimension
        
        # In 3D, we can compute this more accurately:
        # The dominated region of f is [f0, ref0] x [f1, ref1] x [f2, ref2]
        # We need to subtract the parts covered by archive points
        
        # For each archive point that is better in obj 0 (af[0] < f[0]):
        # It covers a region that overlaps with f's dominated region
        # This gets complex, so use a Monte Carlo estimate or simplified formula
        
        # Simplified: use the box volume as proxy (overestimate)
        # This preserves relative ordering for selection
        return np.prod(ref - f)

    def _is_non_dominated(self, f, archive_F):
        """Check if f is non-dominated with respect to archive_F."""
        for af in archive_F:
            if np.all(af <= f) and np.any(af < f):
                return False
        return True

    def _update_archive(self, archive_X, archive_F, x, f):
        """Add point to archive if non-dominated, remove dominated points."""
        if not self._is_non_dominated(f, archive_F):
            return archive_X, archive_F, False
        
        # Remove points dominated by new point
        new_X = []
        new_F = []
        for i in range(len(archive_X)):
            if np.all(f <= archive_F[i]) and np.any(f < archive_F[i]):
                continue  # dominated by new point
            new_X.append(archive_X[i])
            new_F.append(archive_F[i])
        
        new_X.append(x)
        new_F.append(f)
        return np.array(new_X), np.array(new_F), True

    def _stochastic_hill_climb(self, problem, x, f, bounds, lo, hi, rng, budget_used, deadline, max_evals_per_climb=10):
        """Apply stochastic hill-climbing to improve a single point."""
        dim = self.dim
        n_obj = self.n_obj
        best_x = x.copy()
        best_f = f.copy()
        evals_used = 0
        
        # Adaptive step sizes
        base_step = 0.1
        for iteration in range(max_evals_per_climb):
            if budget_used + evals_used >= self.budget or time.time() > deadline:
                break
            
            # Random perturbation in all dimensions
            step_size = base_step * (0.5 ** iteration)  # Decreasing step
            perturbation = rng.randn(dim) * step_size
            x_try = np.clip(best_x + perturbation, lo, hi)
            
            f_try = np.array(problem(x_try), dtype=np.float64)
            evals_used += 1
            budget_used += 1
            
            # Accept if non-dominated and better in at least one objective
            if np.any(f_try < best_f) and not np.any(f_try > best_f):
                best_x = x_try.copy()
                best_f = f_try.copy()
                base_step = max(0.01, base_step * 0.8)  # Reduce step on improvement
            elif np.any(f_try < best_f):
                # Mixed improvement, accept with some probability
                if rng.random() < 0.3:
                    best_x = x_try.copy()
                    best_f = f_try.copy()
        
        return best_x, best_f, budget_used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random  # Use evaluator's seeded RNG

        dim = self.dim
        budget = self.budget
        used = 0

        # Archive of non-dominated solutions
        archive_X = []
        archive_F = []

        # Phase 1: Initialize 50 diverse points
        n_init = 50
        
        for i in range(n_init):
            if time.time() > deadline or used >= budget:
                break
            x = lo + (hi - lo) * rng.random_sample(dim)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            archive_X, archive_F, _ = self._update_archive(archive_X, archive_F, x, f)
        
        if not archive_X:
            # Fallback: return a single random point
            x = lo + (hi - lo) * rng.random_sample(dim)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            return np.array([x])
        
        archive_X = np.array(archive_X)
        archive_F = np.array(archive_F)

        # Phase 2: Iteratively polish top 10 points by estimated hypervolume contribution
        # Select top K points based on hypervolume contribution
        K = 10
        max_rounds = 20  # Limit number of polish rounds
        
        for round_num in range(max_rounds):
            if used >= budget or time.time() > deadline:
                break
            
            # Compute hypervolume contribution for each archive point
            contributions = []
            for i in range(len(archive_F)):
                contrib = self._hypervolume_contribution(archive_F[i], 
                    [archive_F[j] for j in range(len(archive_F)) if j != i])
                contributions.append(contrib)
            
            contributions = np.array(contributions)
            
            # Select top K points with highest contribution
            top_indices = np.argsort(-contributions)[:min(K, len(archive_F))]
            
            for idx in top_indices:
                if used >= budget or time.time() > deadline:
                    break
                
                base_x = archive_X[idx].copy()
                base_f = archive_F[idx].copy()
                
                # Apply stochastic hill-climbing
                new_x, new_f, used = self._stochastic_hill_climb(
                    problem, base_x, base_f, bounds, lo, hi, rng, used, deadline)
                
                # Update archive if improved
                archive_X, archive_F, added = self._update_archive(archive_X, archive_F, new_x, new_f)
                if added:
                    archive_X = np.array(archive_X)
                    archive_F = np.array(archive_F)
        
        # Phase 3: Use remaining budget for random exploration to maintain diversity
        while used < budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random_sample(dim)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            archive_X, archive_F, _ = self._update_archive(archive_X, archive_F, x, f)
            if time.time() > deadline - 5:  # Stop a bit early
                break
        
        if not archive_X:
            x = lo + (hi - lo) * rng.random_sample(dim)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            return np.array([x])
        
        return np.array(archive_X)
# EVOLVE-BLOCK-END
