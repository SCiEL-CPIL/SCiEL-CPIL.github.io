"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Objective Simulated Annealing (MOSA) for WFG7 optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Initialize with stratified sampling for good initial coverage
        n_init = 10
        X_init = np.empty((n_init, self.dim))
        for i in range(n_init):
            # Use Latin hypercube-like stratification
            for d in range(self.dim):
                X_init[i, d] = lo[d] + (hi[d] - lo[d]) * ((i + rng.random()) / n_init)
        
        F_init = np.array([problem(x) for x in X_init])
        evals = n_init
        
        # Build initial non-dominated archive
        archive_X = []
        archive_F = []
        for i in range(n_init):
            if not any(self._dominates(archive_F[j], F_init[i]) for j in range(len(archive_F))):
                # Remove solutions dominated by this new one
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_X)):
                    if not self._dominates(F_init[i], archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                new_archive_X.append(X_init[i])
                new_archive_F.append(F_init[i])
                archive_X = new_archive_X
                archive_F = new_archive_F
        
        # Start from the best hypervolume contributor
        best_idx = self._best_hv_index(archive_F)
        current_X = archive_X[best_idx].copy()
        current_F = archive_F[best_idx].copy()
        
        # SA parameters
        T0 = 1.0  # Initial temperature
        T_end = 0.001
        cooling = 0.995  # Cooling rate per step
        
        # Mutation parameters
        base_sigma = 0.1  # Base mutation step size
        
        T = T0
        step = 0
        
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Adaptive mutation: smaller steps as temperature drops
            sigma = base_sigma * (0.1 + 0.9 * T / T0)
            
            # Generate perturbation
            if rng.random() < 0.3:
                # Large jump: pick a random point and interpolate
                anchor = archive_X[rng.integers(len(archive_X))]
                alpha = rng.random()
                new_X = current_X * (1 - alpha) + anchor * alpha
                new_X = new_X + sigma * rng.standard_normal(self.dim)
            else:
                # Local perturbation
                new_X = current_X + sigma * rng.standard_normal(self.dim)
            
            # Clip to bounds
            new_X = np.clip(new_X, lo, hi)
            
            # Evaluate
            new_F = problem(new_X)
            evals += 1
            
            # Check if new solution is non-dominated
            if not any(self._dominates(archive_F[j], new_F) for j in range(len(archive_F))):
                # Add to archive and remove dominated solutions
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_X)):
                    if not self._dominates(new_F, archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                new_archive_X.append(new_X.copy())
                new_archive_F.append(new_F.copy())
                archive_X = new_archive_X
                archive_F = new_archive_F
                
                # Always accept non-dominated solutions
                current_X = new_X.copy()
                current_F = new_F.copy()
            else:
                # Solution is dominated; use simulated annealing acceptance
                # Compute a scalarization-based penalty
                # Use weighted sum with random weights for diversity
                w = rng.dirichlet(np.ones(self.n_obj))
                penalty = np.dot(w, new_F) - np.dot(w, current_F)
                
                # Accept if penalty is negative or with probability exp(-penalty/T)
                if penalty < 0 or (T > 0 and rng.random() < np.exp(-penalty / T)):
                    current_X = new_X.copy()
                    current_F = new_F.copy()
            
            # Cool down
            T *= cooling
            step += 1
            
            # Periodically restart from best hypervolume contributor if stuck
            if step % 50 == 0 and len(archive_F) > 1:
                best_idx = self._best_hv_index(archive_F)
                if not np.allclose(current_F, archive_F[best_idx]):
                    # Restart from best with some probability
                    if rng.random() < 0.5:
                        current_X = archive_X[best_idx].copy()
                        current_F = archive_F[best_idx].copy()
        
        # Ensure we return something reasonable
        if len(archive_X) > 0:
            return np.array(archive_X)
        return np.array([current_X])

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _best_hv_index(self, F, ref_point=None):
        """Find the index of the solution that contributes most to hypervolume.
        Simplified: use the solution closest to the ideal point (minimum of each objective)."""
        if len(F) == 0:
            return 0
        if ref_point is None:
            # Use the maximum as a rough reference
            ref_point = F.max(axis=0)
        
        # Compute distance to ideal (normalized)
        f_min = F.min(axis=0)
        f_range = ref_point - f_min
        f_range[f_range < 1e-10] = 1e-10
        
        # Normalize and compute distance
        F_norm = (F - f_min) / f_range
        dists = np.sqrt(np.sum(F_norm ** 2, axis=1))
        return np.argmin(dists)
# EVOLVE-BLOCK-END
