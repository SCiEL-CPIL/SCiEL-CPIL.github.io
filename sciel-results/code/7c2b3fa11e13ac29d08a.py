"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using WFG4-aware biased sampling with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Track all evaluated points for hypervolume
        all_X = []
        all_F = []

        def evaluate(x):
            """Evaluate a point and record it."""
            nonlocal all_X, all_F
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            return x, f

        def is_dominated(f, archive_F):
            """Check if f is dominated by any solution in archive_F."""
            for fj in archive_F:
                if np.all(fj <= f) and np.any(fj < f):
                    return True
            return False

        def update_archive(f, archive_F):
            """Update archive with new point, removing dominated points."""
            nonlocal all_F
            # Check if new point is dominated
            if is_dominated(f, archive_F):
                return
            # Remove points dominated by new point
            new_archive = []
            for fj in archive_F:
                if not (np.all(f <= fj) and np.any(f < fj)):
                    new_archive.append(fj)
            new_archive.append(f)
            return new_archive

        # Initialize with a few random points to establish bounds
        pop_size = min(10, self.budget // 10)
        X_init = lo + range_ * rng.random((pop_size, self.dim))
        F_init = []
        for x in X_init:
            x, f = evaluate(x)
            F_init.append(f)
        used = pop_size

        # WFG4-aware sampling: bias toward the Pareto front
        # WFG4's Pareto front has specific geometric properties
        # Sample t uniformly and construct points near the expected front
        n_front_samples = min(100, self.budget // 2)
        
        for i in range(n_front_samples):
            if used >= self.budget or time.time() > deadline:
                break

            # Sample parameter t uniformly in [0, 1]
            t = rng.random()
            
            # For WFG4, construct a point biased toward the Pareto front
            # The first variable often controls the trade-off
            x = np.zeros(self.dim)
            
            # Set first variable based on t
            x[0] = lo[0] + t * range_[0]
            
            # For remaining variables, use a mixture of:
            # 1. Values that tend to produce non-dominated solutions
            # 2. Small random perturbations for diversity
            for d in range(1, self.dim):
                # Bias toward middle of range with some randomness
                base = 0.5 * (lo[d] + hi[d])
                spread = 0.3 * range_[d]
                x[d] = base + spread * rng.standard_normal()
                x[d] = np.clip(x[d], lo[d], hi[d])
            
            # Add small perturbation to first variable for diversity
            x[0] += 0.05 * range_[0] * rng.standard_normal()
            x[0] = np.clip(x[0], lo[0], hi[0])
            
            x, f = evaluate(x)
            used += 1

        # Local refinement phase: refine promising points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline - 5:
            # Select a subset of evaluated points for local refinement
            # Focus on points that seem to be in good regions
            n_refine = min(remaining, 50)
            
            # Sample refinement centers from evaluated points
            # Prefer points with lower objective sums
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                obj_sums = np.sum(F_arr, axis=1)
                
                # Select centers with bias toward better points
                n_centers = min(10, len(all_F))
                # Weight by inverse of objective sum
                weights = 1.0 / (obj_sums + 1e-10)
                weights /= weights.sum()
                
                center_indices = rng.choice(len(all_F), size=min(n_centers, len(all_F)), 
                                           replace=False, p=weights)
                
                for ci in center_indices:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    center = all_X[ci]
                    # Small step local search
                    n_local = min(3, self.budget - used)
                    for _ in range(n_local):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        # Adaptive step size: smaller steps for refinement
                        step_size = 0.05 * range_
                        child = center + step_size * rng.standard_normal(self.dim)
                        child = np.clip(child, lo, hi)
                        
                        x, f = evaluate(child)
                        used += 1

        return all_X
# EVOLVE-BLOCK-END
