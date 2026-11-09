"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer using:
    1. Deterministic Sobol quasi-random sequence for uniform initial coverage.
    2. Local Gaussian perturbations around non-dominated points for refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Sobol quasi-random initialization
        # Use a power of 2 for Sobol sampling
        n_sobol = 1
        while n_sobol < self.budget:
            n_sobol *= 2
        # Don't exceed budget
        n_sobol = min(n_sobol, self.budget)

        sobol = qmc.Sobol(d=self.dim, scramble=False)
        # Generate Sobol points in [0,1]^d
        points_unit = sobol.random(n_sobol)
        # Scale to bounds
        X = lo + points_unit * span
        F = np.array([problem(x) for x in X])
        used = n_sobol

        # Phase 2: Local refinement around non-dominated points
        # Find non-dominated points from the Sobol sweep
        def is_dominated(f, F_all):
            """Check if f is dominated by any point in F_all."""
            better_or_equal = np.all(F_all <= f, axis=1)
            strictly_better = np.any(F_all < f, axis=1)
            return np.any(better_or_equal & strictly_better)

        nondom_mask = np.array([not is_dominated(f, F) for f in F])
        nondom_indices = np.where(nondom_mask)[0]

        if len(nondom_indices) == 0:
            nondom_indices = np.arange(len(F))

        # Use remaining budget for local perturbations
        remaining = self.budget - used

        if remaining > 0:
            rng = np.random.default_rng(42)
            # Adaptive step size: start moderate, shrink over time
            initial_step = 0.1
            final_step = 0.01

            # We'll do local searches around each non-dominated point
            # Distribute remaining budget among non-dominated points
            n_nd = len(nondom_indices)
            if n_nd > 0:
                # Allocate evaluations per non-dominated point
                evals_per_point = max(1, remaining // n_nd)
                extra = remaining - evals_per_point * n_nd

                for nd_i, idx in enumerate(nondom_indices):
                    parent = X[idx].copy()
                    # Number of evaluations for this point
                    n_evals = evals_per_point + (1 if nd_i < extra else 0)
                    
                    # Adaptive step for this local search
                    for e in range(n_evals):
                        # Linearly decrease step size
                        progress = e / max(1, n_evals - 1)
                        step_size = initial_step + (final_step - initial_step) * progress
                        
                        # Gaussian perturbation
                        perturbation = step_size * span * rng.standard_normal(self.dim)
                        child = parent + perturbation
                        child = np.clip(child, lo, hi)

                        fc = problem(child)
                        used += 1

                        # If child is non-dominated, add it to the list
                        # and potentially update parent
                        if not is_dominated(fc, F):
                            # Check if child dominates parent
                            if np.all(fc <= F[idx]) and np.any(fc < F[idx]):
                                X[idx] = child
                                F[idx] = fc
                                parent = child.copy()
                            else:
                                # Add child as new non-dominated point
                                # But we can't easily add to X, so we just keep it
                                # for now by replacing a dominated point if possible
                                pass

                # After all local searches, recompute non-dominated set
                nondom_mask = np.array([not is_dominated(f, F) for f in F])
                nondom_indices = np.where(nondom_mask)[0]

                # Use any remaining budget for additional refinement
                remaining2 = self.budget - used
                if remaining2 > 0 and len(nondom_indices) > 0:
                    for i in range(remaining2):
                        idx = nondom_indices[rng.integers(len(nondom_indices))]
                        parent = X[idx]
                        step_size = final_step
                        perturbation = step_size * span * rng.standard_normal(self.dim)
                        child = parent + perturbation
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        used += 1
                        # Replace worst point
                        worst_idx = int(np.argmax(np.sum(F, axis=1)))
                        X[worst_idx] = child
                        F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
