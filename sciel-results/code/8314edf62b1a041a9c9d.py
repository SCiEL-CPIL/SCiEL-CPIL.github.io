"""Multi-objective optimizer on WFG4 using LHS initialization and DE-based archive update."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Efficient multi-objective optimizer for WFG4 using LHS and DE."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # We use a fixed seed for reproducibility of the LHS/DE process, 
        # but the problem evaluator seeds np.random. 
        # Using a local rng is safer to avoid interference.
        
        # Parameters
        INIT_POP = 30
        F_DE = 0.8
        CR_DE = 0.9

        # 1. Latin Hypercube Sampling for Initialization
        def latin_hypercube(n, dim, rng):
            """Generate n points using Latin Hypercube sampling in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                # For each dimension, divide [0,1] into n intervals and pick one point per interval
                intervals = np.linspace(0, 1, n + 1)
                # Random point within each interval
                points[:, d] = intervals[:-1] + rng.random(n) * (intervals[1] - intervals[0])
            # Shuffle rows to decorrelate dimensions
            perm = rng.permutation(n)
            return points[perm]

        # Generate initial population in [0,1]^dim
        X_init_unit = latin_hypercube(INIT_POP, self.dim, rng)
        # Map to actual bounds
        X_init = lo + (hi - lo) * X_init_unit
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        evaluations = INIT_POP

        # 2. Archive Management
        # Store as lists for easy appending/removal, but use numpy arrays for vectorized checks
        archive_X = []
        archive_F = []

        def update_archive(x_new, f_new):
            """
            Update the archive with a new point.
            Returns True if the point was added to the archive.
            """
            nonlocal archive_X, archive_F
            
            if len(archive_F) == 0:
                archive_X.append(x_new)
                archive_F.append(f_new)
                return True

            # Convert to numpy array for vectorized operations
            archive_F_arr = np.array(archive_F)
            
            # Check if f_new is dominated by any point in archive
            # Dominated if exists f_i such that f_i <= f_new and f_i < f_new
            leq = np.all(archive_F_arr <= f_new, axis=1)
            lt = np.any(archive_F_arr < f_new, axis=1)
            is_dominated = np.any(leq & lt)
            
            if is_dominated:
                return False

            # Remove points in archive dominated by f_new
            # new_f dominates f_i if new_f <= f_i and new_f < f_i
            leq_new = np.all(f_new <= archive_F_arr, axis=1)
            lt_new = np.any(f_new < archive_F_arr, axis=1)
            dominated_mask = leq_new & lt_new
            
            if np.any(dominated_mask):
                # Filter out dominated points
                keep_mask = ~dominated_mask
                archive_X = [archive_X[i] for i in range(len(archive_X)) if keep_mask[i]]
                archive_F = [archive_F[i] for i in range(len(archive_F)) if keep_mask[i]]
                archive_F_arr = np.array(archive_F) if archive_F else np.empty((0, self.n_obj))

            # Add new point
            archive_X.append(x_new)
            archive_F.append(f_new)
            return True

        # Populate initial archive
        for i in range(INIT_POP):
            update_archive(X_init[i], F_init[i])

        # 3. Main Loop: Differential Evolution
        # We have budget - INIT_POP evaluations left
        while evaluations < self.budget:
            if time.time() > deadline:
                break

            # Select 3 distinct parents from the archive
            # If archive is too small, we might need to handle that, but with 30 init points, it should be fine
            if len(archive_X) < 3:
                # Fallback: random perturbation if archive is tiny
                idx = rng.integers(len(archive_X))
                x1 = archive_X[idx]
                x_new = x1 + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
            else:
                # Select 3 distinct indices
                indices = rng.choice(len(archive_X), size=3, replace=False)
                x1 = archive_X[indices[0]]
                x2 = archive_X[indices[1]]
                x3 = archive_X[indices[2]]
                
                # DE Mutation: v = x1 + F * (x2 - x3)
                x_new = x1 + F_DE * (x2 - x3)
                
                # DE Crossover (Binomial)
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR_DE or j == j_rand:
                        pass # Keep v_j
                    else:
                        x_new[j] = x1[j] # Take from x1

            # Clip to bounds
            x_new = np.clip(x_new, lo, hi)
            
            # Evaluate
            f_new = problem(x_new)
            evaluations += 1
            
            # Update Archive
            update_archive(x_new, f_new)

        return archive_X
# EVOLVE-BLOCK-END
