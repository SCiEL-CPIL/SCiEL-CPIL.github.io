"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast multi-objective optimizer for RE34 using hypervolume-guided search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref_point = np.array([1705.0, 12.0, 0.3])

        # We'll maintain a set of non-dominated points
        # For 3 objectives, we can use a grid-based approach for efficiency

        # Initial population: mix of random and structured points
        pop_size = 40
        used = 0

        # Generate initial population
        X = []
        F = []

        # Some random points
        n_random = 20
        for i in range(n_random):
            x = lo + (hi - lo) * np.random.random(dim)
            X.append(x)
            F.append(problem(x))
            used += 1

        # Some structured points along the boundary
        # For RE34, good solutions often have specific patterns
        # Try points near the lower bounds for each variable
        structured_points = []
        for i in range(10):
            x = np.random.uniform(lo, hi, dim)
            # Perturb some variables to be near bounds
            for j in range(dim):
                if np.random.random() < 0.3:
                    x[j] = lo[j] + np.random.random() * 0.1 * (hi[j] - lo[j])
                elif np.random.random() < 0.3:
                    x[j] = hi[j] - np.random.random() * 0.1 * (hi[j] - lo[j])
            structured_points.append(x)

        for x in structured_points:
            X.append(x)
            F.append(problem(x))
            used += 1

        # More random points to fill up
        while len(X) < pop_size and used < budget:
            x = lo + (hi - lo) * np.random.random(dim)
            X.append(x)
            F.append(problem(x))
            used += 1

        X = np.array(X)
        F = np.array(F)

        # Helper: check if point is dominated by any point in the set
        def is_dominated(f_new, F_set):
            """Check if f_new is dominated by any point in F_set."""
            for f in F_set:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        # Helper: remove points dominated by f_new from F_set
        def remove_dominated(f_new, F_set):
            """Remove points from F_set that are dominated by f_new."""
            return [f for f in F_set if not (np.all(f_new <= f) and np.any(f_new < f))]

        # Main search loop
        while used < budget:
            if time.time() > deadline:
                break

            # Generate offspring using mutation of existing points
            n_offspring = min(10, budget - used)
            if n_offspring <= 0:
                break

            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                # Select a parent
                parent_idx = np.random.randint(len(X))
                parent = X[parent_idx].copy()

                # Mutation: Gaussian perturbation with adaptive sigma
                # Start with larger sigma, reduce over time
                progress = used / budget
                sigma = 0.15 * (1.0 - 0.5 * progress)
                
                # Per-variable mutation
                for j in range(dim):
                    if np.random.random() < 0.7:  # mutate each variable with 70% probability
                        parent[j] += sigma * (hi[j] - lo[j]) * np.random.standard_normal()
                        parent[j] = np.clip(parent[j], lo[j], hi[j])

                offspring_X.append(parent)
                f = problem(parent)
                offspring_F.append(f)
                used += 1

            # Combine with existing population and maintain non-dominated set
            # For efficiency, we'll use a simple non-dominated set maintenance
            # Since we have only 3 objectives, this is feasible

            # Add offspring to the pool
            all_F = list(F) + list(offspring_F)
            all_X = list(X) + list(offspring_X)

            # Find non-dominated points
            non_dominated_idx = []
            for i in range(len(all_F)):
                dominated = False
                for j in range(len(all_F)):
                    if i == j:
                        continue
                    if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dominated_idx.append(i)

            # If too many non-dominated points, prune using hypervolume contribution
            if len(non_dominated_idx) > 50:
                # Compute hypervolume contribution of each point (approximate)
                # For 3 objectives, we can use a simple grid-based approximation
                # Or just keep the best 50 by crowding distance

                # Simple approach: keep points that are far apart
                # Use a greedy selection based on distance
                selected = [non_dominated_idx[0]]
                for idx in non_dominated_idx[1:]:
                    if len(selected) >= 50:
                        break
                    # Check minimum distance to already selected points
                    min_dist = float('inf')
                    for sel_idx in selected:
                        dist = np.linalg.norm(all_F[idx] - all_F[sel_idx])
                        min_dist = min(min_dist, dist)
                    if min_dist > 0.01:  # threshold
                        selected.append(idx)

                # If we still don't have enough, fill with remaining
                while len(selected) < 50 and len(non_dominated_idx) > 0:
                    remaining = [i for i in non_dominated_idx if i not in selected]
                    if not remaining:
                        break
                    selected.append(remaining[0])

                non_dominated_idx = selected[:50]

            # Update population to non-dominated set
            X = np.array([all_X[i] for i in non_dominated_idx])
            F = np.array([all_F[i] for i in non_dominated_idx])

        return X
# EVOLVE-BLOCK-END
