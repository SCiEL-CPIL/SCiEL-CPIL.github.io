"""Multi-objective optimizer on WFG9 using MOEA/D with Tchebycheff decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D with Tchebycheff decomposition for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 285.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # MOEA/D parameters
        pop_size = 40  # number of subproblems
        neighborhood_size = 10  # neighborhood size for mating
        archive_max = 200  # max archive size (we keep all non-dominated)

        # Generate weight vectors using the simplex method
        weights = self._generate_weights(pop_size, n_obj, rng)

        # Build neighborhood
        neighbors = self._build_neighbors(weights, neighborhood_size)

        # Initialize population: one solution per subproblem
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            X[i] = lo + (hi - lo) * rng.random(dim)
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Archive: list of (x, f) for all non-dominated solutions
        archive_X = list(X)
        archive_F = list(F)

        # Tchebycheff scalarization
        def tchebycheff(f, w, zmin, zmax):
            """Tchebycheff scalarization value."""
            span = zmax - zmin
            span[span < 1e-10] = 1.0
            vals = np.abs(w * (f - zmin) / span)
            return np.max(vals)

        # Update reference points
        def update_refs(F):
            zmin = np.min(F, axis=0)
            zmax = np.max(F, axis=0)
            return zmin, zmax

        zmin = np.min(F, axis=0)
        zmax = np.max(F, axis=0)

        # Fitness for each subproblem
        def subproblem_fitness(F_pop, i):
            """Compute Tchebycheff fitness for all solutions in subproblem i."""
            return np.array([tchebycheff(f, weights[i], zmin, zmax) for f in F_pop])

        # Mutation: polynomial
        def polynomial_mutation(x, eta=20.0):
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    range_i = hi[i] - lo[i]
                    x_mut[i] += delta_q * range_i
            return np.clip(x_mut, lo, hi)

        # Crossover: simulated binary
        def sbx_crossover(x1, x2, eta=15.0):
            if rng.random() > 0.9:
                return x1.copy(), x2.copy()
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        # Non-dominated check
        def is_non_dominated(f, F_archive):
            """Check if f is non-dominated w.r.t. F_archive."""
            for fa in F_archive:
                if np.all(fa <= f) and np.any(fa < f):
                    return False
            return True

        # Remove dominated from archive
        def remove_dominated(F_archive, F_to_remove):
            """Remove solutions from F_archive that are dominated by any in F_to_remove."""
            keep = []
            for i, fa in enumerate(F_archive):
                dominated = False
                for fr in F_to_remove:
                    if np.all(fr <= fa) and np.any(fr < fa):
                        dominated = True
                        break
                if not dominated:
                    keep.append(i)
            return keep

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Update reference points
            zmin = np.min(F, axis=0)
            zmax = np.max(F, axis=0)

            # For each subproblem, generate offspring
            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select mating pool from neighborhood
                mating_pool = neighbors[i]
                # Pick 2 parents from mating pool
                p1_idx = mating_pool[rng.integers(0, len(mating_pool))]
                p2_idx = mating_pool[rng.integers(0, len(mating_pool))]

                # Crossover and mutation
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)

                # Evaluate
                f1 = problem(c1)
                used += 1
                if used >= budget:
                    # Still add to archive if non-dominated
                    if is_non_dominated(f1, archive_F):
                        archive_X.append(c1)
                        archive_F.append(f1)
                        # Remove dominated from archive
                        keep = remove_dominated(archive_F, [f1])
                        archive_X = [archive_X[j] for j in keep]
                        archive_F = [archive_F[j] for j in keep]
                    break

                f2 = problem(c2)
                used += 1

                # Update archive with new solutions
                new_sols = [(c1, f1), (c2, f2)]
                for x_new, f_new in new_sols:
                    if is_non_dominated(f_new, archive_F):
                        archive_X.append(x_new)
                        archive_F.append(f_new)
                        # Remove dominated from archive
                        keep = remove_dominated(archive_F, [f_new])
                        archive_X = [archive_X[j] for j in keep]
                        archive_F = [archive_F[j] for j in keep]

                # Update population using Tchebycheff fitness
                # Combine current solution, offspring, and some neighbors
                candidate_X = [X[i], c1, c2]
                candidate_F = [F[i], f1, f2]
                # Add a few neighbors for diversity
                for nb in mating_pool[:3]:
                    candidate_X.append(X[nb])
                    candidate_F.append(F[nb])

                # Select best by Tchebycheff fitness for subproblem i
                fitnesses = np.array([tchebycheff(f, weights[i], zmin, zmax) for f in candidate_F])
                best_idx = np.argmin(fitnesses)
                X[i] = candidate_X[best_idx]
                F[i] = candidate_F[best_idx]

        # Return archive (all non-dominated solutions found)
        # The evaluator records all returned objective vectors
        if archive_F:
            return np.array(archive_F)
        return F

    def _generate_weights(self, pop_size, n_obj, rng):
        """Generate weight vectors using a simplex-based approach."""
        # Use a grid-based approach for 3 objectives
        # Generate points on the simplex that sum to 1
        weights = np.zeros((pop_size, n_obj))
        
        if n_obj == 3:
            # Use a stratified approach: generate points on the 2D simplex
            # w1 + w2 + w3 = 1, all >= 0
            # Use a grid with some randomization
            # For pop_size points, use a combination of systematic and random
            # Systematic: use a grid in (w1, w2) space where w3 = 1 - w1 - w2 >= 0
            # Random: sample uniformly from simplex
            
            # Mix: 70% systematic grid, 30% random
            n_sys = int(pop_size * 0.7)
            n_rnd = pop_size - n_sys
            
            # Systematic grid
            # Use a triangular grid
            points = []
            # For a grid of size k x k, we get roughly k*(k+1)/2 points
            # Find k such that k*(k+1)/2 >= n_sys
            k = int(np.ceil((-1 + np.sqrt(1 + 8 * n_sys)) / 2))
            
            for i in range(k + 1):
                for j in range(k + 1 - i):
                    w1 = i / k
                    w2 = j / k
                    w3 = 1 - w1 - w2
                    if w3 >= -1e-10:
                        points.append([w1, w2, max(0, w3)])
                    if len(points) >= n_sys:
                        break
                if len(points) >= n_sys:
                    break
            
            # Random points on simplex
            for _ in range(n_rnd):
                u1 = rng.random()
                u2 = rng.random()
                # Method for uniform sampling on simplex
                r1 = rng.random() ** (1 / (n_obj - 1))
                r2 = rng.random() ** (1 / (n_obj - 2))
                w = np.array([r1 * r2, r1 * (1 - r2), 1 - r1])
                points.append(w)
            
            # Take first pop_size points
            while len(points) < pop_size:
                r1 = rng.random() ** (1 / (n_obj - 1))
                r2 = rng.random() ** (1 / (n_obj - 2))
                w = np.array([r1 * r2, r1 * (1 - r2), 1 - r1])
                points.append(w)
            
            weights = np.array(points[:pop_size])
        else:
            # General case: random sampling from simplex
            for i in range(pop_size):
                r = rng.random(n_obj - 1) ** (1 / (n_obj - 1))
                r = r / r.sum()
                weights[i, :-1] = r
                weights[i, -1] = 1 - r.sum()
        
        # Normalize to ensure they sum to 1
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        return weights

    def _build_neighbors(self, weights, neighborhood_size):
        """Build neighborhood lists based on weight vector distance."""
        n = len(weights)
        neighbors = []
        for i in range(n):
            # Compute distances to all other weight vectors
            dists = np.array([np.linalg.norm(weights[i] - weights[j]) for j in range(n) if j != i])
            indices = [j for j in range(n) if j != i]
            # Sort by distance
            sorted_idx = [indices[k] for k in np.argsort(dists)]
            # Take neighborhood_size nearest neighbors
            neighbors.append(sorted_idx[:neighborhood_size])
        return neighbors
# EVOLVE-BLOCK-END
