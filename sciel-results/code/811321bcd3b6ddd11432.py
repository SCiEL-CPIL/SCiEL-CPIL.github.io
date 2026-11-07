# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using a deterministic space-filling lattice (Halton sequence)
    for initial coverage, followed by iterative local refinement (Nelder-Mead) on the most
    promising non-dominated points to converge to the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim, base=None):
        """Generate n points in [0,1]^dim using Halton sequence."""
        if base is None:
            # First 'dim' primes
            primes = []
            candidate = 2
            while len(primes) < dim:
                is_prime = True
                for p in range(2, int(np.sqrt(candidate)) + 1):
                    if candidate % p == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(candidate)
                candidate += 1
            base = primes

        points = np.zeros((n, dim))
        for d in range(dim):
            b = base[d]
            f = 1.0 / b
            for i in range(1, n + 1):
                rad = i
                result = 0.0
                while rad > 0:
                    rad, rem = divmod(rad, b)
                    result += rem * f
                    f /= b
                points[i - 1, d] = result
        return points

    def _dominance_mask(self, F):
        """
        Return a boolean array where True means the point is non-dominated.
        F: (N, n_obj)
        """
        N = F.shape[0]
        non_dominated = np.ones(N, dtype=bool)
        for i in range(N):
            if non_dominated[i]:
                # Check if any other point dominates i
                # Point j dominates i if F[j] <= F[i] in all objectives and F[j] < F[i] in at least one
                dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                dominated[i] = False  # Ignore self
                if np.any(dominated):
                    non_dominated[i] = False
        return non_dominated

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Step 1: Generate initial space-filling sample using Halton sequence
        # Use a larger initial sample for better coverage
        n_init = min(150, budget - 50)
        if n_init < 20:
            n_init = 20

        halton_points = self._halton(n_init, dim)
        X = lo + (hi - lo) * halton_points
        F = np.array([problem(x) for x in X])
        used = n_init

        # Step 2: Identify non-dominated points
        non_dom_mask = self._dominance_mask(F)
        non_dom_indices = np.where(non_dom_mask)[0]

        # Step 3: Iteratively refine the most promising non-dominated points using Nelder-Mead
        # We'll pick non-dominated points and run local optimization on them
        # Budget remaining
        remaining = budget - used

        if remaining > 0 and len(non_dom_indices) > 0:
            # Sort non-dominated points by sum of objectives (lower is better) to pick most promising
            sums = np.sum(F[non_dom_indices], axis=1)
            sorted_indices = non_dom_indices[np.argsort(sums)]

            # Determine how many points to refine
            # Each Nelder-Mead run uses approximately 2*dim + 10 evaluations (rough estimate)
            # Let's be conservative: assume ~25 evals per refinement
            n_refine = min(len(sorted_indices), max(1, remaining // 25))

            for idx in sorted_indices[:n_refine]:
                if used >= budget - 5:
                    break

                x0 = X[idx].copy()

                # Define a scalar objective: weighted sum with emphasis on different objectives
                # Use different weightings to explore different parts of the Pareto front
                weight_idx = sorted_indices.tolist().index(idx)
                weights = np.ones(n_obj) / n_obj
                # Vary weights slightly based on index to explore different regions
                if weight_idx % n_obj == 0:
                    weights[0] = 2.0
                elif weight_idx % n_obj == 1:
                    weights[1] = 2.0
                elif weight_idx % n_obj == 2:
                    weights[2] = 2.0
                weights = weights / np.sum(weights)

                def scalar_obj(x, w=weights):
                    x_clipped = np.clip(x, lo, hi)
                    f = problem(x_clipped)
                    return np.dot(w, f)

                # Run Nelder-Mead with tighter tolerances
                try:
                    result = minimize(
                        scalar_obj,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5}
                    )
                    x_new = np.clip(result.x, lo, hi)
                    f_new = problem(x_new)
                    used += result.nfev

                    # Add the new point to the population
                    X = np.vstack([X, x_new])
                    F = np.vstack([F, f_new])
                except Exception:
                    pass

        # Final: return the best points (we just return X as per interface, but the evaluator records all F)
        # The interface says "return X" but the evaluator records all objective vectors from problem calls.
        # So we just need to make sure we've called problem() within budget.
        return X
# EVOLVE-BLOCK-END
