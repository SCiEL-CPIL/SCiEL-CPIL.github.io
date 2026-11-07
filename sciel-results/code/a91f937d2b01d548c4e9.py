# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using a deterministic space-filling lattice (Halton)
    for initial coverage, followed by iterative local refinement (Nelder-Mead) on the most
    promising non-dominated points to converge to the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n points in [0,1]^dim using Halton sequence."""
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

        points = np.zeros((n, dim))
        for d in range(dim):
            b = primes[d]
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
        # Use a reasonable number of initial points
        n_init = min(80, max(20, budget // 3))
        
        halton_points = self._halton(n_init, dim)
        X = lo + (hi - lo) * halton_points
        F = np.array([problem(x) for x in X])
        used = n_init

        # Step 2: Identify non-dominated points
        non_dom_mask = self._dominance_mask(F)
        non_dom_indices = np.where(non_dom_mask)[0]

        # Step 3: Iteratively refine non-dominated points using Nelder-Mead
        # Use multiple weight vectors to cover different parts of the Pareto front
        remaining = budget - used

        if remaining > 0 and len(non_dom_indices) > 0:
            # Create weight vectors for different regions of the Pareto front
            # For 3 objectives, use a set of weight vectors
            n_weights = 7
            # Generate weight vectors: simplex-lattice-like
            weight_vectors = []
            # Add the 3 unit vectors
            for i in range(n_obj):
                w = np.zeros(n_obj)
                w[i] = 1.0
                weight_vectors.append(w)
            # Add 3 pair-wise combinations
            for i in range(n_obj):
                for j in range(i+1, n_obj):
                    w = np.zeros(n_obj)
                    w[i] = 0.5
                    w[j] = 0.5
                    weight_vectors.append(w)
            # Add the centroid
            weight_vectors.append(np.ones(n_obj) / n_obj)
            
            # For each weight vector, pick the best non-dominated point according to that weight
            # and refine it
            n_refine_per_weight = max(1, remaining // (len(weight_vectors) * 25))
            
            for w in weight_vectors:
                if used >= budget:
                    break
                
                # Pick the non-dominated point that minimizes the weighted sum
                weighted_sums = F[non_dom_indices] @ w
                best_local_idx = non_dom_indices[np.argmin(weighted_sums)]
                
                x0 = X[best_local_idx].copy()
                
                # Define a scalar objective using this weight vector
                def scalar_obj(x, w=w):
                    x_clipped = np.clip(x, lo, hi)
                    f = problem(x_clipped)
                    return np.dot(f, w)
                
                # Run Nelder-Mead
                try:
                    result = minimize(
                        scalar_obj,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-3, 'fatol': 1e-3}
                    )
                    x_new = np.clip(result.x, lo, hi)
                    f_new = problem(x_new)
                    used += result.nfev

                    # Add the new point to the population
                    X = np.vstack([X, x_new])
                    F = np.vstack([F, f_new])
                except Exception:
                    pass
        
        # Step 4: If we still have budget, do a second round of refinement
        if used < budget:
            # Re-identify non-dominated points
            non_dom_mask = self._dominance_mask(F)
            non_dom_indices = np.where(non_dom_mask)[0]
            
            remaining = budget - used
            if remaining > 0 and len(non_dom_indices) > 0:
                # Pick the top non-dominated points by hypervolume contribution (approximate)
                # Use a simple heuristic: sort by sum of objectives
                sums = np.sum(F[non_dom_indices], axis=1)
                sorted_indices = non_dom_indices[np.argsort(sums)]
                
                # Refine a few more points
                n_more = min(len(sorted_indices), max(1, remaining // 15))
                
                for idx in sorted_indices[:n_more]:
                    if used >= budget:
                        break
                    
                    x0 = X[idx].copy()
                    
                    # Use a slightly perturbed weight for diversity
                    w = np.ones(n_obj) / n_obj
                    w[np.argmin(F[idx])] += 0.3
                    w /= np.sum(w)
                    
                    def scalar_obj(x, w=w):
                        x_clipped = np.clip(x, lo, hi)
                        f = problem(x_clipped)
                        return np.dot(f, w)
                    
                    try:
                        result = minimize(
                            scalar_obj,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': 20, 'xatol': 1e-3, 'fatol': 1e-3}
                        )
                        x_new = np.clip(result.x, lo, hi)
                        f_new = problem(x_new)
                        used += result.nfev

                        X = np.vstack([X, x_new])
                        F = np.vstack([F, f_new])
                    except Exception:
                        pass

        return X
# EVOLVE-BLOCK-END
