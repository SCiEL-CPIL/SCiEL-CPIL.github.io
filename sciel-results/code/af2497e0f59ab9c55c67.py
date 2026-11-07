import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit

        if budget < 2:
            X = lo + (hi - lo) * np.random.random((max(1, budget), dim))
            F = np.array([problem(x) for x in X])
            return X

        # Phase 1: Sobol lattice initialization
        # Use power-of-2 number of points for Sobol sequence
        n_sobol = 1
        while n_sobol < min(budget // 2, 128):
            n_sobol *= 2
        
        # Ensure we don't exceed budget
        n_sobol = min(n_sobol, budget)
        
        # Generate Sobol sequence in [0,1]^dim
        sobol_unit = self._sobol_sequence(n_sobol, dim)
        
        # Map to bounds
        X_sobol = lo + sobol_unit * (hi - lo)
        
        # Evaluate initial lattice
        F_sobol = np.array([problem(x) for x in X_sobol], dtype=np.float64)
        used = n_sobol
        
        # Store all evaluated points
        all_X = X_sobol.copy()
        all_F = F_sobol.copy()
        
        # Phase 2: Iterative Nelder-Mead refinement on non-dominated points
        remaining_budget = budget - used
        
        # Get non-dominated points from initial set
        nd_indices = self._get_nondominated_indices(F_sobol)
        
        if len(nd_indices) == 0:
            # Fallback: random points
            while used < budget:
                x = lo + (hi - lo) * np.random.random(dim)
                f = problem(x)
                all_X = np.vstack([all_X, x.reshape(1, -1)])
                all_F = np.vstack([all_F, f.reshape(1, -1)])
                used += 1
            return all_X
        
        # Sort non-dominated points by sum of objectives (proxy for quality)
        nd_sums = F_sobol[nd_indices].sum(axis=1)
        nd_sorted = nd_indices[np.argsort(nd_sums)]
        
        # Refine the most promising points with Nelder-Mead
        max_refinements = min(len(nd_sorted), remaining_budget // (dim + 1) // 2)
        max_refinements = max(1, min(max_refinements, 15))
        
        for idx in nd_sorted[:max_refinements]:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Nelder-Mead on each objective separately, weighted
            x_start = all_X[idx].copy()
            
            # Use a weighted sum approach for multi-objective
            # Weights based on inverse of objective values (normalized)
            weights = 1.0 / (np.abs(F_sobol[idx]) + 1e-10)
            weights /= weights.sum()
            
            # Run Nelder-Mead with remaining budget
            n_evals_allowed = min(budget - used, 50)
            x_opt, f_opt, evals_used = self._nelder_mead(
                problem, x_start, lo, hi, weights, n_evals_allowed, deadline
            )
            
            used += evals_used
            if evals_used > 0:
                all_X = np.vstack([all_X, x_opt.reshape(1, -1)])
                all_F = np.vstack([all_F, f_opt.reshape(1, -1)])
        
        # Phase 3: Use remaining budget for additional Sobol points or random
        if used < budget:
            n_remaining = budget - used
            if time.time() < deadline:
                # Generate more Sobol points
                n_extra = min(n_remaining, 64)
                sobol_extra_unit = self._sobol_sequence(n_extra + n_sobol, dim)[n_sobol:]
                X_extra = lo + sobol_extra_unit * (hi - lo)
                
                for x in X_extra:
                    if used >= budget or time.time() > deadline:
                        break
                    f = problem(x)
                    all_X = np.vstack([all_X, x.reshape(1, -1)])
                    all_F = np.vstack([all_F, f.reshape(1, -1)])
                    used += 1
            
            # Fill any remaining with random
            while used < budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(dim)
                f = problem(x)
                all_X = np.vstack([all_X, x.reshape(1, -1)])
                all_F = np.vstack([all_F, f.reshape(1, -1)])
                used += 1
        
        return all_X

    def _sobol_sequence(self, n, dim):
        """
        Generate n points from a Sobol sequence in [0,1]^dim.
        Uses a simple implementation based on Gray codes.
        """
        # For simplicity, use a deterministic quasi-random sequence
        # that approximates Sobol properties
        points = np.zeros((n, dim))
        
        # Use Halton-like sequence as approximation
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131]
        
        for i in range(n):
            for d in range(dim):
                prime = primes[d % len(primes)]
                points[i, d] = self._halton_value(i + 1, prime)
        
        return points

    def _halton_value(self, index, base):
        """
        Compute the index-th Halton value in the given base.
        """
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    def _get_nondominated_indices(self, F):
        """
        Get indices of non-dominated points in F.
        """
        n = F.shape[0]
        nd_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not nd_mask[i]:
                continue
            for j in range(n):
                if i == j or not nd_mask[j]:
                    continue
                if self._dominates(F[j], F[i]):
                    nd_mask[i] = False
                    break
        
        return np.where(nd_mask)[0]

    def _dominates(self, a, b):
        """
        Returns True if a dominates b.
        """
        return np.all(a <= b) and np.any(a < b)

    def _nelder_mead(self, problem, x0, lo, hi, weights, max_evals, deadline):
        """
        Nelder-Mead optimization for a weighted sum of objectives.
        Returns (best_x, best_f, evals_used).
        """
        dim = len(x0)
        
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        
        # Small initial step
        step = 0.05 * (hi - lo)
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            simplex[i, i-1] += step[i-1]
            simplex[i] = np.clip(simplex[i], lo, hi)
        
        # Evaluate initial simplex
        f_values = np.zeros(dim + 1)
        evals_used = 0
        
        for i in range(dim + 1):
            if evals_used >= max_evals or time.time() > deadline:
                break
            f_values[i] = self._weighted_objective(problem, simplex[i], weights)
            evals_used += 1
        
        if evals_used < dim + 1:
            # Not enough evaluations to form simplex, return best
            best_idx = np.argmin(f_values[:evals_used])
            return simplex[best_idx], problem(simplex[best_idx]), evals_used
        
        # Nelder-Mead iterations
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        max_iter = max_evals - evals_used
        tol = 1e-10
        
        for _ in range(max_iter):
            if evals_used >= max_evals or time.time() > deadline:
                break
            
            # Sort simplex by objective value
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            # Check convergence
            if np.max(f_values) - np.min(f_values) < tol:
                break
            
            x_worst = simplex[-1]
            x_best = simplex[0]
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            x_r = np.clip(x_r, lo, hi)
            f_r = self._weighted_objective(problem, x_r, weights)
            evals_used += 1
            
            if f_r < f_values[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_worst)
                x_e = np.clip(x_e, lo, hi)
                f_e = self._weighted_objective(problem, x_e, weights)
                evals_used += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    x_c = x_centroid + rho * (x_r - x_worst)
                else:
                    x_c = x_centroid - rho * (x_worst - x_centroid)
                x_c = np.clip(x_c, lo, hi)
                f_c = self._weighted_objective(problem, x_c, weights)
                evals_used += 1
                
                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        f_values[i] = self._weighted_objective(problem, simplex[i], weights)
                        evals_used += 1
                        if evals_used >= max_evals or time.time() > deadline:
                            break
            
            if evals_used >= max_evals or time.time() > deadline:
                break
        
        # Return best point
        order = np.argsort(f_values)
        best_idx = order[0]
        best_x = simplex[best_idx]
        best_f = problem(best_x)
        
        return best_x, best_f, evals_used

    def _weighted_objective(self, problem, x, weights):
        """
        Compute weighted sum of objectives.
        """
        f = np.array(problem(x), dtype=np.float64)
        return np.dot(f, weights)
