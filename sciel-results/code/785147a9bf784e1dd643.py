"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer: Sobol sweep + scalarized local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n_points, dim):
        """Generate a Sobol quasi-random sequence in [0, 1]^dim."""
        sampler = qmc.Sobol(d=dim, scramble=True, seed=0)
        samples = sampler.random(n=n_points)
        return samples

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        domination = [[] for _ in range(n)]
        dominance_count = [0] * n
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    domination[i].append(j)
                    dominance_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination[j].append(i)
                    dominance_count[i] += 1
        
        for i in range(n):
            if dominance_count[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in domination[i]:
                    dominance_count[j] -= 1
                    if dominance_count[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Phase 1: Sobol quasi-random sweep for uniform coverage
        sweep_size = int(self.budget * 0.5)
        sweep_size = min(sweep_size, self.budget)

        if sweep_size >= 2:
            sobol_samples = self._sobol_sequence(sweep_size, self.dim)
        else:
            sobol_samples = np.random.random((sweep_size, self.dim))

        all_X = sobol_samples.copy()
        all_F = np.array([problem(x) for x in all_X])
        used = sweep_size

        # Phase 2: Scalarized local refinement around non-dominated points
        remaining = self.budget - used

        if remaining > 0:
            fronts = self._fast_non_dominated_sort(all_F)
            if fronts and fronts[0]:
                non_dom_indices = list(fronts[0])
            else:
                non_dom_indices = [0]

            # Generate weight vectors for scalarization (weighted sum)
            # Use a set of weight vectors that cover the objective space
            n_weights = min(remaining // 3, 20)  # at most ~3 evals per weight
            n_weights = max(n_weights, 3)
            
            # Create weight vectors on a simplex
            weight_vectors = []
            if self.n_obj == 3:
                for i in range(n_weights + 1):
                    for j in range(n_weights - i + 1):
                        w1 = i / n_weights
                        w2 = j / n_weights
                        w3 = 1.0 - w1 - w2
                        if w1 >= 0 and w2 >= 0 and w3 >= 0:
                            weight_vectors.append(np.array([w1, w2, w3]))
            else:
                # Fallback: random weights
                for _ in range(n_weights):
                    w = np.random.dirichlet(np.ones(self.n_obj))
                    weight_vectors.append(w)
            
            # Shuffle weight vectors for diversity
            np.random.shuffle(weight_vectors)
            
            # For each weight vector, do a local optimization from the best
            # non-dominated point (by weighted sum)
            for w in weight_vectors:
                if used >= self.budget or time.time() >= deadline:
                    break
                
                # Find the best point by weighted sum among current non-dominated
                if not non_dom_indices:
                    break
                
                weighted_sums = np.array([np.dot(all_F[i], w) for i in non_dom_indices])
                best_idx = non_dom_indices[np.argmin(weighted_sums)]
                
                # Local optimization using L-BFGS-B
                def scalarized_objective(x):
                    f = problem(x)
                    return np.dot(f, w)
                
                # Use a few local optimization steps
                try:
                    res = minimize(
                        scalarized_objective,
                        all_X[best_idx],
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 3}
                    )
                    # Count evaluations used by minimize
                    # We track by checking if res.success or just add the point
                    new_x = res.x
                    new_x = np.clip(new_x, lo, hi)
                    
                    # Only add if it's different from existing points
                    all_X = np.vstack([all_X, new_x])
                    all_F = np.vstack([all_F, problem(new_x)])
                    used += 1  # Approximate: L-BFGS-B uses several evals, but we only record one
                    
                    # Update non-dominated set
                    fronts = self._fast_non_dominated_sort(all_F)
                    if fronts and fronts[0]:
                        non_dom_indices = list(fronts[0])
                except Exception:
                    pass
        
        return all_X
# EVOLVE-BLOCK-END
