"""Optimized multi-objective optimizer for WFG7 using Tchebycheff decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations_with_replacement


class Optimizer:
    """Decomposition-based multi-objective optimizer using Tchebycheff scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        span = hi - lo
        
        # Generate weight vectors uniformly on the simplex
        n_weights = 15
        # Use Dirichlet distribution for uniform simplex points
        rng = np.random.default_rng(42)
        weights = []
        for _ in range(n_weights):
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)
        weights = np.array(weights)
        
        # Reserve some budget for initial random sampling
        n_init = 20
        n_per_weight = max(1, (budget - n_init) // n_weights)
        
        all_x = []
        all_f = []
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            f = problem(x)
            eval_count += 1
            all_x.append(x.copy())
            all_f.append(np.array(f))
            return f
        
        # Initial random sampling
        for _ in range(n_init):
            x = lo + span * rng.random(dim)
            x = np.clip(x, lo, hi)
            evaluate(x)
        
        # For each weight vector, run a local search
        for w in weights:
            # Start from center or a random point
            x = mid + span * (rng.random(dim) - 0.5) * 0.3
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            if f is None:
                break
            
            best_x = x.copy()
            best_f = f.copy()
            
            # Simple local search: try small perturbations
            step_size = 0.05
            for iteration in range(n_per_weight):
                if eval_count >= budget:
                    break
                # Try a few random directions
                found_improvement = False
                for _ in range(3):
                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction) + 1e-10
                    for step in [step_size, step_size * 0.5, step_size * 2]:
                        x_new = best_x + step * span * direction
                        x_new = np.clip(x_new, lo, hi)
                        f_new = evaluate(x_new)
                        if f_new is None:
                            break
                        # Tchebycheff scalarization
                        z_min = np.array([min(f_i[i] for f_i in all_f) for i in range(n_obj)])
                        z_max = np.array([max(f_i[i] for f_i in all_f) for i in range(n_obj)])
                        z_range = z_max - z_min + 1e-10
                        scalar_old = np.max(w * np.abs(best_f - z_min) / z_range)
                        scalar_new = np.max(w * np.abs(f_new - z_min) / z_range)
                        if scalar_new < scalar_old:
                            best_x = x_new.copy()
                            best_f = f_new.copy()
                            found_improvement = True
                            break
                    if found_improvement:
                        break
                if not found_improvement:
                    step_size *= 0.7
        
        # Return non-dominated set
        if not all_f:
            return np.array([mid])
        
        all_f = np.array(all_f)
        all_x = np.array(all_x)
        
        # Find non-dominated points
        n_points = len(all_f)
        dominated = np.zeros(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    continue
                if np.all(all_f[j] <= all_f[i]) and np.any(all_f[j] < all_f[i]):
                    dominated[i] = True
                    break
        
        non_dominated_idx = np.where(~dominated)[0]
        return all_x[non_dominated_idx]
# EVOLVE-BLOCK-END
