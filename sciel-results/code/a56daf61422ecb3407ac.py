"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Deterministic global search using Differential Evolution followed by Multi-start L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Count function calls
        call_count = [0]

        def counted_func(x):
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count[0] += 1
            return val

        # --- Phase 1: Differential Evolution (global search) ---
        rng = np.random.default_rng(42)

        # Tuned population size for 5D Rosenbrock: larger for better valley exploration
        pop_size = 40
        # Allocate 80% of budget to DE for thorough global search
        de_budget = int(budget * 0.8)
        
        # Initial population: uniform in [-5, 5]^dim with some concentration
        # Mix of uniform and slightly concentrated samples
        pop1 = rng.uniform(-5.0, 5.0, size=(pop_size // 2, dim))
        pop2 = rng.uniform(-3.0, 3.0, size=(pop_size - pop_size // 2, dim))
        pop = np.vstack([pop1, pop2])
        pop_f = np.array([counted_func(ind) for ind in pop])

        # DE parameters tuned for Rosenbrock-like functions
        F = 0.8  # slightly higher scaling for better exploration
        CR = 0.95  # higher crossover for better exploitation

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE Loop with adaptive F
        gen = 0
        no_improve = 0
        prev_best_f = best_f
        
        while call_count[0] < de_budget:
            # Find current best
            best_idx = np.argmin(pop_f)
            best_x = pop[best_idx].copy()
            best_f = pop_f[best_idx]
            
            # Adaptive F: decrease if no improvement for several generations
            if best_f < prev_best_f - 1e-6:
                no_improve = 0
                prev_best_f = best_f
            else:
                no_improve += 1
            
            # Adapt F: start high, decrease when stuck
            F_current = max(0.4, F - 0.05 * no_improve)
            
            for i in range(pop_size):
                if call_count[0] >= de_budget:
                    break

                # Select three distinct random indices different from i
                mask = np.ones(pop_size, dtype=bool)
                mask[i] = False
                candidates = np.where(mask)[0]
                if len(candidates) < 3:
                    break
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F_current * (pop[b] - pop[c])

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                trial_f = counted_func(trial)

                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f

        # After DE, find best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # --- Phase 2: Single high-precision L-BFGS-B polish on best candidate ---
        remaining = budget - call_count[0]
        if remaining > 5:
            bounds = [(-5.0, 5.0)] * dim
            
            try:
                result = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': remaining, 'maxfun': remaining, 'ftol': 1e-12}
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
