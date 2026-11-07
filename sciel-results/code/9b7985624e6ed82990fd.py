"""Population-based Differential Evolution with adaptive mutation and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DE/rand/1 with adaptive F, population size 15, plus multi-restart Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 15
        # Budget allocation: DE phase uses most, polish uses ~60 calls
        polish_budget = 60
        de_budget = max(budget - polish_budget, 10)

        # Initialize population: LHS + Gaussian around center for better coverage
        pop = np.empty((pop_size, dim))
        # LHS for first half
        n_lhs = pop_size // 2
        lhs_samples = np.array([rng.permutation(pop_size) for _ in range(dim)]).T
        pop[:n_lhs] = lhs_samples[:n_lhs] / (n_lhs - 1) * 10 - 5
        # Gaussian for second half
        n_gauss = pop_size - n_lhs
        pop[n_lhs:] = rng.normal(0, 1.5, size=(n_gauss, dim))
        pop = np.clip(pop, -5, 5)

        fits = np.array([func(ind) for ind in pop])

        # Track best
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]

        # Adaptive mutation factor F: start at 0.6
        F = 0.6
        success_count = 0
        trial_count = 0
        last_best_f = best_f
        no_improve_gen = 0

        calls_used = pop_size  # initial population evaluations

        # DE loop
        while calls_used < de_budget:
            # Adaptive F: smooth adaptation based on recent success rate
            if trial_count >= 10:
                success_rate = success_count / trial_count
                target_F = 0.5 + 0.6 * success_rate
                F = 0.7 * F + 0.3 * np.clip(target_F, 0.3, 1.0)
                success_count = 0
                trial_count = 0

            # Check for stagnation
            if best_f < last_best_f - 1e-10:
                no_improve_gen = 0
                last_best_f = best_f
            else:
                no_improve_gen += 1

            # If stuck for many generations, inject diversity using best-guided mutation
            if no_improve_gen > 4:
                worst_indices = np.argsort(fits)[-3:]
                for idx in worst_indices:
                    if calls_used >= de_budget:
                        break
                    # Best-guided mutation: perturb best with random direction
                    rand_vec = rng.uniform(-1, 1, size=dim)
                    new_point = best_x + 0.5 * rand_vec
                    new_point = np.clip(new_point, -5, 5)
                    pop[idx] = new_point
                    fits[idx] = func(pop[idx])
                    calls_used += 1
                    if fits[idx] < best_f:
                        best_f = fits[idx]
                        best_x = pop[idx].copy()

            for i in range(pop_size):
                if calls_used >= de_budget:
                    break

                # Select three distinct random indices different from i
                candidates = np.array([j for j in range(pop_size) if j != i])
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # DE/rand/1 mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, -5, 5)

                # Evaluate mutant
                fm = func(mutant)
                calls_used += 1
                trial_count += 1

                # Selection: replace if mutant is better
                if fm < fits[i]:
                    pop[i] = mutant
                    fits[i] = fm
                    success_count += 1

                    # Update global best
                    if fm < best_f:
                        best_f = fm
                        best_x = mutant.copy()

        # Nelder-Mead polish: try multiple top solutions with careful call counting
        if calls_used < budget:
            call_counter = {'count': calls_used}

            def bounded_func(x):
                x_clipped = np.clip(x, -5, 5)
                call_counter['count'] += 1
                return func(x_clipped)

            # Get top 4 solutions from population for more diverse starts
            top_indices = np.argsort(fits)[:4]
            nm_starts = [pop[idx].copy() for idx in top_indices]
            # Also include best_x
            if not any(np.allclose(s, best_x, atol=1e-6) for s in nm_starts):
                nm_starts.append(best_x.copy())

            # Budget per NM run
            nm_total_budget = budget - calls_used
            n_starts = min(len(nm_starts), 4)
            
            for start in nm_starts[:n_starts]:
                if call_counter['count'] >= budget:
                    break

                remaining = budget - call_counter['count']
                maxiter = max(1, remaining // 5)
                maxfev = max(1, remaining)

                try:
                    result = minimize(
                        bounded_func,
                        start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': maxiter,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True,
                            'maxfev': maxfev
                        }
                    )
                    # Update calls_used
                    calls_used = call_counter['count']
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    calls_used = call_counter['count']

        return best_x, best_f
# EVOLVE-BLOCK-END
