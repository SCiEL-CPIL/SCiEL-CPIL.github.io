"""Rewritten optimizer: Differential Evolution + L-BFGS-B polish with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE global search followed by L-BFGS-B local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0

        # Wrapper to count calls and enforce deadline
        call_count = 0
        best_x = None
        best_f = np.inf

        def counted(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            call_count += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize with deterministic spread + small random perturbation
        rng = np.random.default_rng(42)

        # --- Phase 1: Differential Evolution ---
        # Population size: small for 500 budget
        pop_size = 20
        # Initial population: grid + random
        pop = np.zeros((pop_size, dim))
        # Use a stratified initial population
        for i in range(pop_size):
            if i < 8:
                # Corner-ish points
                pop[i] = rng.uniform(lo, hi, dim)
            elif i < 16:
                # Mid-range points
                pop[i] = rng.uniform(-2, 2, dim)
            else:
                # Random
                pop[i] = rng.uniform(lo, hi, dim)

        pop_f = np.array([counted(pop[i]) for i in range(pop_size)])

        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        # Adaptive CR
        de_calls = pop_size
        max_de_calls = int(budget * 0.6)  # use 60% of budget for DE

        # Track best
        de_best_idx = np.argmin(pop_f)
        de_best_x = pop[de_best_idx].copy()
        de_best_f = pop_f[de_best_idx]

        gen = 0
        while de_calls < max_de_calls and time.time() < deadline:
            gen += 1
            # Adaptive F and CR based on generation
            F_gen = F * (0.95 ** (gen % 10))  # slight decay then reset
            CR_gen = CR + 0.1 * np.sin(gen * 0.5)
            CR_gen = np.clip(CR_gen, 0.7, 0.95)

            for i in range(pop_size):
                if de_calls >= max_de_calls or time.time() >= deadline:
                    break

                # Select three distinct individuals
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Mutation: DE/rand/1
                v = pop[a] + F_gen * (pop[b] - pop[c])
                v = np.clip(v, lo, hi)

                # Crossover
                r = rng.random()
                j_rand = rng.integers(0, dim)
                trial = np.where((rng.random(dim) <= CR_gen) | (np.arange(dim) == j_rand), v, pop[i])
                trial = np.clip(trial, lo, hi)

                # Evaluate
                trial_f = counted(trial)
                de_calls += 1

                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < de_best_f:
                        de_best_f = trial_f
                        de_best_x = trial.copy()
                        de_best_idx = i

        # --- Phase 2: L-BFGS-B polish on best candidates ---
        # Take top few candidates from DE and polish them
        top_indices = np.argsort(pop_f)[:5]
        polish_candidates = [pop[i].copy() for i in top_indices]
        # Also include the overall best
        if best_x is not None:
            polish_candidates.append(best_x.copy())

        # Deduplicate
        unique_candidates = []
        for c in polish_candidates:
            if not any(np.allclose(c, u, atol=1e-6) for u in unique_candidates):
                unique_candidates.append(c)

        # Polish each candidate with L-BFGS-B
        bounds = [(lo, hi)] * dim
        polish_budget = budget - de_calls - len(unique_candidates)  # approximate remaining

        for cand in unique_candidates:
            if time.time() >= deadline or call_count >= budget:
                break
            try:
                res = minimize(
                    counted,
                    cand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 3: If budget remains, do a short (1+1)-ES polish ---
        if time.time() < deadline and call_count < budget:
            if best_x is None:
                best_x = np.zeros(dim)
                best_f = counted(best_x)
                call_count += 1

            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            for _ in range(budget - call_count):
                if time.time() >= deadline or call_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), lo, hi)
                fy = counted(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
