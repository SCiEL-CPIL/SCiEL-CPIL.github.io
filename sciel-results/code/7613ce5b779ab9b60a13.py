"""Differential Evolution with adaptive strategies and multi-start Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based optimizer with adaptive mutation strategies and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        deadline = time.time() + 270  # safety margin before 300s limit
        self._call_count = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal best_x, best_f
            if self._call_count >= self.budget:
                return np.inf
            self._call_count += 1
            val = func(np.clip(x, -5, 5))
            if val < best_f:
                best_f = val
                best_x = np.clip(x, -5, 5).copy()
            return val

        def nelder_mead_local(start_x, max_evals):
            """Quick local NM polish around start_x."""
            nonlocal best_x, best_f, deadline
            if self._call_count >= self.budget or time.time() > deadline:
                return
            x0 = np.clip(start_x, -5, 5)
            f0 = counted_func(x0)

            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                step = 0.05
                xi[i] = x0[i] + step if x0[i] < 4.95 else x0[i] - step
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, self.dim + 1):
                simplex_f.append(counted_func(simplex[i]))

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            iters = 0
            while iters < max_evals and self._call_count < self.budget and time.time() < deadline:
                iters += 1
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = counted_func(xr)

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = counted_func(xe)
                    simplex[-1] = xe if fe < fr else xr
                    simplex_f[-1] = fe if fe < fr else fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = counted_func(xc)
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = counted_func(simplex[i])

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # DE parameters
        NP = 20  # population size
        F = 0.8   # differential weight
        CR = 0.5  # crossover rate
        # Adaptive F and CR
        F_min, F_max = 0.4, 1.0
        CR_min, CR_max = 0.3, 0.9

        # Initialize population
        pop = rng.uniform(-5, 5, (NP, dim))
        pop_f = np.full(NP, np.inf)

        for i in range(NP):
            if self._call_count >= budget or time.time() > deadline:
                break
            f = counted_func(pop[i])
            pop_f[i] = f

        # Track global best
        global_best_idx = np.argmin(pop_f)
        global_best = pop[global_best_idx].copy()
        global_best_f = pop_f[global_best_idx]

        # Adaptive strategy tracking
        # Strategies: 0=DE/rand/1, 1=DE/best/1, 2=DE/current-to-best/1
        strategy_success = np.zeros(3)
        strategy_attempts = np.zeros(3)
        strategy_counts = np.array([NP // 3, NP // 3, NP - 2 * (NP // 3)])

        # DE Phase: use ~70% of budget
        de_budget = int(budget * 0.70)
        max_de_iters = de_budget // NP
        stagnation = 0
        last_best_f = global_best_f

        for it in range(max_de_iters):
            if self._call_count >= budget or time.time() > deadline:
                break

            # Adaptive F and CR based on progress
            progress = min(1.0, (it + 1) / max_de_iters)
            F_adapt = F_min + (F_max - F_min) * (1 - progress)
            CR_adapt = CR_min + (CR_max - CR_min) * (1 - progress)

            # Select strategy for each individual based on success rates
            # Weighted random selection
            total_success = strategy_success.sum() + 1e-10
            probs = strategy_success / total_success
            # Ensure minimum probability for exploration
            probs = 0.7 * probs + 0.3 / 3

            new_pop = np.copy(pop)
            new_pop_f = np.full(NP, np.inf)
            improvements = 0

            for i in range(NP):
                if self._call_count >= budget or time.time() > deadline:
                    break

                # Choose strategy
                strat = rng.choice(3, p=probs)
                strategy_attempts[strat] += 1

                # Generate trial vector
                if strat == 0:  # DE/rand/1
                    r1, r2 = np.random.choice(NP, 2, replace=False)
                    v = pop[r1] + F_adapt * (pop[r2] - pop[r3 if (r3 := rng.integers(NP)) != r1 and r3 != r2 else (r1 + 1) % NP])
                elif strat == 1:  # DE/best/1
                    r1, r2 = np.random.choice(NP, 2, replace=False)
                    while r1 == global_best_idx or r2 == global_best_idx:
                        r1, r2 = np.random.choice(NP, 2, replace=False)
                    v = global_best + F_adapt * (pop[r1] - pop[r2])
                else:  # DE/current-to-best/1
                    r1 = np.random.choice(NP)
                    while r1 == i:
                        r1 = np.random.choice(NP)
                    v = pop[i] + F_adapt * (global_best - pop[i]) + F_adapt * 0.5 * (global_best - pop[r1])

                # Binary crossover
                j_rand = rng.integers(dim)
                trial = np.copy(pop[i])
                for j in range(dim):
                    if rng.random() < CR_adapt or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5, 5)
                f_trial = counted_func(trial)
                new_pop[i] = trial
                new_pop_f[i] = f_trial

                # Selection
                if f_trial < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
                    improvements += 1
                    strategy_success[strat] += 1
                    if f_trial < global_best_f:
                        global_best_f = f_trial
                        global_best = trial.copy()
                        global_best_idx = i
                        stagnation = 0
                else:
                    pop_f[i] = min(pop_f[i], f_trial)  # keep best of old and trial

            # Update global best index
            global_best_idx = np.argmin(pop_f)
            if pop_f[global_best_idx] < global_best_f:
                global_best_f = pop_f[global_best_idx]
                global_best = pop[global_best_idx].copy()
                stagnation = 0
            else:
                stagnation += 1

            # Decay strategy success rates
            strategy_success *= 0.95

            # Early stop if severe stagnation
            if stagnation > 5 * NP:
                break

        # Local refinement phase: use remaining budget
        polish_budget = budget - self._call_count
        if polish_budget > 10 and best_x is not None:
            # Multi-start NM from diverse points
            # 1. Global best
            alloc1 = min(polish_budget // 3, polish_budget)
            nelder_mead_local(global_best, alloc1)

            if self._call_count < budget:
                # 2. Second best
                sorted_idx = np.argsort(pop_f)
                if sorted_idx[1] != global_best_idx:
                    alloc2 = min(polish_budget // 3, budget - self._call_count)
                    nelder_mead_local(pop[sorted_idx[1]], alloc2)

            if self._call_count < budget:
                # 3. A diverse point (far from global best)
                dists = np.linalg.norm(pop - global_best, axis=1)
                diverse_idx = np.argmax(dists)
                alloc3 = max(5, budget - self._call_count)
                nelder_mead_local(pop[diverse_idx], alloc3)

        return best_x, best_f
# EVOLVE-BLOCK-END
