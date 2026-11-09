"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """JADE-style adaptive Differential Evolution with local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def check_time():
            return calls_used < budget and time.time() < deadline - 5.0

        def check_time_early():
            return calls_used < budget and time.time() < deadline - 20.0

        rng = np.random.default_rng(12345)

        # Phase 1: Initial diverse sampling to get a good population
        # Use 2*dim + 4 initial points spread across the box
        n_init = 2 * dim + 4
        pop = []
        # Center
        pop.append(np.zeros(dim))
        # Axis-aligned points
        for axis in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[axis] = sign * 3.5
                pop.append(p)
        # Random fill
        while len(pop) < n_init:
            pop.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate initial population
        pop_evaluated = []
        for x in pop:
            if not check_time_early():
                break
            fx = func(x)
            calls_used += 1
            pop_evaluated.append((x, fx))
            try_update(x, fx)

        # If we have enough evaluated points, use them; otherwise continue with what we have
        if len(pop_evaluated) >= 4:
            # Use JADE-style DE
            pop = [x for x, f in pop_evaluated]
            pop_f = [f for x, f in pop_evaluated]

            # JADE parameters
            CR = 0.5
            F = 0.5
            tau1 = 0.1
            tau2 = 0.1
            p = 0.1  # probability of using pbest

            # Main DE loop
            while check_time():
                # Sort by fitness
                indices = np.argsort(pop_f)
                pop_sorted = [pop[i] for i in indices]
                pop_f_sorted = [pop_f[i] for i in indices]

                # Keep best half for next generation
                half = max(2, len(pop) // 2)
                new_pop = []
                new_pop_f = []

                # Preserve the best individuals
                for i in range(min(half, len(pop_sorted))):
                    new_pop.append(pop_sorted[i].copy())
                    new_pop_f.append(pop_f_sorted[i])

                # Generate new individuals using JADE mutation
                while len(new_pop) < len(pop) and check_time():
                    # Select victim
                    r1 = rng.integers(len(pop))
                    victim = pop[r1].copy()
                    victim_f = pop_f[r1]

                    # JADE mutation: x'_i = x_i + F1*(x_pbest - x_i) + F2*(x_r1 - x_r2)
                    # pbest selection
                    r2 = rng.integers(len(pop))
                    if rng.random() < p:
                        # Use pbest
                        # Find the best in a random subset
                        subset_size = max(1, int(p * len(pop)))
                        subset_indices = rng.choice(len(pop), size=subset_size, replace=False)
                        pbest_idx = subset_indices[np.argmin([pop_f[i] for i in subset_indices])]
                        x_pbest = pop[pbest_idx]
                    else:
                        x_pbest = pop_sorted[0]  # best individual

                    # Random distinct indices
                    r3 = rng.integers(len(pop))
                    r4 = rng.integers(len(pop))
                    while r3 == r4 or r3 == r2:
                        r3 = rng.integers(len(pop) - 1)
                        r4 = rng.integers(len(pop) - 1)
                        if r3 == r4:
                            r4 = (r4 + 1) % len(pop)
                        if r3 == r2:
                            r3 = (r3 + 1) % len(pop)
                        if r4 == r2:
                            r4 = (r4 + 1) % len(pop)

                    # Adaptive F
                    F1 = rng.uniform(0.1, 1.0)
                    F2 = rng.uniform(0.1, 1.0)

                    # Mutation
                    mutant = victim + F1 * (x_pbest - victim) + F2 * (pop[r3] - pop[r4])
                    mutant = np.clip(mutant, -5, 5)

                    # Evaluate
                    f_mutant = func(mutant)
                    calls_used += 1

                    # Selection
                    if f_mutant < victim_f:
                        new_pop.append(mutant)
                        new_pop_f.append(f_mutant)
                        try_update(mutant, f_mutant)
                    else:
                        # Keep a random individual from current population
                        keep_idx = rng.integers(len(pop))
                        new_pop.append(pop[keep_idx].copy())
                        new_pop_f.append(pop_f[keep_idx])

                # Update population
                pop = new_pop
                pop_f = new_pop_f

                # Adapt CR
                # Simple adaptation: reduce CR slightly over time
                CR = max(0.3, CR * 0.999)

        # Phase 2: Local refinement around best point
        if check_time() and not np.isnan(best_x).any():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            # Nelder-Mead style or simple coordinate descent
            # Use a simple adaptive step local search
            no_improve = 0
            max_no_improve = 30

            while check_time() and no_improve < max_no_improve:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    try_update(x, fx)
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

            # Coordinate-wise refinement
            if check_time() and not np.isnan(best_x).any():
                x = best_x.copy()
                fx = best_f
                step = 0.5
                for coord in range(dim):
                    if not check_time():
                        break
                    for _ in range(10):
                        if not check_time():
                            break
                        y = x.copy()
                        y[coord] += step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < fx:
                            x, fx = y, fy
                            try_update(x, fx)
                        else:
                            y = x.copy()
                            y[coord] -= step
                            y = np.clip(y, -5, 5)
                            fy = func(y)
                            calls_used += 1
                            if fy < fx:
                                x, fx = y, fy
                                try_update(x, fx)
                            else:
                                break
                    step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
