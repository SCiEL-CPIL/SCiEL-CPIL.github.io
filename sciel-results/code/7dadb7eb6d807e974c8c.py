"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like adaptive search with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Phase 1: random scan to find a few good basins
        scan_size = min(50, budget // 10)
        best_x = None
        best_f = np.inf
        basin_starts = []

        for i in range(scan_size):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            basin_starts.append((x.copy(), fx))

        # Sort by fitness and take top few distinct basins
        basin_starts.sort(key=lambda t: t[1])
        # Deduplicate: keep starts that are at least 1.0 apart
        selected = []
        for x, fx in basin_starts:
            if all(np.linalg.norm(x - sx) > 1.0 for sx, _ in selected):
                selected.append((x, fx))
            if len(selected) >= 5:
                break

        # Phase 2: Run a simple adaptive ES per basin with shared budget
        # We'll use a population-based approach: maintain a population of candidates
        # and adapt step size based on success rate (1/5 rule generalized)

        # Total remaining budget
        remaining = budget - scan_size
        if remaining <= 0:
            return best_x, best_f

        # Allocate budget among basins: more to better basins
        n_basins = len(selected)
        if n_basins == 0:
            return best_x, best_f

        # Simple allocation: proportional to (1/rank)
        ranks = np.arange(1, n_basins + 1, dtype=float)
        weights = 1.0 / ranks
        weights /= weights.sum()
        allocs = np.floor(weights * remaining).astype(int)
        # Ensure at least 1 eval per basin
        for i in range(n_basins):
            if allocs[i] < 1:
                allocs[i] = 1
        # Adjust for rounding
        total_alloc = allocs.sum()
        diff = remaining - total_alloc
        if diff > 0:
            allocs[0] += diff
        elif diff < 0:
            # Take from the worst basin
            for i in range(n_basins - 1, -1, -1):
                if allocs[i] > 1:
                    take = min(allocs[i] - 1, -diff)
                    allocs[i] -= take
                    diff += take
                    if diff == 0:
                        break

        # For each basin, run a (lambda+mu)-ES with adaptive sigma
        lambda_pop = max(4, dim + 1)
        mu = lambda_pop // 2

        for bi, (x0, f0) in enumerate(selected):
            alloc = allocs[bi]
            if alloc <= 0:
                continue

            x = x0.copy()
            fx = f0
            sigma = 1.0
            successes = 0
            total_tries = 0

            for step in range(alloc):
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for k in range(lambda_pop):
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    offspring.append((y, fy))
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                # Sort offspring
                offspring.sort(key=lambda t: t[1])
                # Take top mu
                selected_off = offspring[:mu]
                new_x = np.mean([o[0] for o in selected_off], axis=0)
                new_fx = selected_off[0][1]

                # Adapt sigma based on success rate
                total_tries += lambda_pop
                successes += sum(1 for _, fy in offspring if fy < fx)
                rate = successes / total_tries
                if rate > 0.3:
                    sigma *= 1.1
                elif rate < 0.1:
                    sigma *= 0.9
                else:
                    sigma *= 1.0

                x = new_x
                fx = new_fx
                total_tries = 0
                successes = 0

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
