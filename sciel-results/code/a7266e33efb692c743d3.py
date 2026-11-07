"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Interleaved multi-start (1+1)-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        # Phase 1: Small multi-point exploration to seed diverse starts
        n_starts = 4
        n_explore = max(8, min(15, budget // 10))
        seeds = []
        for i in range(n_explore):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            # Pick top n_starts diverse points as seeds
            if len(seeds) < n_starts:
                seeds.append((x.copy(), fx))
            else:
                # Replace worst seed if this is better
                worst_idx = max(range(len(seeds)), key=lambda j: seeds[j][1])
                if fx < seeds[worst_idx][1]:
                    seeds[worst_idx] = (x.copy(), fx)

        # Ensure we have n_starts seeds
        while len(seeds) < n_starts and calls_made < budget:
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            seeds.append((x.copy(), fx))

        # Initialize interleaved ES instances
        instances = []
        for i, (sx, sf) in enumerate(seeds):
            instances.append({
                'x': sx.copy(),
                'fx': sf,
                'sigma': 1.5,
                'stall': 0,
                'restarts': 0
            })

        # Phase 2: Interleaved (1+1)-ES with adaptive restarts
        idx = 0
        while calls_made < budget:
            if not time_ok():
                break

            inst = instances[idx % len(instances)]
            idx += 1

            # Adaptive restart if stuck
            if inst['stall'] >= 20 and inst['restarts'] < 8:
                inst['sigma'] = max(inst['sigma'] * 2.0, 0.5)
                inst['x'] = best_x.copy() if best_x is not None else rng.uniform(-5, 5, dim)
                inst['fx'] = best_f if best_x is not None else np.inf
                inst['stall'] = 0
                inst['restarts'] += 1
                # Small perturbation
                inst['x'] = np.clip(inst['x'] + 0.1 * rng.standard_normal(dim), -5, 5)
                if inst['fx'] == np.inf:
                    inst['fx'] = func(inst['x'])
                    calls_made += 1
                    try_update(inst['x'], inst['fx'])
                continue

            # Standard (1+1)-ES step
            y = np.clip(inst['x'] + inst['sigma'] * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            try_update(y, fy)

            if fy < inst['fx']:
                inst['x'] = y
                inst['fx'] = fy
                inst['sigma'] = min(inst['sigma'] * 1.15, 10.0)
                inst['stall'] = 0
            else:
                inst['sigma'] = max(inst['sigma'] * 0.9, 1e-6)
                inst['stall'] += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
