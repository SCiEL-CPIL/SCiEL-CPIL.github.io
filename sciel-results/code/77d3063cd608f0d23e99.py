"""Combined optimizer for BBOB f24 (Lunacek bi-Rastrigin):
structured basin scan + (1+1)-ES with adaptive sigma, periodic re-exploration,
and stagnation restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse scan to locate basins, then (1+1)-ES with adaptive sigma and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0  # safety margin before the 300s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        calls = 0
        best_x = None
        best_f = np.inf

        def record(x, f):
            nonlocal calls, best_x, best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def eval_pt(x):
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # ---- Phase 1: coarse structured scan to find promising basins ----
        # Lunacek f24 has two basins; structured points help locate both.
        candidates = [np.zeros(n)]
        # Axis-aligned offsets
        for i in range(n):
            for s in (-3.0, -1.5, 1.5, 3.0):
                p = np.zeros(n)
                p[i] = s
                candidates.append(p)
        # Diagonal points
        for s in (-2.0, -1.0, 1.0, 2.0):
            candidates.append(np.full(n, s))
        # Random points in the box
        for _ in range(40):
            candidates.append(rng.uniform(-4.5, 4.5, n))

        for x in candidates:
            if calls >= budget or time.time() > deadline:
                break
            record(x, eval_pt(x))

        if best_x is None:
            best_x = np.zeros(n)
            best_f = eval_pt(best_x)
            record(best_x, best_f)

        if time.time() > deadline:
            return best_x, best_f

        # ---- Phase 2: Multi-start population-based (1+1)-ES ----
        # Maintain a small population of independent lineages, each with its own
        # position and sigma. This structurally addresses the two-basin structure
        # of Lunacek f24 by giving multiple lineages independent chances.
        
        pop_size = 4
        # Initialize population: one from best found, rest randomly spread
        population = []
        population.append({'x': best_x.copy(), 'f': best_f, 'sigma': 1.5, 'no_improve': 0})
        
        for i in range(pop_size - 1):
            if calls >= budget or time.time() > deadline:
                break
            px = rng.uniform(-4.5, 4.5, n)
            pf = eval_pt(px)
            record(px, pf)
            population.append({'x': px, 'f': pf, 'sigma': 1.5, 'no_improve': 0})
        
        # Round-robin over population lineages
        round_idx = 0
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Select which lineage to advance this round
            lineage = population[round_idx % len(population)]
            round_idx += 1
            
            # Generate candidate for this lineage
            y = np.clip(lineage['x'] + lineage['sigma'] * rng.standard_normal(n), -5.0, 5.0)
            fy = eval_pt(y)
            record(y, fy)
            
            if fy < lineage['f']:
                lineage['x'] = y
                lineage['f'] = fy
                lineage['sigma'] = min(lineage['sigma'] * 1.2, 5.0)
                lineage['no_improve'] = 0
            else:
                lineage['sigma'] = max(lineage['sigma'] * 0.8, 0.01)
                lineage['no_improve'] += 1
            
            # If a lineage is very stuck, reinitialize it from best global
            if lineage['no_improve'] > 40:
                if calls >= budget or time.time() > deadline:
                    break
                lineage['x'] = best_x.copy() + 0.5 * rng.standard_normal(n)
                lineage['x'] = np.clip(lineage['x'], -5.0, 5.0)
                lineage['f'] = eval_pt(lineage['x'])
                record(lineage['x'], lineage['f'])
                lineage['sigma'] = 1.0
                lineage['no_improve'] = 0
            
            # Occasionally inject a fresh random point into a random lineage
            if round_idx % 50 == 0 and calls < budget:
                if time.time() > deadline:
                    break
                target = population[rng.integers(len(population))]
                xr = rng.uniform(-4.5, 4.5, n)
                fr = eval_pt(xr)
                record(xr, fr)
                if fr < target['f']:
                    target['x'] = xr
                    target['f'] = fr
                    target['sigma'] = 1.5
                    target['no_improve'] = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
