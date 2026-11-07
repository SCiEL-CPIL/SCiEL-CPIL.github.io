"""Rastrigin-aware optimizer: coarse lattice restarts + local (1+1)-ES refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Coarse periodic lattice search with local (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0          # safety margin before 300s kill
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # --- global state ---
        best_x = None
        best_f = np.inf
        calls = 0

        def call(x):
            nonlocal calls
            calls += 1
            return func(np.clip(x, -5.0, 5.0))

        # --- local refinement: (1+1)-ES starting from x0 ---
        def refine(x0, fx0, remaining):
            nonlocal best_x, best_f, calls
            if remaining <= 0:
                return
            x = x0.copy()
            fx = fx0
            sigma = 0.3
            # adapt sigma using 1/5 rule over short windows
            steps = 0
            successes = 0
            win = 0
            win_len = 10
            for _ in range(remaining):
                if calls >= budget:
                    break
                if time.time() > deadline - 1.0:
                    break
                y = x + sigma * rng.standard_normal(d)
                fy = call(y)
                steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                win += 1
                if win >= win_len:
                    if successes / win < 0.2:
                        sigma *= 0.8
                    elif successes / win > 0.8:
                        sigma *= 1.2
                    else:
                        sigma *= 1.0
                    win = 0
                    successes = 0
                # clamp sigma
                sigma = min(max(sigma, 1e-4), 5.0)

        # --- Phase 1: coarse periodic lattice random restarts ---
        # Rastrigin period is 2*pi ~ 6.283. Sample a coarse grid of candidate
        # minima across the box, evaluate, and keep the best few as seeds.
        # Use a stratified sampling of the period lattice.
        n_coarse = min(40, budget)
        # build candidate lattice points: base offset + k*period, clipped to box
        period = 2.0 * np.pi
        # sample random offsets in [-5, 5] and snap to period lattice near center
        coarse_points = []
        # deterministic stratified base grid: 3 points per dim along period
        base_vals = np.linspace(-5.0, 5.0, 3)
        # generate a set of candidate points by combining base offsets
        # Use a small full factorial over a subset of dims for diversity
        # Instead: random points but ensure coverage via period snapping
        for i in range(n_coarse):
            # random point, but snap each coord to nearest period multiple of a base
            p = rng.uniform(-5.0, 5.0, d)
            # snap to period lattice: p_snap = base + round((p-base)/period)*period
            base = -5.0
            p_snap = base + np.round((p - base) / period) * period
            p_snap = np.clip(p_snap, -5.0, 5.0)
            coarse_points.append(p_snap)

        # evaluate coarse points
        coarse_results = []
        for p in coarse_points:
            if calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            fp = call(p)
            coarse_results.append((fp, p))
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # sort by fitness, keep top seeds
        coarse_results.sort(key=lambda t: t[0])
        top_seeds = coarse_results[:min(5, len(coarse_results))]

        # --- Phase 2: local refinement from top seeds, split budget ---
        if best_x is None:
            # fallback: evaluate a single random point
            if calls < budget:
                p = rng.uniform(-5.0, 5.0, d)
                fp = call(p)
                best_x = p.copy()
                best_f = fp
            return best_x, best_f

        n_seeds = len(top_seeds)
        if n_seeds == 0:
            return best_x, best_f

        # allocate remaining budget among seeds, giving more to better seeds
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f

        # weights: inverse rank
        weights = np.array([1.0 / (i + 1.0) for i in range(n_seeds)])
        weights = weights / weights.sum()
        allocs = np.floor(weights * remaining).astype(int)
        # give leftover to best seed
        leftover = remaining - allocs.sum()
        allocs[0] += leftover

        for idx, (fp, p) in enumerate(top_seeds):
            alloc = int(allocs[idx])
            if alloc <= 0:
                continue
            if calls >= budget:
                break
            if time.time() > deadline - 1.0:
                break
            # refine from this seed
            refine(p.copy(), fp, alloc)

        # --- Phase 3: if budget remains, do a few more random restarts + refine ---
        if calls < budget and time.time() < deadline - 5.0:
            # a couple of fresh random restarts to escape local minima
            n_extra = min(3, (budget - calls) // 20)
            for _ in range(n_extra):
                if calls >= budget - 5:
                    break
                if time.time() > deadline - 2.0:
                    break
                p = rng.uniform(-5.0, 5.0, d)
                fp = call(p)
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
                # small refinement
                refine(p.copy(), fp, min(30, budget - calls))

        if best_x is None:
            best_x = np.zeros(d)
            best_f = 0.0
        return best_x, best_f
# EVOLVE-BLOCK-END
