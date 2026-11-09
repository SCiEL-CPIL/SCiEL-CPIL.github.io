"""Structured WFG9 multi-objective optimizer: deterministic sweep + adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured search exploiting WFG9 structure: sweep primary var, refine rest."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim

        # Base point: midpoint is a reasonable "neutral" for the penalty variables.
        base = (lo + hi) / 2.0

        all_F = []  # list of objective vectors (each length n_obj)

        def record(x):
            fx = problem(x)
            fx = np.asarray(fx, dtype=float)
            all_F.append(fx)
            return fx

        # ---- Phase 1: deterministic sweep along x[0] (primary trade-off var) ----
        # Use a moderate number of sweep points to get good spread.
        n_sweep = 40
        t_vals = np.linspace(0.0, 1.0, n_sweep)
        for t in t_vals:
            x = base.copy()
            x[0] = lo[0] + t * span[0]
            record(x)

        # ---- Phase 2: adaptive refinement ----
        # Keep a set of candidate base points (good x[0] values) and perturb other vars.
        # Strategy: for each sweep point, do a few Gaussian perturbations on vars 1..dim-1
        # to try to reduce the shared penalty / improve convergence.
        remaining = self.budget - len(all_F)
        # Distribute remaining budget among refinement steps.
        # We'll do rounds: each round picks a random existing point, perturbs, evaluates.
        # To keep spread, we bias selection toward points that are currently non-dominated-ish.

        # Maintain arrays for fast selection.
        F_arr = np.array(all_F) if all_F else np.empty((0, self.n_obj))
        X_arr = np.array([base.copy()] * 0)  # placeholder, we track separately

        # Re-track X alongside F for refinement.
        # Rebuild: we know the sweep points.
        sweep_X = []
        for t in t_vals:
            x = base.copy()
            x[0] = lo[0] + t * span[0]
            sweep_X.append(x)
        sweep_X = np.array(sweep_X)

        # Combine: full population arrays
        X_pop = list(sweep_X)
        F_pop = list(F_arr)

        def _non_dominated_mask(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd

        rng = np.random.default_rng(0)

        # Adaptive sigma that shrinks over time.
        sigma0 = 0.15
        while len(F_pop) < self.budget:
            if time.time() > deadline - 5.0:
                break
            # Pick a parent: prefer non-dominated, with some randomness.
            nd = _non_dominated_mask(np.array(F_pop))
            if np.any(nd):
                idx = rng.choice(np.where(nd)[0])
            else:
                idx = rng.integers(len(F_pop))
            parent = X_pop[idx].copy()

            # Perturb: primarily the penalty variables (1..dim-1), small change to x[0]
            sigma = sigma0 * span
            delta = rng.standard_normal(dim) * sigma
            # Reduce perturbation on x[0] to keep the trade-off structure
            delta[0] *= 0.3
            child = parent + delta
            child = np.clip(child, lo, hi)

            fc = record(child)
            F_pop.append(fc)
            X_pop.append(child)

            # Replace worst point to keep population bounded and diverse.
            # Use a simple crowding-like: remove point with largest sum (coarse).
            Fmat = np.array(F_pop)
            sums = np.sum(Fmat, axis=1)
            worst = int(np.argmax(sums))
            # Only replace if child is not strictly worse than worst in all objectives
            if np.any(Fmat[worst] <= fc) and np.any(Fmat[worst] >= fc):
                F_pop[worst] = fc
                X_pop[worst] = child

            # Periodically re-compact: drop clearly dominated points to keep front tight
            if len(F_pop) > 60 and len(F_pop) % 20 == 0:
                nd = _non_dominated_mask(np.array(F_pop))
                if np.sum(nd) >= 10:
                    keep = np.where(nd)[0]
                    F_pop = [F_pop[i] for i in keep]
                    X_pop = [X_pop[i] for i in keep]

        # Ensure we don't exceed budget (record() already counts).
        return X_pop
# EVOLVE-BLOCK-END
