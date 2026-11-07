"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided adaptive differential evolution for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv2d(self, F, ref):
        """Exact hypervolume for 2D points (only those strictly better than ref)."""
        pts = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
        if len(pts) == 0:
            return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        hv = 0.0
        prev_y = ref[1]
        for i in range(len(pts)):
            x, y = pts[i]
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return hv

    def _non_dominated(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)
        ref = np.array([3000.0, 0.06])

        # --- Phase 1: Deterministic grid + random initialization ---
        # Bias toward low values in each dimension (since minimizing both objectives)
        # Use a stratified grid to cover the space efficiently
        n_init = min(60, budget // 3)
        X = np.zeros((n_init, dim))
        F = np.zeros((n_init, 2))

        # Stratified initialization: divide each dim into bins
        n_per_dim = max(2, int(np.ceil(n_init ** (1.0 / dim))))
        grid_idx = np.array(np.unravel_index(np.arange(n_init), [n_per_dim] * dim)).T
        # Clip to valid range
        grid_idx = np.clip(grid_idx, 0, n_per_dim - 1)
        for i in range(n_init):
            for d in range(dim):
                frac = (grid_idx[i, d] + rng.random()) / n_per_dim
                X[i, d] = lo[d] + frac * span[d]
            F[i] = problem(X[i])
        used = n_init

        # Add some pure random points for diversity
        n_rand = min(20, budget - n_init)
        if n_rand > 0:
            Xr = lo + span * rng.random((n_rand, dim))
            Fr = np.array([problem(x) for x in Xr])
            X = np.vstack([X, Xr])
            F = np.vstack([F, Fr])
            used += n_rand

        # --- Phase 2: Hypervolume-guided adaptive DE ---
        pop_size = min(40, budget - used)
        if pop_size < 4:
            pop_size = max(4, min(40, budget - used))

        # Build population from best hypervolume contributors
        # Sort by contribution: non-dominated first, then by HV contribution
        nd_mask = self._non_dominated(F)
        if np.sum(nd_mask) >= pop_size:
            pop_idx = np.where(nd_mask)[0][:pop_size]
        else:
            # Take all non-dominated + best dominated
            nd_idx = np.where(nd_mask)[0]
            dom_idx = np.where(~nd_mask)[0]
            # Sort dominated by sum of objectives (lower is better)
            dom_scores = np.sum(F[dom_idx], axis=1)
            dom_sorted = dom_idx[np.argsort(dom_scores)]
            pop_idx = np.concatenate([nd_idx, dom_sorted[:max(0, pop_size - len(nd_idx))]])
            if len(pop_idx) < pop_size:
                pop_idx = np.concatenate([pop_idx, rng.integers(0, len(F), size=pop_size - len(pop_idx))])
            pop_idx = pop_idx[:pop_size]

        Pop = X[pop_idx].copy()
        PopF = F[pop_idx].copy()

        # Track all evaluated points for final hypervolume
        all_F = [F.copy()]

        gen = 0
        while used < budget:
            if time.time() > deadline:
                break

            # Adaptive mutation factor: start large, shrink over generations
            progress = min(1.0, gen / max(1, (budget - used) // pop_size))
            F_mut = 0.8 * (1.0 - 0.5 * progress) + rng.random() * 0.2  # 0.6-0.8 early, 0.4-0.6 late

            # Select parents: hypervolume-guided tournament
            # Compute HV contribution for each point
            nd_pop = self._non_dominated(PopF)
            # For selection, prefer non-dominated, then lower objective sum
            weights = np.zeros(len(PopF))
            weights[nd_pop] = 1.0
            weights[~nd_pop] = 0.1 / (1.0 + np.sum(PopF[~nd_pop], axis=1) / 100.0)
            weights /= (np.sum(weights) + 1e-10)

            # Differential evolution: pick 3 distinct parents
            while used < budget and time.time() < deadline:
                # Tournament selection with hypervolume bias
                a = rng.choice(len(PopF), p=weights)
                b = rng.choice(len(PopF), p=weights)
                c = rng.choice(len(PopF), p=weights)
                while b == a:
                    b = rng.choice(len(PopF), p=weights)
                while c == a or c == b:
                    c = rng.choice(len(PopF), p=weights)

                # Mutation
                v = Pop[a] + F_mut * (Pop[b] - Pop[c])
                # Crossover (binomial)
                cross_rate = 0.5
                mask = rng.random(dim) < cross_rate
                if not np.any(mask):
                    mask[rng.integers(dim)] = True
                trial = np.where(mask, v, Pop[a])
                trial = np.clip(trial, lo, hi)

                # Evaluate
                trial_F = problem(trial)
                used += 1

                # Replace if trial is non-dominated by current pop or improves HV
                # Quick check: does trial dominate any pop member?
                dominates_any = False
                for i in range(len(PopF)):
                    if np.all(trial_F <= PopF[i]) and np.any(trial_F < PopF[i]):
                        dominates_any = True
                        break

                # Is trial dominated by any pop member?
                dominated_by = False
                for i in range(len(PopF)):
                    if np.all(PopF[i] <= trial_F) and np.any(PopF[i] < trial_F):
                        dominated_by = True
                        break

                if dominates_any or not dominated_by:
                    # Find worst member to replace (highest objective sum among dominated)
                    if dominated_by:
                        # Only replace if trial has better HV contribution
                        # Simple heuristic: replace worst dominated member
                        dom_in_pop = ~self._non_dominated(PopF)
                        if np.any(dom_in_pop):
                            worst = np.where(dom_in_pop)[0][np.argmax(np.sum(PopF[dom_in_pop], axis=1))]
                            Pop[worst] = trial
                            PopF[worst] = trial_F
                    else:
                        # Replace worst member
                        worst = int(np.argmax(np.sum(PopF, axis=1)))
                        Pop[worst] = trial
                        PopF[worst] = trial_F

                all_F.append(trial_F.reshape(1, -1))

            gen += 1

        # --- Final: collect all evaluated points and return ---
        all_F_arr = np.vstack(all_F)
        return Pop
# EVOLVE-BLOCK-END
