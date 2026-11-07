"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining Pareto-front sampling with evolutionary refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _nondominated_mask(self, F):
        """Return boolean mask of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        le = (F[:, None, :] <= F[None, :, :])
        lt = (F[:, None, :] < F[None, :, :])
        D = le.all(axis=2) & lt.any(axis=2)
        dominated = D.any(axis=1)
        return ~dominated

    def _crowding_distance(self, F, nd_mask):
        """Compute crowding distance for non-dominated points."""
        n = len(F)
        cd = np.zeros(n)
        nd_indices = np.where(nd_mask)[0]
        m = len(nd_indices)
        if m <= 2:
            cd[nd_indices] = np.inf
            return cd
        for k in range(self.n_obj):
            sorted_idx = nd_indices[np.argsort(F[nd_indices, k])]
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            f_min = F[sorted_idx[0], k]
            f_max = F[sorted_idx[-1], k]
            if f_max - f_min < 1e-15:
                continue
            for j in range(1, m - 1):
                cd[sorted_idx[j]] += (F[sorted_idx[j + 1], k] - F[sorted_idx[j - 1], k]) / (f_max - f_min)
        return cd

    def _sample_near_pareto(self, problem, bounds, rng, n_samples):
        """
        Sample points biased toward the WFG4 Pareto front.
        For WFG4, the Pareto front has a specific structure where
        the first variable (x[0]) largely determines the position on the front.
        We sample x[0] uniformly and set other variables to values that
        tend to produce non-dominated solutions.
        """
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        
        results = []
        for _ in range(n_samples):
            # Sample first variable uniformly - this drives the Pareto front position
            x = np.empty(dim)
            x[0] = rng.uniform(lo[0], hi[0])
            
            # For remaining variables, use values that minimize their contribution
            # to objectives while allowing exploration
            for i in range(1, dim):
                # Mix of boundary values and small random perturbations
                if rng.random() < 0.5:
                    # Use lower bound (often optimal for WFG4 auxiliary variables)
                    x[i] = lo[i]
                else:
                    # Small perturbation around lower bound
                    x[i] = lo[i] + rng.random() * 0.1 * span[i]
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            results.append(x)
        
        return np.array(results)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial population with Pareto-front bias
        # Use a mix of random and Pareto-biased samples
        pop_size = min(40, budget // 2)
        
        # Half random, half Pareto-biased
        n_random = pop_size // 2
        n_pareto = pop_size - n_random
        
        X_random = lo + span * rng.random((n_random, dim))
        X_pareto = self._sample_near_pareto(problem, bounds, rng, n_pareto)
        
        X = np.vstack([X_random, X_pareto])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Phase 2: Evolutionary refinement with adaptive mutation
        step = 0.2

        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Find non-dominated set
            nd_mask = self._nondominated_mask(F)
            if not np.any(nd_mask):
                nd_mask = np.ones(len(F), dtype=bool)

            # Use crowding distance for parent selection among non-dominated
            cd = self._crowding_distance(F, nd_mask)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) > 0:
                # Select parent with highest crowding distance
                parent_idx = nd_indices[np.argmax(cd[nd_indices])]
            else:
                parent_idx = rng.integers(len(F))

            # Adaptive mutation: occasionally larger steps for exploration
            if rng.random() < 0.1:
                local_step = step * 2.0
            else:
                local_step = step

            child = X[parent_idx].copy()
            # Gaussian mutation
            child += local_step * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point
            if np.any(~nd_mask):
                dominated_indices = np.where(~nd_mask)[0]
                worst_in_dominated = dominated_indices[np.argmax(F[dominated_indices].sum(axis=1))]
                X[worst_in_dominated] = child
                F[worst_in_dominated] = fc
            else:
                worst_idx = nd_indices[np.argmin(cd[nd_indices])]
                X[worst_idx] = child
                F[worst_idx] = fc

            # Adaptive step adjustment
            if used % 20 == 0:
                if rng.random() < 0.5:
                    step *= 0.9
                else:
                    step *= 1.1
                step = np.clip(step, 0.05, 0.8)

        # Phase 3: Final polish - inject more Pareto-biased samples if budget allows
        if time.time() < deadline - 10.0:
            # Get non-dominated set for polishing
            nd_mask = self._nondominated_mask(F)
            nd_indices = np.where(nd_mask)[0]
            
            # Polish each non-dominated point with small perturbations
            for idx in nd_indices[:10]:
                if time.time() > deadline - 5.0:
                    break
                for _ in range(3):
                    if used >= budget or time.time() > deadline - 5.0:
                        break
                    polished = X[idx].copy()
                    polished += 0.05 * span * rng.standard_normal(dim)
                    polished = np.clip(polished, lo, hi)
                    fp = problem(polished)
                    used += 1
                    if np.all(fp <= F[idx]) and np.any(fp < F[idx]):
                        X[idx] = polished
                        F[idx] = fp
            
            # Also try injecting a few more Pareto-biased samples
            remaining = budget - used
            if remaining > 5 and time.time() < deadline - 5.0:
                n_extra = min(remaining, 10)
                X_extra = self._sample_near_pareto(problem, bounds, rng, n_extra)
                for x in X_extra:
                    if used >= budget or time.time() > deadline - 5.0:
                        break
                    fc = problem(x)
                    used += 1
                    # Add to population if it's non-dominated
                    nd_mask_new = self._nondominated_mask(np.vstack([F, fc.reshape(1, -1)]))
                    if nd_mask_new[-1]:
                        # Replace worst point
                        if np.any(~nd_mask):
                            dominated_indices = np.where(~nd_mask)[0]
                            worst = dominated_indices[np.argmax(F[dominated_indices].sum(axis=1))]
                        else:
                            cd_new = self._crowding_distance(F, nd_mask)
                            worst = np.where(nd_mask)[0][np.argmin(cd_new[nd_mask])]
                        X[worst] = x
                        F[worst] = fc

        return X
# EVOLVE-BLOCK-END
