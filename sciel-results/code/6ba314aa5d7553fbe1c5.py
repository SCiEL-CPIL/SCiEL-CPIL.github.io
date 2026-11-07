"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).

Uses a pattern search (direct search) approach with orthogonal and diagonal
directions, combined with adaptive step-size control and sparse random restarts.
This is a structural change from (1+1)-ES, replacing trajectory-based evolution
with deterministic local refinement that is robust to ill-conditioning.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Pattern search optimizer for ill-conditioned BBOB-style problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _pattern_search(self, func, x0, f0, budget, deadline):
        """
        Pattern search with orthogonal and diagonal directions.
        Uses binary search on step size for each direction.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        f = f0
        calls = 0
        best_x = x.copy()
        best_f = f

        # Build direction set: unit vectors and normalized diagonals
        directions = []
        # Orthogonal directions (±e_i)
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)
        # Diagonal directions (±(1,...,1)/sqrt(dim))
        diag = np.ones(dim) / np.sqrt(dim)
        directions.append(diag)
        directions.append(-diag)
        # Additional diagonal patterns for better coverage
        for i in range(dim):
            for j in range(i + 1, dim):
                d = np.zeros(dim)
                d[i] = 1.0
                d[j] = 1.0
                d /= np.sqrt(2)
                directions.append(d)
                directions.append(-d)

        n_dirs = len(directions)

        # Adaptive step size
        step = 1.0
        min_step = 1e-8
        max_step = 5.0

        # Track progress
        no_improve_count = 0
        max_no_improve = 30

        while calls < budget and time.time() < deadline:
            improved = False
            for d in directions:
                if calls >= budget or time.time() > deadline:
                    break

                # Binary search on step size in direction d
                # Try a few candidate step sizes
                found = False
                for trial in range(5):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Adaptive step: shrink if no improvement, grow if improving
                    s = step * (0.5 ** trial)
                    if s < min_step:
                        break

                    y = x + s * d
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls += 1

                    if fy < f:
                        x = y
                        f = fy
                        improved = True
                        found = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        break
                    else:
                        # If moving in this direction worsens, try smaller step
                        continue

                if found:
                    # Step worked; keep step size or slightly increase
                    step = min(step * 1.1, max_step)
                else:
                    # No improvement in this direction; shrink step
                    step = max(step * 0.7, min_step)

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If no improvement in a full cycle, shrink step more aggressively
                step = max(step * 0.5, min_step)

            if no_improve_count >= max_no_improve:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Phase 1: Sparse random sampling to find a good starting point
        n_samples = min(15, budget // 10)
        samples = rng.uniform(-4.0, 4.0, size=(n_samples, dim))
        sample_values = []
        for i in range(n_samples):
            if total_calls >= budget or time.time() > deadline - 5.0:
                break
            x = samples[i]
            fx = func(x)
            total_calls += 1
            sample_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Pattern search from the best sample
        if len(sample_values) > 0:
            sorted_idx = np.argsort(sample_values)
            n_starts = min(3, len(sorted_idx))

            remaining_budget = budget - total_calls
            budget_per_start = max(20, remaining_budget // (n_starts + 1))

            for start_i in range(n_starts):
                if total_calls >= budget or time.time() > deadline - 5.0:
                    break

                x0 = samples[sorted_idx[start_i]].copy()
                f0 = sample_values[sorted_idx[start_i]]

                x_best, f_best, calls_used = self._pattern_search(
                    func, x0, f0, budget_per_start, deadline
                )
                total_calls += calls_used

                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()

        # Phase 3: Final local polish from the best point with smaller steps
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline - 5.0:
            x_local = best_x.copy()
            f_local = best_f

            # Use a fine-grained pattern search with small initial step
            dim = self.dim
            directions = []
            for i in range(dim):
                e = np.zeros(dim)
                e[i] = 1.0
                directions.append(e)
                directions.append(-e)

            step = 0.1
            min_step = 1e-7
            max_no_improve = 20
            no_improve_count = 0

            while remaining > 0 and time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
                improved = False
                for d in directions:
                    if remaining <= 0 or time.time() > deadline - 5.0:
                        break

                    y = x_local + step * d
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    remaining -= 1
                    total_calls += 1

                    if fy < f_local:
                        x_local = y
                        f_local = fy
                        improved = True
                        step = min(step * 1.2, 1.0)
                        if f_local < best_f:
                            best_f = f_local
                            best_x = x_local.copy()
                    else:
                        step = max(step * 0.8, min_step)

                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    step = max(step * 0.5, min_step)

        return best_x, best_f
# EVOLVE-BLOCK-END
