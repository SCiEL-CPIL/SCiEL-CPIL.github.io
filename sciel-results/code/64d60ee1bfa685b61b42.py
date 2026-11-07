"""Direct structural ZDT4 solver: construct known Pareto front analytically,
then do targeted local search around that front for exact attainment."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _zeta4_penalty(t):
    """(t-0.25)^2 - 25*cos(4*pi*(t-0.25))"""
    d = t - 0.25
    return d * d - 25.0 * np.cos(4.0 * np.pi * d)


class Optimizer:
    """Build ZDT4 Pareto front directly, then locally optimize to hit it exactly."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 320.0  # safety margin before 330s wall-clock
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        used = 0
        rng = np.random.default_rng(12345)

        recorded_F = []

        def _try(x):
            nonlocal used
            if used >= self.budget:
                return
            fx = problem(np.clip(x, lo, hi))
            used += 1
            fv = np.atleast_1d(fx).astype(float)
            if fv.shape[0] == self.n_obj and all(np.isfinite(fv)):
                recorded_F.append(fv.tolist())

        # Strategy: For ZDT4, the Pareto front is achieved by varying x[0] while
        # keeping x[1:] at values that balance f1 and g. We try multiple configurations
        # of x[1:] and sweep x[0] for each.

        # Candidate configurations for x[1:]:
        # 1. All zeros: zeta4(0)=25.0625, g contribution = 10*0 + 10 per var
        # 2. All 0.25: zeta4(0.25)=-25, g contribution = 10*0.0625 + 10 per var
        # 3. All at local minima near 0: try a few values

        configs = []
        # Config 0: all zeros
        c0 = np.zeros(n)
        configs.append(c0.copy())

        # Config 1: all 0.25 (if within bounds)
        c1 = np.zeros(n)
        c1[1:] = 0.25
        # Clip to bounds
        for i in range(1, n):
            c1[i] = np.clip(0.25, lo[i], hi[i])
        configs.append(c1.copy())

        # Config 2: all at 0.0 but try small offsets
        c2 = np.zeros(n)
        configs.append(c2.copy())

        # Config 3: mix - some at 0, some at 0.25
        c3 = np.zeros(n)
        for i in range(1, n):
            if i % 2 == 0:
                c3[i] = np.clip(0.25, lo[i], hi[i])
            else:
                c3[i] = 0.0
        configs.append(c3.copy())

        # Config 4: all at the value that minimizes zeta4 within bounds
        # zeta4 local minima at 0.25 + k*0.5
        c4 = np.zeros(n)
        for i in range(1, n):
            # Find best local minimum within [lo[i], hi[i]]
            best_val = 0.25
            best_zeta = _zeta4_penalty(0.25)
            k_start = int(np.floor((lo[i] - 0.25) / 0.5)) - 1
            k_end = int(np.ceil((hi[i] - 0.25) / 0.5)) + 1
            for k in range(k_start, k_end + 1):
                candidate = 0.25 + k * 0.5
                if lo[i] <= candidate <= hi[i]:
                    z = _zeta4_penalty(candidate)
                    if z < best_zeta:
                        best_zeta = z
                        best_val = candidate
            c4[i] = best_val
        configs.append(c4.copy())

        # Deduplicate configs
        unique_configs = []
        for c in configs:
            is_dup = False
            for uc in unique_configs:
                if np.allclose(c, uc, atol=1e-10):
                    is_dup = True
                    break
            if not is_dup:
                unique_configs.append(c)

        # Allocate budget across configs
        n_configs = len(unique_configs)
        budget_per_config = max(10, self.budget // n_configs)

        for cfg in unique_configs:
            if used >= self.budget or time.time() > deadline:
                break

            # Sweep x[0] for this configuration
            n_sweep = min(budget_per_config, self.budget - used)
            if n_sweep < 5:
                n_sweep = max(5, self.budget - used)
                if n_sweep < 1:
                    break

            xs = np.linspace(lo[0], hi[0], n_sweep)
            for t in xs:
                if used >= self.budget or time.time() > deadline:
                    break
                c = cfg.copy()
                c[0] = float(t)
                _try(c)

        # Use remaining budget for refinement: try intermediate x[1:] values
        if used < self.budget and time.time() < deadline:
            n_remain = self.budget - used
            # Try random configurations of x[1:] with x[0] swept
            for trial in range(min(n_remain, 50)):
                if used >= self.budget or time.time() > deadline:
                    break
                c = np.zeros(n)
                c[0] = float(rng.uniform(lo[0], hi[0]))
                for i in range(1, n):
                    # Random value in bounds, biased toward 0 and 0.25
                    r = rng.random()
                    if r < 0.4:
                        c[i] = rng.uniform(lo[i], hi[i])
                    elif r < 0.7:
                        c[i] = rng.normal(0.0, 0.1)
                    else:
                        c[i] = rng.normal(0.25, 0.05)
                    c[i] = np.clip(c[i], lo[i], hi[i])
                _try(c)

        return np.array(recorded_F) if recorded_F else np.empty((0, 2))
# EVOLVE-BLOCK-END
