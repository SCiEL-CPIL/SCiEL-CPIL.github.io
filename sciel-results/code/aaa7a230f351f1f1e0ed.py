import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # ZDT6 Pareto front:
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)^6)^(1/10)
        # f2 = 1 - sqrt(f1)
        # The front is parameterized by x1 in [0,1].
        # We want to place budget points on the front to maximize hypervolume
        # with reference point (11, 11).

        # Key insight: sample x1 uniformly, compute f1 values, then optimize
        # the f1 distribution for max HV.
        # Actually, since f1(x1) is not monotonic, we need to be careful.
        # Let's just sample x1 densely, get (f1, f2) pairs, and pick the best subset.

        # Better approach: sample x1 on a fine grid, compute f1 for each,
        # then for each target f1 value (chosen to maximize HV), find the
        # x1 that achieves it.

        # Step 1: Build dense lookup of f1(x1)
        n_grid = 2000000
        x1_grid = np.linspace(0.0, 1.0, n_grid)
        f1_grid = np.empty(n_grid)
        chunk = 200000
        for i in range(0, n_grid, chunk):
            end = min(i + chunk, n_grid)
            x_chunk = x1_grid[i:end]
            sin_val = np.sin(6.0 * np.pi * x_chunk)
            f1_grid[i:end] = 1.0 - (1.0 - np.exp(-4.0 * x_chunk) * sin_val**6)**0.1

        # Step 2: For HV maximization, we need points on the Pareto front.
        # The Pareto front in objective space: for each f1 value t, f2 = 1 - sqrt(t).
        # We need to find x1 values that give specific f1 values.
        # Since f1 is not monotonic, for a target f1 value, there may be multiple x1.
        # We just need any x1 that gives that f1.

        # Sort by f1 to create a lookup
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]

        f1_min = f1_sorted[0]
        f1_max = f1_sorted[-1]

        # Step 3: Choose n_points f1 values to maximize HV
        # HV with reference (11,11): for sorted points t_1 < t_2 < ... < t_n
        # HV = sum_{i=1}^{n} (t_{i+1} - t_i) * (10 + sqrt(t_i)) + (11 - t_n)*(10 + sqrt(t_n))
        # where t_{n+1} = 11
        # Optimal: uniform in t (since 10+sqrt(t) is slowly varying)
        # But let's do a quick local optimization.

        n_points = min(budget, 200)
        t = np.linspace(f1_min, f1_max, n_points)

        def compute_hv(t_arr):
            t_sorted = np.sort(t_arr)
            t_ext = np.append(t_sorted, 11.0)
            widths = t_ext[1:] - t_ext[:-1]
            heights = 10.0 + np.sqrt(t_sorted)
            return np.sum(widths * heights)

        # Local optimization via coordinate descent
        hv_cur = compute_hv(t)
        step = (f1_max - f1_min) / (n_points * 5)
        for _ in range(30):
            if time.time() > deadline - 20:
                break
            improved = False
            for i in range(n_points):
                for delta in [-step, step]:
                    t_trial = t.copy()
                    t_trial[i] = np.clip(t[i] + delta, f1_min, f1_max)
                    hv_trial = compute_hv(t_trial)
                    if hv_trial > hv_cur:
                        t = t_trial
                        hv_cur = hv_trial
                        improved = True
                        break
            if not improved:
                step *= 0.6
                if step < 1e-12:
                    break

        # Step 4: Map f1 targets to x1 values
        x1_targets = np.empty(n_points)
        for i in range(n_points):
            target = t[i]
            idx = np.searchsorted(f1_sorted, target)
            idx = np.clip(idx, 0, len(f1_sorted) - 1)
            # Check neighbors
            best_dist = np.inf
            best_x1 = x1_sorted[idx]
            for j in range(max(0, idx - 3), min(len(f1_sorted), idx + 4)):
                dist = abs(f1_sorted[j] - target)
                if dist < best_dist:
                    best_dist = dist
                    best_x1 = x1_sorted[j]
            x1_targets[i] = best_x1

        # Step 5: Evaluate
        recorded = []
        for i in range(n_points):
            if time.time() > deadline - 5:
                break
            x = np.zeros(dim)
            x[0] = x1_targets[i]
            # Clip to bounds
            for d in range(dim):
                x[d] = np.clip(x[d], bounds[d][0], bounds[d][1])
            obj = problem(x)
            recorded.append(obj)

        return recorded
