"""WFG4 multi-objective optimizer exploiting variable separability."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid structural optimizer for WFG4: decouple front variables from penalty variables."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        all_F = []
        eval_count = 0

        def eval_point(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            all_F.append(f)
            return f

        def time_ok():
            return time.time() < deadline

        # Phase 1: Find good penalty-minimizing configurations (9 vars) for given front vars (3 vars)
        # WFG4 structure: x[0:3] -> front, x[3:12] -> penalty
        # The penalty is minimized when shape variables are at specific values.

        # First, do a broad random search to understand the landscape
        n_init = min(30, self.budget // 5)
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.array([eval_point(x) for x in X_init])

        if not time_ok() or eval_count >= self.budget:
            return np.array(all_F)

        # Phase 2: For several front configurations, optimize penalty variables
        # Sample diverse front variable configurations
        n_front_samples = min(10, max(5, self.budget // 20))
        front_configs = []
        for i in range(n_front_samples):
            # Random front variables in [0, 1] (typical WFG4 range)
            if i == 0:
                front = np.array([0.0, 0.0, 0.0])
            elif i == 1:
                front = np.array([1.0, 1.0, 1.0])
            elif i == 2:
                front = np.array([0.5, 0.5, 0.5])
            else:
                front = rng.random(3)
            front_configs.append(front)

        # For each front config, optimize the 9 penalty variables
        best_penalty_configs = []
        for front in front_configs:
            if not time_ok() or eval_count >= self.budget - 5:
                break

            # Start from best-known penalty configuration or random
            best_x = np.zeros(self.dim)
            best_x[0:3] = front
            # Initialize penalty vars from a random point or best found
            if len(all_F) > 0:
                # Use the point with lowest total objective as starting penalty
                F_arr = np.array(all_F)
                best_idx = np.argmin(np.sum(F_arr, axis=1))
                best_x[3:12] = X_init[best_idx % len(X_init)][3:12]
            else:
                best_x[3:12] = lo[3:12] + (hi[3:12] - lo[3:12]) * rng.random(9)

            f = eval_point(best_x)
            if f is None:
                break
            best_f = f.copy()

            # Coordinate descent on penalty variables
            step = (hi[3:12] - lo[3:12]) * 0.2
            for iteration in range(15):
                if not time_ok() or eval_count >= self.budget - 3:
                    break
                improved = False
                for j in range(9):
                    if not time_ok() or eval_count >= self.budget - 2:
                        break
                    # Try positive and negative steps
                    for direction in [1.0, -1.0]:
                        trial_x = best_x.copy()
                        trial_x[3 + j] = np.clip(best_x[3 + j] + direction * step[j], lo[3 + j], hi[3 + j])
                        f_trial = eval_point(trial_x)
                        if f_trial is None:
                            break
                        # Check if improved (any objective better, none worse)
                        if np.all(f_trial <= best_f) and np.any(f_trial < best_f):
                            best_x = trial_x
                            best_f = f_trial.copy()
                            improved = True
                            break
                # Reduce step size if no improvement
                if not improved:
                    step *= 0.5
                    if np.max(step) < 1e-6:
                        break

            best_penalty_configs.append((best_x, best_f.copy()))

        if not time_ok() or eval_count >= self.budget:
            return np.array(all_F)

        # Phase 3: Fine-tune front variables using the best penalty configurations
        # Use the best penalty config as base and vary front variables
        if best_penalty_configs:
            # Use the configuration with lowest total objective
            best_base = min(best_penalty_configs, key=lambda t: np.sum(t[1]))
            base_x = best_base[0].copy()

            # Grid search over front variables
            n_grid = min(5, max(3, int((self.budget - eval_count) ** 0.5)))
            grid_vals = np.linspace(0.0, 1.0, n_grid)

            for g0 in grid_vals:
                if not time_ok() or eval_count >= self.budget:
                    break
                for g1 in grid_vals:
                    if not time_ok() or eval_count >= self.budget:
                        break
                    for g2 in grid_vals:
                        if not time_ok() or eval_count >= self.budget:
                            break
                        trial_x = base_x.copy()
                        trial_x[0] = g0
                        trial_x[1] = g1
                        trial_x[2] = g2
                        f = eval_point(trial_x)
                        if f is None:
                            break

        # Phase 4: Final local refinement on non-dominated points
        if time_ok() and eval_count < self.budget - 5:
            F_arr = np.array(all_F)
            # Find non-dominated points
            n_dom = np.zeros(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_dom[i] = True
                        break
            non_dom_idx = np.where(~n_dom)[0]

            if len(non_dom_idx) > 0:
                # Local search around each non-dominated point
                n_local = min(len(non_dom_idx), max(3, (self.budget - eval_count) // 5))
                for idx in non_dom_idx[:n_local]:
                    if not time_ok() or eval_count >= self.budget - 2:
                        break
                    center = X_init[idx % len(X_init)] if idx < len(X_init) else np.zeros(self.dim)
                    # Small perturbations
                    for _ in range(3):
                        if not time_ok() or eval_count >= self.budget:
                            break
                        perturbation = 0.05 * (hi - lo) * rng.standard_normal(self.dim)
                        trial_x = np.clip(center + perturbation, lo, hi)
                        eval_point(trial_x)

        return np.array(all_F)
# EVOLVE-BLOCK-END
