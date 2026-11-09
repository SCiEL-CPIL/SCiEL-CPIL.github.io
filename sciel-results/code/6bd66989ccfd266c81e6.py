import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280

        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            f_val = func(x)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)

        # Phase 1: Reduced initial sampling to find a good starting region quickly
        n_init = min(50, self.budget // 10)
        init_points = []
        init_vals = []

        for _ in range(n_init):
            if time.time() > deadline or self.calls >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = safe_func(x)
            init_points.append(x)
            init_vals.append(f)

        if not init_points:
            x = np.zeros(self.dim)
            safe_func(x)
            return self.best_x, self.best_f

        best_idx = np.argmin(init_vals)
        x0 = init_points[best_idx].copy()

        # Phase 2: Multi-scale coordinate descent with geometrically shrinking steps
        step_sizes = [10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125,
                      0.0390625, 0.01953125, 0.009765625, 0.0048828125,
                      0.00244140625, 0.001220703125, 0.0006103515625,
                      0.00030517578125, 0.000152587890625, 7.62939453125e-05,
                      3.814697265625e-05, 1.9073486328125e-05, 9.5367431640625e-06,
                      4.76837158203125e-06, 2.384185791015625e-06, 1.1920928955078125e-06,
                      5.9604644775390625e-07, 2.9802322387695312e-07, 1.4901161193847656e-07,
                      7.450580298461914e-08, 3.725290298461914e-08, 1.862645149230957e-08]

        dim_order = list(range(self.dim))

        current_x = x0.copy()
        current_f = self.best_f

        for scale_idx, step in enumerate(step_sizes):
            if self.calls >= self.budget or time.time() > deadline:
                break

            offset = scale_idx % self.dim
            rotated_dims = dim_order[offset:] + dim_order[:offset]

            for d in rotated_dims:
                if self.calls >= self.budget or time.time() > deadline:
                    break

                for direction in [1, -1]:
                    if self.calls >= self.budget or time.time() > deadline:
                        break

                    new_x = current_x.copy()
                    new_x[d] += direction * step
                    new_x = np.clip(new_x, -5, 5)

                    f_new = safe_func(new_x)

                    if f_new < current_f:
                        current_x = new_x
                        current_f = f_new

        # Phase 3: Random restarts with local refinement to escape local minima
        if self.calls < self.budget:
            remaining_budget = self.budget - self.calls
            n_restarts = min(8, remaining_budget // 30)

            for r in range(n_restarts):
                if self.calls >= self.budget or time.time() > deadline:
                    break

                x_r = rng.uniform(-5, 5, self.dim)
                f_r = safe_func(x_r)

                if f_r < self.best_f:
                    current_x = x_r.copy()
                    current_f = f_r
                else:
                    current_x = self.best_x.copy()
                    current_f = self.best_f

                local_steps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]
                for step in local_steps:
                    if self.calls >= self.budget or time.time() > deadline:
                        break

                    for d in range(self.dim):
                        if self.calls >= self.budget or time.time() > deadline:
                            break

                        for direction in [1, -1]:
                            if self.calls >= self.budget or time.time() > deadline:
                                break

                            new_x = current_x.copy()
                            new_x[d] += direction * step
                            new_x = np.clip(new_x, -5, 5)

                            f_new = safe_func(new_x)

                            if f_new < current_f:
                                current_x = new_x
                                current_f = f_new

        # Phase 4: Final fine-tuning on the best point if budget remains
        if self.calls < self.budget and self.best_x is not None:
            fine_steps = [0.001, 0.0005, 0.0002, 0.0001, 5e-05, 2e-05, 1e-05, 5e-06]
            current_x = self.best_x.copy()
            current_f = self.best_f

            for step in fine_steps:
                if self.calls >= self.budget or time.time() > deadline:
                    break

                for d in range(self.dim):
                    if self.calls >= self.budget or time.time() > deadline:
                        break

                    for direction in [1, -1]:
                        if self.calls >= self.budget or time.time() > deadline:
                            break

                        new_x = current_x.copy()
                        new_x[d] += direction * step
                        new_x = np.clip(new_x, -5, 5)

                        f_new = safe_func(new_x)

                        if f_new < current_f:
                            current_x = new_x
                            current_f = f_new

        return self.best_x, self.best_f
