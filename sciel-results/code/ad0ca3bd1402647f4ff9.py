import numpy as np


class Optimizer:
    """Lattice-based random walk exploiting Rastrigin's periodic structure."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Discrete step sizes, commensurate with Rastrigin period (2.0)
        step_sizes = np.array([
            2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125,
            0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
            0.001953125, 0.0009765625, 0.00048828125, 0.000244140625,
            0.0001220703125, 0.00006103515625, 0.000030517578125
        ])
        n_steps = len(step_sizes)

        # Build direction set: axis-aligned and some diagonals
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
            directions.append(-d)
        # Add some diagonal directions
        for i in range(dim):
            for j in range(i + 1, dim):
                d = np.zeros(dim)
                d[i] = 1.0
                d[j] = 1.0
                directions.append(d / np.sqrt(2.0))
                directions.append(-d / np.sqrt(2.0))
        directions = np.array(directions)
        n_dirs = len(directions)

        def make_move(x, step_size, rng):
            """Make a lattice-based move from x with given step size."""
            # Pick a random direction
            d = directions[rng.integers(n_dirs)]
            # Sometimes flip a random coordinate sign for diversity
            if rng.random() < 0.1:
                flip = rng.integers(dim)
                d = d.copy()
                d[flip] *= -1
            y = x + step_size * d
            return np.clip(y, -5, 5)

        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        x_best = x.copy()
        fx_best = fx
        calls = 1

        # Start at a medium-large step size
        step_idx = 2  # start at 1.0
        no_improve = 0
        max_no_improve = 30

        for _ in range(budget - 1):
            if calls >= budget:
                break

            step_size = step_sizes[step_idx]
            y = make_move(x, step_size, rng)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                # Slightly increase step size (move toward larger steps)
                if step_idx > 0 and rng.random() < 0.3:
                    step_idx -= 1
                # Update best
                if fy < fx_best:
                    x_best = x.copy()
                    fx_best = fy
            else:
                no_improve += 1
                # Reduce step size if stuck
                if no_improve >= max_no_improve:
                    no_improve = 0
                    if step_idx < n_steps - 1:
                        step_idx += 1
                    # If at smallest step, try a restart
                    if step_idx == n_steps - 1:
                        # Restart from a random point, but keep best
                        x = rng.uniform(-4, 4, dim)
                        fx = func(x)
                        calls += 1
                        if fx < fx_best:
                            x_best = x.copy()
                            fx_best = fx
                        step_idx = 2  # reset to medium step
                        no_improve = 0

        return x_best, fx_best
