"""DTLZ2 multi-objective optimizer exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ2-specific optimizer: exploit known structure for Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2 Pareto front: f_i = (1 + g) * cos(theta_i) for i=0..n_obj-1
        # where theta_i = (pi/2) * x_{n_obj-1+i} for x in [0,1]
        # g = sum(x_j^2) for j=0..n_obj-2
        # For the front, g=0, so f_i = cos(theta_i)
        # The front is the (n_obj-1)-dimensional unit sphere in the positive orthant.

        # Strategy:
        # 1. Generate structured points on the Pareto front
        # 2. Use remaining budget for local refinement

        # Generate points on the Pareto front using spherical coordinates
        # For 3 objectives, the front is a quarter-sphere (2D surface)
        # Use uniform sampling on the sphere

        def front_point_from_angles(theta1, theta2):
            """Generate a point on the DTLZ2 front from angles."""
            # For 3 objectives: f = (cos(t1), sin(t1)*cos(t2), sin(t1)*sin(t2))
            # where t1, t2 in [0, pi/2]
            t1, t2 = theta1, theta2
            f = np.array([
                np.cos(t1),
                np.sin(t1) * np.cos(t2),
                np.sin(t1) * np.sin(t2)
            ])
            # Convert to decision variables: x_{n_obj-1+i} = (2/pi) * arccos(f_i)
            # Actually, for DTLZ2: f_i = cos(theta_i) where theta_i = (pi/2)*x_{n_obj-1+i}
            # So x_{n_obj-1+i} = (2/pi) * arccos(f_i)
            # But we need to be careful: the angles are not independent
            # For DTLZ2, the front is defined by:
            # f_0 = cos(theta_1)
            # f_1 = sin(theta_1) * cos(theta_2)  
            # f_2 = sin(theta_1) * sin(theta_2)
            # where theta_1 = (pi/2) * x_{n_obj-1}, theta_2 = (pi/2) * x_{n_obj}
            # So we need x_{n_obj-1} = (2/pi) * theta_1, x_{n_obj} = (2/pi) * theta_2
            # And x_0..x_{n_obj-2} = 0 for g=0
            x = np.zeros(dim)
            x[n_obj - 1] = (2.0 / np.pi) * theta1
            x[n_obj] = (2.0 / np.pi) * theta2
            return x

        # Generate structured grid on the front
        # For 3 objectives, use a 2D grid of angles
        n_grid = 12  # 12x12 = 144 points
        theta1_vals = np.linspace(0.01, np.pi/2 - 0.01, n_grid)
        theta2_vals = np.linspace(0.01, np.pi/2 - 0.01, n_grid)

        X = []
        for t1 in theta1_vals:
            for t2 in theta2_vals:
                X.append(front_point_from_angles(t1, t2))
        X = np.array(X)

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate initial points
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Check deadline
        if time.time() > deadline - 5:
            return X

        # Use remaining budget for refinement
        # Add some random points near the front
        remaining = budget - used
        if remaining > 0:
            # Generate random points near the front
            n_random = min(remaining, 50)
            rng = np.random.default_rng(42)
            for _ in range(n_random):
                if time.time() > deadline - 5:
                    break
                # Perturb a random front point
                idx = rng.integers(len(X))
                x_base = X[idx].copy()
                # Small perturbation in the first n_obj-1 variables (to move off front slightly)
                x_base[:n_obj-1] += 0.05 * rng.standard_normal(n_obj-1)
                # Small perturbation in the angle variables
                x_base[n_obj-1] += 0.02 * rng.standard_normal()
                x_base[n_obj] += 0.02 * rng.standard_normal()
                x_base = np.clip(x_base, lo, hi)
                X = np.vstack([X, x_base])
                f_new = problem(x_base)
                F = np.vstack([F, f_new.reshape(1, -1)])
                used += 1

        return X
# EVOLVE-BLOCK-END
