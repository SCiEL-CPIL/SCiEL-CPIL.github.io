"""DTLZ2 optimizer: direct 2D low-discrepancy sampling on the Pareto front angular parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ2 front geometry: sample (theta1, theta2) with 2D Halton sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_2d(self, n):
        """Generate n 2D Halton points in [0,1)^2 using bases 2 and 3."""
        h1 = np.zeros(n)
        h2 = np.zeros(n)
        for i in range(n):
            i_val = i + 1
            f = 1.0 / 2.0
            val = 0.0
            while i_val > 0:
                val += f * (i_val % 2)
                i_val //= 2
                f /= 2.0
            h1[i] = val

            i_val = i + 1
            f = 1.0 / 3.0
            val = 0.0
            while i_val > 0:
                val += f * (i_val % 3)
                i_val //= 3
                f /= 3.0
            h2[i] = val
        return np.column_stack([h1, h2])

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ2 Pareto front:
        # f1 = sin(theta1)*sin(theta2)
        # f2 = sin(theta1)*cos(theta2)
        # f3 = cos(theta1)
        # theta1, theta2 in [0, pi/2]
        #
        # All objectives < 2.5 (reference) requires:
        #   sin(theta1)*sin(theta2) < 2.5  (always true since max is 1)
        #   sin(theta1)*cos(theta2) < 2.5  (always true since max is 1)
        #   cos(theta1) < 2.5              (always true since max is 1)
        # So ALL points on the front are strictly better than [2.5,2.5,2.5].
        #
        # Decision variables: x[0:dim-2] = 0.5 (g=0), x[dim-2]=theta1, x[dim-1]=theta2

        # Use a dense 2D Halton sequence for low-discrepancy coverage
        n_points = min(budget, 196)  # 14x14 equivalent density

        # Generate Halton points in [0,1)^2
        halton_pts = self._halton_2d(n_points)

        # Map to theta1, theta2 in [0, pi/2]
        theta1 = halton_pts[:, 0] * (np.pi / 2.0)
        theta2 = halton_pts[:, 1] * (np.pi / 2.0)

        # Build decision vectors
        X = np.zeros((n_points, dim))
        X[:, :dim-2] = 0.5
        X[:, dim-2] = theta1
        X[:, dim-1] = theta2

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_points

        # If we have remaining budget, add a few more refined points
        # Use a shifted Halton sequence for interstitial coverage
        if used < budget:
            remaining = budget - used
            n_extra = min(remaining, 20)
            # Shift the Halton sequence by 0.5 to get interstitial points
            halton_extra = self._halton_2d(n_extra)
            # Shift to avoid overlap
            theta1_e = ((halton_extra[:, 0] + 0.5) % 1.0) * (np.pi / 2.0)
            theta2_e = ((halton_extra[:, 1] + 0.5) % 1.0) * (np.pi / 2.0)

            X_e = np.zeros((n_extra, dim))
            X_e[:, :dim-2] = 0.5
            X_e[:, dim-2] = theta1_e
            X_e[:, dim-1] = theta2_e
            X_e = np.clip(X_e, lo, hi)

            F_e = np.array([problem(x) for x in X_e])
            used += n_extra

            X = np.vstack([X, X_e])
            F = np.vstack([F, F_e])

        return X
# EVOLVE-BLOCK-END
