"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic coarse-to-fine search using Halton sequence with local refinement."""

    @staticmethod
    def _halton_sequence(n, dim, base=None):
        """Generate n points in [0,1)^dim using the Halton sequence."""
        if base is None:
            # First dim primes
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            base = primes[:dim]
        points = np.zeros((n, dim))
        for d in range(dim):
            b = base[d]
            f = 1.0
            for i in range(1, n + 1):
                i_val = i
                while i_val > 0:
                    f /= b
                    points[i - 1, d] += f * (i_val % b)
                    i_val //= b
        return points

    @staticmethod
    def _halton_sequence_fast(n, dim):
        """Faster Halton sequence generation using vectorized operations."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        base = primes[:dim]
        points = np.zeros((n, dim))
        for d in range(dim):
            b = base[d]
            idx = np.arange(1, n + 1)
            f = 1.0 / b
            frac = np.zeros(n)
            i_val = idx.copy()
            while np.any(i_val > 0):
                frac += f * (i_val % b)
                i_val //= b
                f /= b
            points[:, d] = frac
        return points

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.low = -5.0
        self.high = 5.0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        low, high = self.low, self.high

        # Phase 1: Coarse grid search with Halton sequence
        # Use a reasonable number of initial samples
        n_coarse = min(100, budget // 4)
        halton = self._halton_sequence_fast(n_coarse, dim)
        x_coarse = low + (high - low) * halton

        # Evaluate all coarse points
        best_x = None
        best_f = np.inf
        calls_made = 0

        for i in range(n_coarse):
            x = x_coarse[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement around the best point
        # Use a local search with adaptive step size
        remaining = budget - calls_made
        if remaining > 0:
            # Local refinement using a combination of coordinate search and small random perturbations
            x = best_x.copy()
            fx = best_f
            step = 1.0

            # Adaptive local search
            while calls_made < budget and step > 0.001:
                improved = False
                # Try moves in each coordinate direction
                for d in range(dim):
                    for direction in [-1, 1]:
                        y = x.copy()
                        y[d] += direction * step
                        y = np.clip(y, low, high)
                        if calls_made >= budget:
                            break
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                    if calls_made >= budget:
                        break
                if not improved:
                    step *= 0.5
                else:
                    step *= 0.8

            best_x = x
            best_f = fx

        # Phase 3: If we have budget left, do a second pass with finer Halton around best region
        remaining = budget - calls_made
        if remaining > 10:
            # Focus search around the current best
            center = best_x
            radius = 2.0
            n_fine = min(remaining, 50)
            halton_fine = self._halton_sequence_fast(n_fine, dim)
            x_fine = center + radius * (2 * halton_fine - 1)
            x_fine = np.clip(x_fine, low, high)

            for i in range(n_fine):
                if calls_made >= budget:
                    break
                x = x_fine[i]
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Another local refinement if we found something better
            if calls_made < budget:
                x = best_x.copy()
                fx = best_f
                step = 0.5
                while calls_made < budget and step > 0.0005:
                    improved = False
                    for d in range(dim):
                        for direction in [-1, 1]:
                            y = x.copy()
                            y[d] += direction * step
                            y = np.clip(y, low, high)
                            if calls_made >= budget:
                                break
                            fy = func(y)
                            calls_made += 1
                            if fy < fx:
                                x = y
                                fx = fy
                                improved = True
                        if calls_made >= budget:
                            break
                    if not improved:
                        step *= 0.5
                    else:
                        step *= 0.8
                best_x = x
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
