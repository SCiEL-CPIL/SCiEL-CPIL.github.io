"""Rewritten optimizer: deterministic Halton multi-scale grid + adaptive local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Halton-based deterministic exploration followed by adaptive local exploitation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    @staticmethod
    def _halton_sequence(n, dim, bases=None):
        """Generate n Halton sequence points in [0,1]^dim."""
        if bases is None:
            # Use first dim primes
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            bases = primes[:dim]
        pts = np.zeros((n, dim))
        for d in range(dim):
            base = bases[d]
            for i in range(n):
                f = 1.0
                result = 0.0
                i_val = i + 1
                while i_val > 0:
                    i_val, remainder = divmod(i_val, base)
                    f /= base
                    result += remainder * f
                pts[i, d] = result
        return pts

    def __call__(self, func):
        start = time.time()
        deadline = start + 285.0  # safety margin vs 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def consider(x, f):
            nonlocal best_x, best_f, call_count
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f < best_f

        # ---- Phase 1: Deterministic multi-scale Halton exploration ----
        # Generate Halton points at multiple scales to cover the domain systematically.
        # Scale 1: coarse grid over [-5, 5]^dim
        # Scale 2: medium grid
        # Scale 3: fine grid
        # We allocate calls to each scale based on diminishing returns.

        # Total exploration budget: ~40% of total budget
        n_explore = min(200, budget // 2)

        # Split exploration into 3 scales
        n_coarse = n_explore // 4      # ~50
        n_medium = n_explore // 3      # ~66
        n_fine = n_explore - n_coarse - n_medium  # remainder

        # Coarse scale: [-5, 5]
        if n_coarse > 0:
            pts_coarse = self._halton_sequence(n_coarse, dim)
            pts_coarse = 10.0 * pts_coarse - 5.0
            for x in pts_coarse:
                if time.time() > deadline or call_count >= budget:
                    break
                f = func(x)
                call_count += 1
                consider(x, f)

        # Medium scale: [-4, 4]
        if n_medium > 0:
            pts_medium = self._halton_sequence(n_medium, dim)
            pts_medium = 8.0 * pts_medium - 4.0
            for x in pts_medium:
                if time.time() > deadline or call_count >= budget:
                    break
                f = func(x)
                call_count += 1
                consider(x, f)

        # Fine scale: [-3, 3]
        if n_fine > 0:
            pts_fine = self._halton_sequence(n_fine, dim)
            pts_fine = 6.0 * pts_fine - 3.0
            for x in pts_fine:
                if time.time() > deadline or call_count >= budget:
                    break
                f = func(x)
                call_count += 1
                consider(x, f)

        if best_x is None:
            # Fallback: uniform random
            rng = np.random.default_rng(42)
            for _ in range(min(budget - call_count, 32)):
                if time.time() > deadline or call_count >= budget:
                    break
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                call_count += 1
                consider(x, f)
            return best_x, best_f

        # ---- Phase 2: Adaptive local search around best point ----
        # Use a multi-start local search with adaptive step sizes.
        # Weierstrass is non-smooth, so we use a simple perturbation-based approach
        # with careful step size management.

        x = best_x.copy()
        fx = best_f
        remaining = budget - call_count
        if remaining <= 0:
            return best_x, best_f

        rng = np.random.default_rng(777)

        # Adaptive step size: start moderate, adjust based on success
        sigma = 1.0
        min_sigma = 1e-4
        max_sigma = 3.0
        stagnation_counter = 0
        stagnation_threshold = 15

        for step in range(remaining):
            if time.time() > deadline or call_count >= budget:
                break

            # If stagnating, try a larger jump to escape local minima
            if stagnation_counter >= stagnation_threshold:
                # Large perturbation to escape
                y = np.clip(x + max_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                if fy < fx:
                    x, fx = y, fy
                    consider(x, fx)
                    sigma = max(sigma, 1.0)
                    stagnation_counter = 0
                else:
                    # Try another escape direction
                    y = np.clip(x + max_sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    call_count += 1
                    if fy < fx:
                        x, fx = y, fy
                        consider(x, fx)
                        sigma = max(sigma, 1.0)
                        stagnation_counter = 0
                    else:
                        stagnation_counter = 0  # Reset after escape attempt
                        sigma = max(sigma * 0.5, min_sigma)
                continue

            # Normal exploration step
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x, fx = y, fy
                consider(x, fx)
                sigma = min(sigma * 1.2, max_sigma)
                stagnation_counter = 0
            else:
                sigma = max(sigma * 0.8, min_sigma)
                stagnation_counter += 1

            # Every 50 steps, try a coordinate-wise refinement
            if step % 50 == 49 and sigma > 0.1:
                for d in range(dim):
                    if time.time() > deadline or call_count >= budget:
                        break
                    # Try positive and negative step in each dimension
                    for sign in [1.0, -1.0]:
                        if time.time() > deadline or call_count >= budget:
                            break
                        y_coord = x.copy()
                        y_coord[d] += sign * sigma * 0.5
                        y_coord = np.clip(y_coord, -5, 5)
                        fy_coord = func(y_coord)
                        call_count += 1
                        if fy_coord < fx:
                            x, fx = y_coord, fy_coord
                            consider(x, fx)
                            break  # Found improvement, stop trying other directions

        return best_x, best_f
# EVOLVE-BLOCK-END
