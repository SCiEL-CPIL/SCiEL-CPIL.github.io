"""Optimizer for BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Parallel coordinate descent with golden-section 1D refinement and random shuffles."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial random point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        if budget <= 1:
            return x, fx

        # Golden-section search along a single coordinate
        # phi = (sqrt(5)-1)/2 ≈ 0.6180339887
        phi = (np.sqrt(5.0) - 1.0) / 2.0

        def golden_section_1d(x_base, coord, lo, hi, max_iter):
            """Minimize along coordinate `coord` of x_base in [lo, hi] using golden-section.
            Returns (best_x, best_f, num_calls)."""
            nonlocal calls
            a, b = lo, hi
            c = b - phi * (b - a)
            d = a + phi * (b - a)

            # Evaluate c and d
            xc = x_base.copy()
            xc[coord] = c
            fc = func(xc)
            calls += 1

            xd = x_base.copy()
            xd[coord] = d
            fd = func(xd)
            calls += 1

            best_x = x_base.copy()
            best_f = fc if fc < fd else fd
            best_val = c if fc < fd else d

            for _ in range(max_iter - 2):
                if calls >= budget:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - phi * (b - a)
                    xc = x_base.copy()
                    xc[coord] = c
                    fc = func(xc)
                    calls += 1
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + phi * (b - a)
                    xd = x_base.copy()
                    xd[coord] = d
                    fd = func(xd)
                    calls += 1

                if fc < fd:
                    if fc < best_f:
                        best_f = fc
                        best_val = c
                else:
                    if fd < best_f:
                        best_f = fd
                        best_val = d

            best_x = x_base.copy()
            best_x[coord] = best_val
            return best_x, best_f, calls

        # Coordinate order: start with natural order, then shuffle periodically
        coords = list(range(dim))
        
        # Number of full coordinate sweeps to attempt
        # Each sweep uses dim coordinates. Each golden-section uses ~8-12 evaluations.
        # Let's allocate budget carefully.
        # We'll do a loop: for each coordinate in current order, do a 1D search.
        # After each full sweep, shuffle coordinates.
        # Stop when budget is exhausted.

        # Estimate cost per coordinate: golden-section with ~10 iterations -> ~10 calls
        # Let's use a fixed number of golden-section iterations per coordinate per sweep
        gs_iters = 10  # evaluations per coordinate per sweep

        while calls < budget:
            # Shuffle coordinates to avoid getting stuck in a particular order
            if calls > 1:  # Don't shuffle before the first sweep
                rng.shuffle(coords)

            for coord in coords:
                if calls >= budget:
                    break

                # Determine search range: current value ± some window
                # Use a window that covers the box but focused around current value
                # Start with a wide window, narrow down
                window = 2.0
                lo = max(-5.0, x[coord] - window)
                hi = min(5.0, x[coord] + window)

                if hi - lo < 1e-10:
                    continue

                # Perform golden-section search
                x_new, fx_new, calls = golden_section_1d(x, coord, lo, hi, gs_iters)

                if fx_new < fx:
                    x = x_new
                    fx = fx_new

                if calls >= budget:
                    break

        return x, fx
# EVOLVE-BLOCK-END
