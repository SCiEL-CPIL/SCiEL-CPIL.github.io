"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with golden-section search exploiting Bent Cigar anisotropy."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Golden section search in 1D
        golden_ratio = (np.sqrt(5) - 1) / 2  # ~0.618
        inv_golden = 1 - golden_ratio  # ~0.382

        def golden_section_1d(f, a, b, x_fixed, dim_idx, max_evals=10, tol=1e-10):
            """Perform golden-section search on dimension dim_idx."""
            # f here is a function that takes a full x vector
            # We'll create a 1D wrapper
            def f1d(t):
                x = x_fixed.copy()
                x[dim_idx] = t
                return f(x)

            # Initial bracket
            c = a + inv_golden * (b - a)
            d = a + golden_ratio * (b - a)
            fc = f1d(c)
            fd = f1d(d)
            evals = 2

            for _ in range(max_evals - 2):
                if time.time() > deadline:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = a + inv_golden * (b - a)
                    fc = f1d(c)
                    evals += 1
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + golden_ratio * (b - a)
                    fd = f1d(d)
                    evals += 1

                if abs(b - a) < tol:
                    break

            # Return the better point
            if fc < fd:
                return c, fc, evals
            else:
                return d, fd, evals

        # Initialize with a random point
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # First pass: coarse coordinate descent with wider brackets
        # For Bent Cigar, dimension 0 has a very narrow valley
        # We do multiple passes, each refining all dimensions

        # Strategy: alternate between wide and narrow brackets
        # In each pass, for each dimension, do a golden section search

        passes = 0
        while calls < budget and time.time() < deadline:
            passes += 1
            improved = False

            for dim_idx in range(dim):
                if calls >= budget or time.time() > deadline:
                    break

                # Determine bracket based on pass number
                # Early passes: wide bracket, later: narrow
                if passes <= 2:
                    # Wide bracket: explore the full range around current best
                    center = best_x[dim_idx]
                    # For dim 0 (narrow valley), use a smaller initial bracket
                    if dim_idx == 0:
                        half_width = 2.0 / (passes + 1)
                    else:
                        half_width = 5.0 / (passes + 1)

                    a = max(-5.0, center - half_width)
                    b = min(5.0, center + half_width)
                else:
                    # Narrow bracket: refine around current best
                    center = best_x[dim_idx]
                    if dim_idx == 0:
                        half_width = 0.5 / passes
                    else:
                        half_width = 1.0 / passes

                    a = max(-5.0, center - half_width)
                    b = min(5.0, center + half_width)

                # Ensure bracket is valid
                if b - a < 1e-12:
                    continue

                # Determine max evaluations for this 1D search
                # More evaluations for narrow valley dimension
                if dim_idx == 0:
                    max_evals_1d = min(12, budget - calls)
                else:
                    max_evals_1d = min(8, budget - calls)

                if max_evals_1d < 3:
                    break

                # Perform golden section search
                best_t, best_f, evals_used = golden_section_1d(
                    func, a, b, best_x, dim_idx, max_evals_1d
                )
                calls += evals_used

                # Update best
                if best_f < best_fx:
                    best_x[dim_idx] = best_t
                    best_fx = best_f
                    improved = True

            # If no improvement in a full pass, tighten brackets
            if not improved:
                # Try a random restart in a small neighborhood
                if passes > 10 and calls < budget - 20:
                    # Perturb the best solution
                    for dim_idx in range(dim):
                        if time.time() > deadline:
                            break
                        # Try a few random perturbations
                        for _ in range(3):
                            if calls >= budget or time.time() > deadline:
                                break
                            perturbation = rng.normal(0, 0.1 / (passes + 1), 1)
                            x_new = best_x.copy()
                            x_new[dim_idx] += perturbation[0]
                            x_new = np.clip(x_new, -5, 5)
                            f_new = func(x_new)
                            calls += 1
                            if f_new < best_fx:
                                best_x = x_new
                                best_fx = f_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
