"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Anisotropic search for BBOB f12 (Bent Cigar) optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        d = self.dim
        N = self.budget

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_f, best_x, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse sampling to find a good basin
        # Sample points in a grid-like fashion
        n_coarse = min(50, N - 5)
        for i in range(n_coarse):
            if time.time() > deadline or calls_made >= N:
                break
            # Use a quasi-random sequence for better coverage
            t = (i + 1) / (n_coarse + 1)
            x = np.zeros(d)
            for j in range(d):
                # Use golden ratio sequence for low-discrepancy
                x[j] = -5 + 10 * ((t * (j + 1) * 0.618033988749895) % 1.0)
            x = np.clip(x, -5, 5)
            evaluate(x)

        # Also sample some random points
        for i in range(min(20, N - calls_made)):
            if time.time() > deadline or calls_made >= N:
                break
            x = rng.uniform(-5, 5, size=d)
            evaluate(x)

        if best_x is None:
            return np.zeros(d), np.inf

        # Phase 2: Identify the high-curvature direction
        # Probe each dimension to estimate curvature
        probe_step = 0.5
        curvatures = np.zeros(d)
        base_f = best_f
        for j in range(d):
            if time.time() > deadline or calls_made >= N:
                break
            x_plus = best_x.copy()
            x_plus[j] += probe_step
            x_plus = np.clip(x_plus, -5, 5)
            fp = evaluate(x_plus)

            x_minus = best_x.copy()
            x_minus[j] -= probe_step
            x_minus = np.clip(x_minus, -5, 5)
            fm = evaluate(x_minus)

            # Approximate curvature: (f+ + f- - 2*f0) / step^2
            curvatures[j] = (fp + fm - 2 * base_f) / (probe_step ** 2)

        # The high-curvature dimension is the one with largest curvature
        high_curv_dim = np.argmax(curvatures)

        # Phase 3: 1D search along the high-curvature dimension
        # Use golden section search
        a = max(-5, best_x[high_curv_dim] - 2.0)
        b = min(5, best_x[high_curv_dim] + 2.0)
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate

        # Golden section search iterations
        max_1d_iters = 30
        for _ in range(max_1d_iters):
            if time.time() > deadline or calls_made >= N:
                break
            c = b - gr * (b - a)
            c = np.clip(c, -5, 5)
            d_val = a + gr * (b - a)
            d_val = np.clip(d_val, -5, 5)

            x_c = best_x.copy()
            x_c[high_curv_dim] = c
            fc = evaluate(x_c)

            x_d = best_x.copy()
            x_d[high_curv_dim] = d_val
            fd = evaluate(x_d)

            if fc < fd:
                b = d_val
            else:
                a = c

        # After 1D search, best_x should be updated
        # Phase 4: Optimize the remaining dimensions
        # Use a simple coordinate descent / local search on other dimensions
        remaining_dims = [j for j in range(d) if j != high_curv_dim]
        
        # Do a few rounds of coordinate descent on remaining dims
        for round_num in range(3):
            if time.time() > deadline or calls_made >= N:
                break
            for j in remaining_dims:
                if time.time() > deadline or calls_made >= N:
                    break
                # Golden section search along dimension j
                a = max(-5, best_x[j] - 2.0)
                b = min(5, best_x[j] + 2.0)
                for _ in range(15):
                    if time.time() > deadline or calls_made >= N:
                        break
                    c = b - gr * (b - a)
                    c = np.clip(c, -5, 5)
                    d_val = a + gr * (b - a)
                    d_val = np.clip(d_val, -5, 5)

                    x_c = best_x.copy()
                    x_c[j] = c
                    fc = evaluate(x_c)

                    x_d = best_x.copy()
                    x_d[j] = d_val
                    fd = evaluate(x_d)

                    if fc < fd:
                        b = d_val
                    else:
                        a = c

        # Phase 5: Fine-tune all dimensions with small steps
        # Use a simple random perturbation search
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        for step in step_sizes:
            if time.time() > deadline or calls_made >= N:
                break
            # Do a few iterations at this step size
            max_iters = max(5, (N - calls_made) // 10)
            improved = False
            for _ in range(max_iters):
                if time.time() > deadline or calls_made >= N:
                    break
                x_new = best_x + rng.normal(0, step, size=d)
                x_new = np.clip(x_new, -5, 5)
                fx = evaluate(x_new)
                if fx < best_f - 1e-12:
                    improved = True

            if not improved:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
