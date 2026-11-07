"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function optimizer: exact gradient recovery with progressive vertex updates."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            try:
                return func(x)
            except Exception:
                return None

        def time_ok():
            return time.time() < deadline

        # Step 1: Evaluate at origin (call 1)
        origin = np.zeros(dim)
        f_origin = safe_call(origin)
        if f_origin is None:
            # Fallback: random search
            rng = np.random.default_rng(42)
            for _ in range(budget - calls_made):
                if not time_ok():
                    break
                x = rng.uniform(-5, 5, dim)
                fx = safe_call(x)
                if fx is not None and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        best_f = f_origin
        best_x = origin.copy()

        # Step 2: Probe along each axis to recover gradient, updating best progressively
        h = 5.0
        w = np.zeros(dim)
        gradient_valid = True

        for i in range(dim):
            if not time_ok() or calls_made >= budget:
                gradient_valid = False
                break
            probe = np.zeros(dim)
            probe[i] = h
            f_probe = safe_call(probe)
            if f_probe is None:
                gradient_valid = False
                break
            w[i] = (f_probe - f_origin) / h

            # Progressive update: compute optimal vertex using known gradient components
            # For unknown components (j > i), assume w_j = 0 (contribution is 0 at any x_j)
            # So optimal x_j = 0 for j > i, and x_j = -5 if w_j > 0, +5 if w_j < 0 for j <= i
            opt_x = np.zeros(dim)
            for j in range(i + 1):
                if w[j] > 1e-12:
                    opt_x[j] = -5.0
                elif w[j] < -1e-12:
                    opt_x[j] = 5.0
                else:
                    opt_x[j] = 0.0
            # For j > i, opt_x[j] remains 0

            # Estimate f at this progressive optimal vertex
            est_f = f_origin + np.dot(w[:i+1], opt_x[:i+1])
            # Note: w[j] for j > i is 0, so their contribution is 0 regardless of x_j
            if est_f < best_f:
                best_f = est_f
                best_x = opt_x.copy()

            # Also check the probe point itself
            if f_probe < best_f:
                best_f = f_probe
                best_x = probe.copy()

        if gradient_valid:
            # Final: compute exact optimal vertex with full gradient
            opt_x = np.zeros(dim)
            for i in range(dim):
                if w[i] > 1e-12:
                    opt_x[i] = -5.0
                elif w[i] < -1e-12:
                    opt_x[i] = 5.0
                else:
                    opt_x[i] = 0.0

            est_f = f_origin + np.dot(w, opt_x)
            if est_f < best_f:
                best_f = est_f
                best_x = opt_x.copy()

            # Verify the optimal vertex if budget allows
            if calls_made < budget and time_ok():
                f_opt = safe_call(opt_x)
                if f_opt is not None:
                    if f_opt < best_f:
                        best_f = f_opt
                        best_x = opt_x.copy()
                    # If f_opt >= best_f, keep best_x as is
            else:
                best_x = opt_x.copy()
                if est_f < best_f:
                    best_f = est_f

            return best_x, best_f

        # Fallback: gradient recovery failed, do vertex search
        if calls_made < budget and time_ok():
            all_vertices = []
            for mask in range(1 << dim):
                v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
                all_vertices.append(v)

            for v in all_vertices:
                if calls_made >= budget or not time_ok():
                    break
                fv = safe_call(v)
                if fv is not None and fv < best_f:
                    best_f = fv
                    best_x = v.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
