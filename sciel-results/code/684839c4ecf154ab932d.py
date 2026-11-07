"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function optimizer: exact gradient recovery via simplex probing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution
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

        # Step 1: Evaluate at origin (call 1) to get intercept b
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

        # Step 2: Probe along each axis to recover exact gradient w
        # f(e_i * h) = w_i * h + b  =>  w_i = (f(h*e_i) - f(0)) / h
        # Use h=5 to maximize signal-to-noise and stay within bounds
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
            if f_probe < best_f:
                best_f = f_probe
                best_x = probe.copy()

        if gradient_valid:
            # Step 3: Compute optimal vertex directly from gradient
            # For linear f(x) = w^T x + b, min over [-5,5]^d is at:
            # x_i = -5 if w_i > 0, +5 if w_i < 0, 0 if w_i == 0
            opt_x = np.zeros(dim)
            for i in range(dim):
                if w[i] > 1e-12:
                    opt_x[i] = -5.0
                elif w[i] < -1e-12:
                    opt_x[i] = 5.0
                else:
                    opt_x[i] = 0.0

            # Verify the optimal vertex (one more call)
            if calls_made < budget and time_ok():
                f_opt = safe_call(opt_x)
                if f_opt is not None and f_opt < best_f:
                    best_f = f_opt
                    best_x = opt_x.copy()
            else:
                # Can't verify, but we're confident in the linear model
                best_x = opt_x.copy()
                # Estimate f_opt from the model
                est_f = np.dot(w, opt_x) + f_origin
                best_f = min(best_f, est_f)

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
