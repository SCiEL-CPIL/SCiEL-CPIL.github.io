"""Rewritten optimizer: multi-start stochastic quasi-Newton with finite-difference gradients."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start BFGS-like optimizer with finite-difference gradients."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        # Safe function wrapper
        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            try:
                val = func(x)
                if not np.isfinite(val):
                    return np.inf
                return val
            except Exception:
                return np.inf

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))
        # Axis-aligned points
        for d in range(dim):
            p = np.zeros(dim)
            p[d] = 3.0
            starts.append(p)
            q = np.zeros(dim)
            q[d] = -3.0
            starts.append(q)

        # Shuffle starts
        idx = rng.permutation(len(starts))
        starts = [starts[i] for i in idx]

        # BFGS-like local optimizer from a starting point
        def local_optimize(x0, remaining_budget):
            nonlocal best_x, best_f, calls_made

            if calls_made >= budget:
                return

            x = x0.copy()
            fx = safe_func(x)
            if not np.isfinite(fx):
                return

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Finite-difference gradient estimation
            eps = 1e-4

            def gradient(x):
                nonlocal calls_made
                if calls_made >= budget:
                    return np.zeros(dim)
                g = np.zeros(dim)
                for i in range(dim):
                    xp = x.copy()
                    xp[i] += eps
                    fp = safe_func(xp)
                    if not np.isfinite(fp):
                        return None
                    xm = x.copy()
                    xm[i] -= eps
                    fm = safe_func(xm)
                    if not np.isfinite(fm):
                        return None
                    g[i] = (fp - fm) / (2 * eps)
                return g

            # BFGS inverse Hessian approximation
            H = np.eye(dim)
            step_size = 0.5

            for step in range(100):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                # Check budget for gradient (dim*2 + 1 calls per iteration)
                if calls_made + 2 * dim + 1 > budget:
                    break

                g = gradient(x)
                if g is None:
                    break

                if np.linalg.norm(g) < 1e-6:
                    break

                # Compute search direction
                d = -H @ g
                d_norm = np.linalg.norm(d)
                if d_norm < 1e-10:
                    d = -g
                    d_norm = np.linalg.norm(d)

                # Line search
                alpha = step_size
                improved = False
                for ls in range(10):
                    x_new = np.clip(x + alpha * d, -5, 5)
                    f_new = safe_func(x_new)
                    if not np.isfinite(f_new):
                        alpha *= 0.5
                        continue
                    if f_new < fx:
                        # Update BFGS
                        s = x_new - x
                        g_new = gradient(x_new)
                        if g_new is not None:
                            y = g_new - g
                            sy = s @ y
                            if sy > 1e-8:
                                Hy = H @ y
                                H = H - (s @ Hy) / (y @ Hy) * np.outer(y, s.T) + (s @ s) / sy
                                # Symmetrize
                                H = (H + H.T) / 2
                                # Ensure positive definite-ish
                                H = np.maximum(H, 0) + 1e-6 * np.eye(dim)
                        x, fx = x_new, f_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        step_size = min(step_size * 1.5, 2.0)
                        improved = True
                        break
                    else:
                        alpha *= 0.5
                        step_size = max(step_size * 0.7, 0.01)

                if not improved:
                    # Try a random restart from current point
                    x_try = np.clip(x + 0.5 * rng.standard_normal(dim), -5, 5)
                    f_try = safe_func(x_try)
                    if np.isfinite(f_try) and f_try < fx:
                        x, fx = x_try, f_try
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        step_size = 0.3
                    else:
                        step_size = max(step_size * 0.5, 0.01)
                        if step_size < 0.01:
                            break

            return x, fx

        # Multi-start loop
        for x0 in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline - 10.0:
                break

            # Evaluate starting point
            f0 = safe_func(x0)
            if not np.isfinite(f0):
                continue
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # If this is a good point, run local optimization
            local_optimize(x0, budget - calls_made)

        # If we still have budget, do more random exploration
        while calls_made < budget and time.time() < deadline - 10.0:
            x_rand = rng.uniform(-5, 5, dim)
            f_rand = safe_func(x_rand)
            if np.isfinite(f_rand):
                if f_rand < best_f:
                    best_f = f_rand
                    best_x = x_rand.copy()
                # If significantly better than current best, try local opt
                if f_rand < best_f + 0.1 and calls_made + 2 * dim + 10 < budget:
                    local_optimize(x_rand, budget - calls_made)

        return best_x, best_f
# EVOLVE-BLOCK-END
