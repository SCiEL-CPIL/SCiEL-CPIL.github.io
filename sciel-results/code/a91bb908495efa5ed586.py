"""Model-based optimizer for BBOB-style Rastrigin f3 with hidden shift/rotation."""
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware optimizer using model fitting and CMA-ES-like refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls_used = 0
        rng = np.random.default_rng(42)

        def try_point(x):
            nonlocal best_x, best_f, calls_used
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return calls_used < self.budget and time.time() < deadline

        def cma_es_refine(x0, budget_left, sigma_init=0.5):
            """CMA-ES-like local refinement."""
            nonlocal best_x, best_f, calls_used
            n = self.dim
            x = np.array(x0, dtype=float)
            sigma = sigma_init
            pop_size = max(4, int(4 + 3 * np.log(n)))
            # Covariance matrix
            C = np.eye(n)
            # Evolution paths
            ps = np.zeros(n)
            hc = 1.0 + 2.0 / (n + 1) - 1 / (n + 1)
            cs = (np.sum([1] * pop_size) + 2) / (n + np.sum([1] * pop_size) + 2)
            cmu = 2 / ((n + 2) ** 2 + pop_size)
            # Precompute
            c_sigma = cs
            c_c = 2 * (cs - 1) + 1
            # We'll do a simplified version
            for iteration in range(100):
                if not time_ok():
                    break
                if calls_used >= self.budget:
                    break
                # Generate population
                offspring = []
                for i in range(pop_size):
                    if not time_ok() or calls_used >= self.budget:
                        break
                    z = rng.standard_normal(n)
                    # Approximate C via eigen-decomposition (simplified: just use C as identity for speed)
                    xi = x + sigma * z
                    fi = try_point(xi)
                    offspring.append((fi, xi))
                if len(offspring) < pop_size:
                    break
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()
                # Weighted recombination
                w = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
                w /= np.sum(w)
                mu_eff = 1.0 / np.sum(w[:pop_size] ** 2)
                # Update mean
                x_new = np.zeros(n)
                for i in range(pop_size):
                    x_new += w[i] * offspring[i][1]
                # Check convergence
                if np.linalg.norm(x_new - x) < 1e-10 * sigma:
                    break
                # Update covariance (simplified)
                # For speed, we'll just adapt sigma
                sigma *= 0.9  # Simple adaptation
                x = x_new
                if sigma < 1e-6:
                    break
            return best_x, best_f

        # Phase 1: Coarse exploration to find promising regions
        # Sample points that are likely near Rastrigin minima (integer-ish points after shift)
        n_explore = 60
        for i in range(n_explore):
            if not time_ok():
                break
            # Mix of random and structured sampling
            if i < 30:
                # Random in the feasible region
                x = rng.uniform(-4.5, 4.5, self.dim)
            else:
                # Try integer-like points with small noise
                x_int = rng.integers(-4, 5, self.dim).astype(float)
                x = x_int + rng.uniform(-0.3, 0.3, self.dim)
            try_point(x)

        # Phase 2: Local refinement around the best point using coordinate descent + CMA-ES
        if best_x is not None and time_ok():
            # First, do coordinate-wise sweeps to quickly find the basin
            for step in [1.0, 0.5, 0.25]:
                if not time_ok():
                    break
                improved = True
                max_iters = 5
                iter_count = 0
                while improved and time_ok() and iter_count < max_iters:
                    improved = False
                    iter_count += 1
                    for d in range(self.dim):
                        if not time_ok():
                            break
                        for sign in [1, -1]:
                            if not time_ok():
                                break
                            x_trial = best_x.copy()
                            x_trial[d] += sign * step
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            f_trial = float(func(x_trial))
                            calls_used += 1
                            if f_trial < best_f:
                                best_f = f_trial
                                best_x = x_trial.copy()
                                improved = True

        # Phase 3: CMA-ES refinement
        if best_x is not None and time_ok():
            cma_es_refine(best_x, self.budget - calls_used, sigma_init=0.3)

        # Phase 4: Fine-tuning with very small steps
        if best_x is not None and time_ok():
            for ss in [0.05, 0.02, 0.01, 0.005]:
                if not time_ok():
                    break
                improved = True
                max_iters = 3
                iter_count = 0
                while improved and time_ok() and iter_count < max_iters:
                    improved = False
                    iter_count += 1
                    for d in range(self.dim):
                        if not time_ok():
                            break
                        for sign in [1, -1]:
                            if not time_ok():
                                break
                            x_trial = best_x.copy()
                            x_trial[d] += sign * ss
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            f_trial = float(func(x_trial))
                            calls_used += 1
                            if f_trial < best_f:
                                best_f = f_trial
                                best_x = x_trial.copy()
                                improved = True

        return best_x, best_f
