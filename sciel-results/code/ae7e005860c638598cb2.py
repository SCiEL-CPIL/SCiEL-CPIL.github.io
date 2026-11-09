"""Portfolio-based continuous optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer combining:
    1. Tight PSO for global exploration (no redundant evaluations)
    2. CMA-ES-lite with success-rate adaptation for local refinement
    3. Coordinate descent for final fine-tuning
    
    Uses time guard to stay within wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 330s limit, keep 45s safety margin
        budget = self.budget
        dim = self.dim

        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            try:
                x = np.asarray(x, dtype=float)
                x = np.clip(x, -5.0, 5.0)
                f = func(x)
                f = float(f)
            except Exception:
                return np.inf
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Strategy 1: Tight PSO for global exploration ---
        n_particles = 15
        n_pso_iters = 20  # Total calls: 15*20 = 300 max, but we'll stop early
        
        # Initialize particles
        particles = rng.uniform(-4, 4, (n_particles, dim))
        velocities = rng.uniform(-0.5, 0.5, (n_particles, dim))
        pbest_x = particles.copy()
        pbest_f = np.full(n_particles, np.inf)
        gbest_x = None
        gbest_f = np.inf

        # Evaluate initial positions
        for i in range(n_particles):
            if calls_made >= budget or time.time() > deadline:
                break
            f = safe_call(particles[i])
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest_x = particles[i].copy()

        # PSO iteration - evaluate AFTER update to save calls
        w = 0.7
        c1 = 1.4
        c2 = 1.4

        for it in range(n_pso_iters):
            if calls_made >= budget or time.time() > deadline:
                break

            # Update velocities and positions
            for i in range(n_particles):
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] +
                                c1 * r1 * (pbest_x[i] - particles[i]) +
                                c2 * r2 * (gbest_x - particles[i]))
                velocities[i] = np.clip(velocities[i], -2.0, 2.0)
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)

            # Evaluate updated positions
            for i in range(n_particles):
                if calls_made >= budget or time.time() > deadline:
                    break
                f = safe_call(particles[i])
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = particles[i].copy()
                if f < gbest_f:
                    gbest_f = f
                    gbest_x = particles[i].copy()

            # Adapt parameters
            w = max(0.4, w * 0.97)
            c1 = max(0.5, c1 * 0.98)
            c2 = min(2.0, c2 * 1.01)

        # Use the best point found by PSO as starting point for CMA-ES
        if gbest_x is not None:
            x_start = gbest_x.copy()
            fx_start = gbest_f
        else:
            x_start = np.zeros(dim)
            fx_start = safe_call(x_start)

        # --- Strategy 2: CMA-ES-lite with success-rate adaptation ---
        x = x_start.copy()
        fx = fx_start
        sigma = 1.0
        success_count = 0
        window = 12

        for step in range(budget - calls_made):
            if calls_made >= budget or time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = safe_call(y)

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                sigma *= 1.15
            else:
                sigma *= 0.88

            # Adapt sigma based on success rate in window
            if step > 0 and step % window == 0:
                # Target success rate ~0.35 (optimal for (1+1)-ES in high-dim)
                if success_count / window > 0.5:
                    sigma *= 1.3
                elif success_count / window < 0.2:
                    sigma *= 0.7
                success_count = 0

            if sigma < 1e-5:
                break
            if sigma > 10.0:
                sigma = 10.0

        # --- Strategy 3: Additional restarts from diverse points ---
        n_additional = 4
        for s in range(n_additional):
            if calls_made >= budget or time.time() > deadline:
                break

            # Diverse starting points
            if s == 0 and best_x is not None:
                x = best_x + rng.normal(0, 0.3, dim)
            elif s == 1:
                x = np.zeros(dim) + rng.normal(0, 0.5, dim)
            elif s == 2:
                x = rng.uniform(-3, 3, dim)
            else:
                x = rng.uniform(-4, 4, dim)
            x = np.clip(x, -5.0, 5.0)
            fx = safe_call(x)
            if not np.isfinite(fx):
                continue

            # Short CMA-ES-lite run
            sigma = 0.8
            success_count = 0
            for step in range(min(40, budget - calls_made)):
                if calls_made >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.15
                else:
                    sigma *= 0.88

                if step > 0 and step % 10 == 0:
                    if success_count / 10 > 0.5:
                        sigma *= 1.2
                    elif success_count / 10 < 0.2:
                        sigma *= 0.75
                    success_count = 0

                if sigma < 1e-5:
                    break

        # --- Strategy 4: Final coordinate descent refinement ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            best_x = self._coordinate_descent(func, safe_call, best_x, best_f, budget, deadline, dim)
            best_f = safe_call(best_x)

        # Ensure we've made at least some calls
        if calls_made == 0:
            x0 = np.zeros(dim)
            safe_call(x0)

        return best_x, best_f

    def _coordinate_descent(self, func, safe_call, x, fx, budget, deadline, dim):
        """Coordinate descent with adaptive step size for final refinement."""
        x = x.copy()
        fx = safe_call(x)
        step_size = 0.3

        for _ in range(150):
            if time.time() > deadline or calls_made >= budget:
                break

            improved = False
            for d in range(dim):
                for direction in [1, -1]:
                    y = x.copy()
                    y[d] += direction * step_size
                    y = np.clip(y, -5.0, 5.0)
                    fy = safe_call(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                step_size *= 0.6
                if step_size < 1e-4:
                    break

        return x
# EVOLVE-BLOCK-END
