"""Portfolio-based continuous optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer combining:
    1. Robust PSO for global exploration (longer run, more particles)
    2. Adaptive (1+1)-ES for local refinement (handles step functions)
    3. Coordinate descent for final fine-tuning (if budget remains)
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s safety margin
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
        
        # --- Strategy 1: Robust PSO for global exploration ---
        # Inspired by the top-scoring solution: use more particles and more iterations
        # to ensure thorough exploration of the step-ellipsoidal landscape.
        n_particles = 20
        # We will run PSO until we have used ~80% of the budget, leaving room for ES
        max_pso_calls = int(budget * 0.8)
        
        # Initialize particles
        particles = rng.uniform(-4, 4, (n_particles, dim))
        velocities = rng.uniform(-1, 1, (n_particles, dim))
        pbest_x = particles.copy()
        pbest_f = np.full(n_particles, np.inf)
        gbest_x = None
        gbest_f = np.inf
        
        # Evaluate initial positions
        for i in range(n_particles):
            if calls_made >= max_pso_calls or time.time() > deadline:
                break
            f = safe_call(particles[i])
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest_x = particles[i].copy()

        # PSO Iterations
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        
        # Run PSO until budget or time is nearly exhausted
        while calls_made < max_pso_calls and time.time() < deadline:
            # Update velocities and positions
            for i in range(n_particles):
                if calls_made >= max_pso_calls or time.time() > deadline:
                    break
                r1 = rng.random()
                r2 = rng.random()
                velocities[i] = (w * velocities[i] + 
                                c1 * r1 * (pbest_x[i] - particles[i]) + 
                                c2 * r2 * (gbest_x - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], -5.0, 5.0)
                
                # Evaluate
                fx = safe_call(particles[i])
                if fx < pbest_f[i]:
                    pbest_f[i] = fx
                    pbest_x[i] = particles[i].copy()
                if fx < gbest_f:
                    gbest_f = fx
                    gbest_x = particles[i].copy()
            
            # Adapt parameters slightly to encourage convergence later
            w = max(0.4, w * 0.99)
            c1 = max(0.5, c1 * 0.995)
            c2 = min(2.0, c2 * 1.005)

        # Use the best point found by PSO as starting point for (1+1)-ES
        if gbest_x is not None:
            x_start = gbest_x.copy()
            fx_start = gbest_f
        else:
            x_start = np.zeros(dim)
            fx_start = safe_call(x_start)
        
        # --- Strategy 2: Adaptive (1+1)-ES from PSO result ---
        # This method is robust to step functions and flat regions.
        x = x_start.copy()
        fx = fx_start
        sigma = 1.0
        success_count = 0
        window = 15
        
        # Use remaining budget for ES
        remaining_budget = budget - calls_made
        for step in range(remaining_budget):
            if calls_made >= budget or time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = safe_call(y)
            
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                sigma *= 1.25  # Increase step size on success
            else:
                sigma *= 0.85  # Decrease step size on failure
            
            # Adapt sigma based on success rate in window
            if step > 0 and step % window == 0:
                if success_count / window > 0.5:
                    sigma *= 1.3
                elif success_count / window < 0.2:
                    sigma *= 0.7
                success_count = 0

            if sigma < 1e-5:
                break
            if sigma > 10.0:
                sigma = 10.0

        # --- Strategy 3: Final Coordinate Descent Refinement ---
        # Only if we have budget left and time permits
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Use a small portion of remaining budget for CD
            cd_budget = min(20, budget - calls_made)
            if cd_budget > 0:
                best_x = self._coordinate_descent(func, safe_call, best_x, best_f, budget, deadline, dim, max_calls=cd_budget)
                # Ensure best_f is updated to the result of coordinate descent
                if calls_made < budget:
                    best_f = safe_call(best_x)

        # Ensure we've made at least some calls
        if calls_made == 0:
            x0 = np.zeros(dim)
            safe_call(x0)

        return best_x, best_f

    def _coordinate_descent(self, func, safe_call, x, fx, budget, deadline, dim, max_calls=20):
        """Coordinate descent with adaptive step size for final refinement."""
        x = x.copy()
        fx = safe_call(x)
        step_size = 0.3
        calls_in_cd = 0

        for _ in range(50):  # Max iterations, but limited by max_calls
            if time.time() > deadline or calls_in_cd >= max_calls or calls_made >= budget:
                break

            improved = False
            for d in range(dim):
                for direction in [1, -1]:
                    if calls_in_cd >= max_calls or calls_made >= budget or time.time() > deadline:
                        break
                    y = x.copy()
                    y[d] += direction * step_size
                    y = np.clip(y, -5.0, 5.0)
                    fy = safe_call(y)
                    calls_in_cd += 1
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
