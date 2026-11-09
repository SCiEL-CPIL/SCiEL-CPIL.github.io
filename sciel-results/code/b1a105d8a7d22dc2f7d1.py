"""Portfolio-based continuous optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer combining:
    1. PSO for global exploration
    2. Adaptive (1+1)-ES with multiple restarts for local refinement
    
    Uses time guard to stay within wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s safety margin
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
        
        # --- Strategy 1: PSO for global exploration ---
        # Use PSO to find a good starting point
        n_particles = 20
        n_iterations_pso = 30
        
        # Initialize particles
        particles = rng.uniform(-4, 4, (n_particles, dim))
        velocities = rng.uniform(-1, 1, (n_particles, dim))
        pbest_x = particles.copy()
        pbest_f = np.full(n_particles, np.inf)
        gbest_x = None
        gbest_f = np.inf
        
        for it in range(n_iterations_pso):
            if calls_made >= budget or time.time() > deadline:
                break
                
            # Evaluate particles
            for i in range(n_particles):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = particles[i]
                fx = safe_call(x)
                if fx < pbest_f[i]:
                    pbest_f[i] = fx
                    pbest_x[i] = x.copy()
                if fx < gbest_f:
                    gbest_f = fx
                    gbest_x = x.copy()
            
            if gbest_x is None:
                break
                
            # Update velocities and positions
            w = 0.7  # inertia
            c1 = 1.5  # cognitive
            c2 = 1.5  # social
            
            for i in range(n_particles):
                r1 = rng.random()
                r2 = rng.random()
                velocities[i] = (w * velocities[i] + 
                                c1 * r1 * (pbest_x[i] - particles[i]) + 
                                c2 * r2 * (gbest_x - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], -5.0, 5.0)
        
        # Use the best point found by PSO as starting point for (1+1)-ES
        if gbest_x is not None:
            x_start = gbest_x.copy()
            fx_start = gbest_f
        else:
            x_start = np.zeros(dim)
            fx_start = safe_call(x_start)
        
        # --- Strategy 2: Adaptive (1+1)-ES from PSO result ---
        x = x_start.copy()
        fx = fx_start
        sigma = 1.0
        success = 0
        window = 15
        
        for step in range(budget - calls_made):
            if calls_made >= budget or time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = safe_call(y)
            
            if fy < fx:
                x, fx = y, fy
                success += 1
                sigma *= 1.25
            else:
                sigma *= 0.85
            
            # Adapt sigma based on success rate
            if step > 0 and step % window == 0:
                success = 0
                if sigma < 0.01:
                    sigma = 0.05
                elif sigma > 5.0:
                    sigma = 3.0
            
            if sigma < 1e-4:
                break
        
        # --- Strategy 3: Additional restarts from random points if budget remains ---
        n_additional_starts = 5
        for s in range(n_additional_starts):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Random starting point
            x = rng.uniform(-4, 4, dim)
            fx = safe_call(x)
            if not np.isfinite(fx):
                continue
            
            # Short (1+1)-ES run
            sigma = 1.0
            for step in range(min(50, budget - calls_made)):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                else:
                    sigma *= 0.85
                
                if sigma < 1e-4:
                    break
        
        # --- Strategy 4: Final local refinement around best point ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            best_x = self._coordinate_descent(func, safe_call, best_x, best_f, budget, deadline, dim)
            best_f = safe_call(best_x)

        # Ensure we've made at least some calls
        if calls_made == 0:
            x0 = np.zeros(dim)
            safe_call(x0)

        return best_x, best_f

    def _coordinate_descent(self, func, safe_call, x, fx, budget, deadline, dim):
        """Simple coordinate descent to refine a point."""
        x = x.copy()
        fx = safe_call(x)
        step_size = 0.5
        
        for _ in range(100):  # Max iterations
            if time.time() > deadline:
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
                step_size *= 0.5
                if step_size < 1e-4:
                    break
        
        return x
# EVOLVE-BLOCK-END
