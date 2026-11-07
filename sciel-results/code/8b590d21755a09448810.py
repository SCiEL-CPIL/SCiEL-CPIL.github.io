"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES hybrid optimizer for Sharp Ridge (f13)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        best_x_global = None
        best_f_global = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def check_and_update(x, f):
            nonlocal best_x_global, best_f_global
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()

        def nelder_mead(x0, f0, max_calls=50):
            nonlocal calls, best_x_global, best_f_global
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(self.dim):
                step = 0.1
                x_new = x0.copy()
                x_new[i] += step
                simplex.append(x_new)
            
            # Evaluate simplex
            f_values = [f0]
            for i in range(1, self.dim + 1):
                if not time_ok() or calls >= self.budget - 5:
                    break
                fx = func(simplex[i])
                calls += 1
                f_values.append(fx)
                check_and_update(simplex[i], fx)
            
            if len(f_values) < self.dim + 1:
                return best_x_global, best_f_global

            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma = 0.5    # shrink

            # Limit iterations based on remaining budget
            # NM uses ~2-3 calls per iteration on average
            est_calls_per_iter = 3
            max_iter = min(200, max(1, (self.budget - calls - 5) // est_calls_per_iter))
            
            for _ in range(max_iter):
                if not time_ok() or calls >= self.budget - 5:
                    break

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check convergence
                if f_values[-1] - f_values[0] < 1e-10:
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                check_and_update(x_r, f_r)

                if f_values[0] <= f_r < f_values[-2]:
                    # Replace worst with reflected
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    check_and_update(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    check_and_update(x_c, f_c)

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_new = func(simplex[i])
                            calls += 1
                            f_values[i] = f_new
                            check_and_update(simplex[i], f_new)
                            if calls >= self.budget - 5 or not time_ok():
                                break

            return best_x_global, best_f_global

        # Strategy:
        # 1. Use CMA-ES to explore the landscape globally.
        #    CMA-ES adapts its covariance matrix, which is excellent for
        #    handling rotated and anisotropic landscapes like the Sharp Ridge.
        # 2. Reserve some budget for a final Nelder-Mead polish to squeeze
        #    out the last bits of precision.
        
        # CMA-ES Parameters
        # Population size rule: (4 + ln(n)) * n
        pop_size = int(np.ceil((4 + np.log(self.dim)) * self.dim))
        # Initial sigma: scale to problem size
        # Box is [-5, 5], so range is 10. Start with sigma ~ 2.5
        sigma = 2.5
        
        # We will run CMA-ES until we have used 80% of the budget.
        # Then we use the remaining 20% for NM polish.
        # Note: CMA-ES calls func internally, so we need to wrap it to count calls.
        
        # Since we can't easily hook into scipy's cma_es call count without
        # writing our own or using a wrapper, let's implement a simple
        # population-based search that mimics CMA-ES logic or just use
        # a robust multi-start NM with better initialization.
        
        # Actually, implementing a full CMA-ES from scratch is complex.
        # Let's try a different "different family": 
        # Multi-start with a specific focus on the ridge structure using
        # a "Ridge-Aligned" heuristic.
        
        # Wait, the prompt asks for a strategy MATERIALLY DIFFERENT.
        # The current one is Multi-start NM.
        # Inspiration 2 was Powell.
        # Let's try a **Global Search + Local Refinement** using **Particle Swarm Optimization (PSO)**
        # or **Simulated Annealing** for the global phase, which explores differently than NM/Powell.
        
        # Let's implement a lightweight PSO for the global phase.
        # PSO is good for finding the basin, then NM refines.
        
        def pso_global_search(num_particles, max_iters):
            nonlocal calls, best_x_global, best_f_global
            
            # Initialize particles
            particles = rng.uniform(-4, 4, (num_particles, self.dim))
            velocities = rng.uniform(-1, 1, (num_particles, self.dim))
            
            pbest_x = particles.copy()
            pbest_f = np.array([np.inf] * num_particles)
            
            gbest_x = particles[0].copy()
            gbest_f = np.inf
            
            w = 0.7  # Inertia
            c1 = 1.5  # Cognitive
            c2 = 1.5  # Social
            
            for it in range(max_iters):
                if not time_ok() or calls >= self.budget - 20: # Reserve 20 for NM
                    break
                
                for i in range(num_particles):
                    if not time_ok() or calls >= self.budget - 20:
                        break
                    
                    # Evaluate current position
                    x = particles[i]
                    f = func(x)
                    calls += 1
                    check_and_update(x, f)
                    
                    # Update personal best
                    if f < pbest_f[i]:
                        pbest_f[i] = f
                        pbest_x[i] = x.copy()
                    
                    # Update global best
                    if f < gbest_f:
                        gbest_f = f
                        gbest_x = x.copy()
                    
                    # Update velocity and position
                    r1 = rng.random(self.dim)
                    r2 = rng.random(self.dim)
                    
                    velocities[i] = w * velocities[i] + c1 * r1 * (pbest_x[i] - particles[i]) + c2 * r2 * (gbest_x - particles[i])
                    
                    # Clip velocity to prevent oscillation
                    velocities[i] = np.clip(velocities[i], -1.0, 1.0)
                    
                    particles[i] = particles[i] + velocities[i]
                    particles[i] = np.clip(particles[i], -5, 5)

            return gbest_x, gbest_f

        # Run PSO
        # Use 10 particles, ~40 iterations (400 calls)
        # This leaves ~100 calls for NM
        # But 400 calls for PSO might be too many if NM is better.
        # Let's use 5 particles, 20 iterations (100 calls) for PSO.
        # Then use remaining 400 for NM.
        
        # Actually, NM is very efficient locally. PSO is good for global.
        # Let's do 50 calls of PSO to find a good region, then NM.
        
        # PSO setup
        n_particles = 5
        # Max iters for PSO: (budget - 100) / n_particles
        # Let's reserve 100 calls for NM.
        pso_budget = max(10, self.budget - 100)
        max_pso_iters = pso_budget // n_particles
        
        gbest_x, gbest_f = pso_global_search(n_particles, max_pso_iters)
        
        # Now run NM from the best point found by PSO
        if best_x_global is not None:
            nelder_mead(best_x_global, best_f_global, max_calls=100)
            
        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
