"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Optimizer for BBOB f15 (Rotated Rastrigin).
    
    Strategy:
    1. Particle Swarm Optimization (PSO) for robust global exploration.
       Uses a small swarm with inertia decay to navigate the multimodal landscape.
       Includes strategic initialization (origin, symmetric points) to quickly
       find a promising basin.
    2. Intensive (1+1)-ES refinement with adaptive step size.
       Focuses remaining budget on polishing the best candidate found.
       Includes periodic local coordinate refinement to handle Rastrigin's
       periodic structure and escape shallow local minima.
    
    This combines the population-based exploration strength of PSO
    with the precise exploitation of (1+1)-ES and local search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin for 300s limit
        
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        # --- Local refinement: coordinate descent with Rastrigin snap ---
        def local_refine(x, fx, max_rounds=5):
            """Coordinate descent with Rastrigin-periodic candidates."""
            nonlocal calls, best_x, best_f
            if not budget_ok() or not time_ok():
                return x, fx

            for _ in range(max_rounds):
                if not budget_ok() or not time_ok():
                    break
                improved = False
                for axis in range(dim):
                    if not budget_ok() or not time_ok():
                        break
                    x0 = x[axis]
                    # Generate candidates: current, nearby, and Rastrigin-aligned
                    candidates = [x0]
                    # Small perturbations
                    for d in [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]:
                        candidates.append(x0 + d)
                    # Rastrigin: minima at multiples of 2*pi in the unrotated frame.
                    # Note: The function is rotated, but the periodicity remains.
                    # We try snapping to nearby multiples of 2*pi.
                    k = np.round(x0 / (2.0 * np.pi))
                    for kk in [k - 1, k, k + 1]:
                        candidates.append(kk * 2.0 * np.pi)
                    # Also try halfway points (maxima of cos, so avoid)
                    for kk in [k, k + 1]:
                        candidates.append(kk * 2.0 * np.pi + np.pi)

                    best_cand = x0
                    best_cand_f = fx
                    for c in candidates:
                        if not budget_ok():
                            break
                        c = float(np.clip(c, -5.0, 5.0))
                        if abs(c - x0) < 1e-12:
                            continue
                        y = x.copy()
                        y[axis] = c
                        fy = eval_func(y)
                        if fy < best_cand_f:
                            best_cand_f = fy
                            best_cand = c
                    if best_cand != x0:
                        x[axis] = best_cand
                        fx = best_cand_f
                        improved = True
                if not improved:
                    break
            return x, fx

        # --- Phase 1: Particle Swarm Optimization ---
        # PSO with a small swarm for efficient exploration of the 5D space.
        
        n_particles = 12
        # Allocate ~70% of budget to PSO, rest to (1+1)-ES refinement
        # We want enough budget for PSO to actually find a good basin
        pso_budget = int(budget * 0.70)
        
        # Initialize particles
        # Use a mix of deterministic and random initialization
        particles = np.zeros((n_particles, dim))
        # First particle at origin (likely near global min for many rotations)
        particles[0] = np.zeros(dim)
        # Some at symmetric points
        particles[1] = np.full(dim, 2.0)
        particles[2] = np.full(dim, -2.0)
        particles[3] = np.array([2.0, -2.0, 2.0, -2.0, 2.0])[:dim]
        particles[4] = np.array([-2.0, 2.0, -2.0, 2.0, -2.0])[:dim]
        particles[5] = np.array([0.0, 2.0, -2.0, 0.0, 2.0])[:dim]
        particles[6] = np.array([2.0, 0.0, -2.0, 2.0, 0.0])[:dim]
        # Rest random
        for i in range(7, n_particles):
            particles[i] = rng.uniform(-3.0, 3.0, dim)
        
        # Evaluate initial positions
        p_fits = np.zeros(n_particles)
        for i in range(n_particles):
            if calls >= pso_budget or time.time() > deadline:
                break
            p_fits[i] = eval_func(particles[i])
        
        # Initialize velocities
        velocities = np.zeros((n_particles, dim))
        # Small initial velocities
        for i in range(n_particles):
            velocities[i] = rng.uniform(-0.5, 0.5, dim)
        
        # Personal bests
        p_best = particles.copy()
        p_best_f = p_fits.copy()
        
        # Global best
        g_best_idx = np.argmin(p_best_f)
        g_best = p_best[g_best_idx].copy()
        g_best_f = p_best_f[g_best_idx]
        
        # PSO parameters
        # Inertia weight: decays from 0.9 to 0.4
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        
        n_pso_calls = pso_budget - calls
        n_iterations = max(1, n_pso_calls // n_particles)
        
        # Stagnation counter
        stagnation_counter = 0
        max_stagnation = 50 # If no improvement for 50 iterations, break PSO early
        
        for it in range(n_iterations):
            if calls >= pso_budget or time.time() > deadline:
                break
            
            # Inertia weight decay
            w = 0.9 - 0.5 * (it / max(1, n_iterations))
            
            improved_this_iter = False
            
            for i in range(n_particles):
                if calls >= pso_budget or time.time() > deadline:
                    break
                
                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (p_best[i] - particles[i]) + 
                               c2 * r2 * (g_best - particles[i]))
                
                # Clip velocity to prevent overshooting
                velocities[i] = np.clip(velocities[i], -1.5, 1.5)
                
                # Update position
                new_pos = particles[i] + velocities[i]
                new_pos = np.clip(new_pos, -5.0, 5.0)
                
                # Evaluate
                f_new = eval_func(new_pos)
                
                # Update personal best
                if f_new < p_best_f[i]:
                    p_best[i] = new_pos.copy()
                    p_best_f[i] = f_new
                    improved_this_iter = True
                    
                    # Update global best
                    if f_new < g_best_f:
                        g_best = new_pos.copy()
                        g_best_f = f_new
                
                particles[i] = new_pos
            
            if improved_this_iter:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter > max_stagnation:
                break

        # --- Phase 2: Adaptive (1+1)-ES Refinement ---
        # Refine the best point found by PSO using (1+1)-ES.
        # This phase is critical for exploiting the local structure of Rastrigin.
        
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Start with a moderate sigma for local search
            # Rastrigin has local minima at integers, so we start somewhat small
            # but not too small to escape shallow local minima if we are not at the global one yet.
            sigma = 0.5 
            window = 10
            recent = [False] * window
            idx = 0
            stagnation = 0
            max_stagnation_es = 100
            
            # Run until budget or time is up
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                    stagnation = 0
                else:
                    recent[idx % window] = False
                    stagnation += 1
                idx += 1
                
                # Adapt sigma based on success rate
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.2
                    elif sr < 0.2:
                        sigma *= 0.8
                    else:
                        sigma *= 1.0
                sigma = np.clip(sigma, 0.01, 2.0)
                
                # If we are stuck for a long time, try a larger jump to escape potential local minima
                # This helps if we converged to a local minimum far from the global one
                if stagnation > max_stagnation_es:
                    # Random restart around current best with larger step
                    y = np.clip(x + 1.0 * rng.standard_normal(dim), -5.0, 5.0)
                    fy = eval_func(y)
                    if fy < fx:
                        x, fx = y, fy
                    stagnation = 0
                    max_stagnation_es = 50 # Reset threshold

                # Periodic local refinement to exploit Rastrigin structure
                # Do this every 50 iterations of ES or when sigma is small
                if idx % 50 == 0 and sigma < 1.0:
                    x_ref, f_ref = local_refine(x.copy(), fx, max_rounds=3)
                    if f_ref < fx:
                        x = x_ref
                        fx = f_ref
                        # Reset stagnation since we improved
                        stagnation = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
