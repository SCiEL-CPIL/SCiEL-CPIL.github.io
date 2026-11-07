"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derandomized Discrete Particle Swarm Optimization (ddPSO) with ring topology."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # PSO Parameters
        n_particles = 20
        # We have 500 calls. 
        # Initialization: 20 calls.
        # Remaining: 480 calls.
        # 480 / 20 particles = 24 iterations.
        # This is a tight schedule, but PSO is very sample efficient in low dimensions.
        
        # If time allows, we can do more iterations by reducing particles or using a mixed strategy.
        # However, 20 particles is a good balance for dim=5.
        
        # Initialize swarm
        # Spread particles using a quasi-random sequence (Sobol-like via rng) for better coverage
        # Standard random is okay, but let's try to ensure some spread.
        # Actually, simple uniform is standard for PSO.
        
        positions = rng.uniform(-5, 5, (n_particles, dim))
        velocities = np.zeros((n_particles, dim))
        
        # Personal bests
        pbest_pos = positions.copy()
        pbest_val = np.full(n_particles, np.inf)
        
        # Global best
        gbest_pos = None
        gbest_val = np.inf
        
        used_calls = 0
        
        # Initial Evaluation
        for i in range(n_particles):
            if time.time() > deadline:
                break
            xi = positions[i]
            fi = func(xi)
            used_calls += 1
            pbest_val[i] = fi
            if fi < gbest_val:
                gbest_val = fi
                gbest_pos = xi.copy()
        
        if gbest_pos is None:
            # Fallback if no calls were made (should not happen)
            x = rng.uniform(-5, 5, dim)
            return x, func(x)

        # PSO Loop
        # We will run iterations until budget is exhausted or time runs out.
        # Inertia weight decays linearly from 0.9 to 0.4
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        v_max = 1.0  # Fixed maximum velocity bound as per directive
        
        # Estimate number of iterations we can afford
        # We have budget - used_calls remaining.
        # Each iteration costs n_particles calls.
        # Let's just loop and check budget/time.
        
        iter_count = 0
        max_iters = (budget - used_calls) // n_particles + 10 # Buffer for time checks
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if used_calls >= budget:
                break
            
            # Update inertia weight
            # Linear decay based on iteration count relative to expected max
            # If we don't know max_iters exactly, we can use a fixed schedule or
            # decay based on 'it' assuming we use most of the budget.
            # Let's assume we use ~24 iterations.
            # To be safe, let's decay over the first 50 iterations if we run that long,
            # or just use a fixed small decay per step.
            # Directive says: "decreases linearly from 0.9 to 0.4 over the 500-call budget"
            # 500 calls / 20 particles = 25 iterations.
            # So decay over 25 iterations.
            progress = min(1.0, it / 25.0)
            w = w_start + (w_end - w_start) * progress
            
            # Ring topology: Each particle sees 2 neighbors (left and right)
            # For particle i, neighbors are (i-1)%n and (i+1)%n
            # We need the best among pbest of neighbors and self?
            # Standard ddPSO often uses the global best or a local best.
            # "Ring topology" usually implies a local best (lbest) from neighbors.
            # Let's use the best of the pbests of the neighbors and self.
            
            for i in range(n_particles):
                if used_calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Find local best (lbest) in ring neighborhood
                # Neighbors: i-1, i, i+1
                idxs = [(i - 1) % n_particles, i, (i + 1) % n_particles]
                lbest_val = min(pbest_val[idxs])
                # Find which one it is
                for idx in idxs:
                    if pbest_val[idx] == lbest_val:
                        lbest_pos = pbest_pos[idx]
                        break
                
                # Derandomized PSO:
                # Instead of random r1, r2, we use fixed or deterministic values?
                # "Derandomized" usually means replacing random coefficients with
                # deterministic ones, often 0.5 or based on time.
                # However, some randomness is often kept in the velocity update 
                # or the position update is clipped.
                # Let's use r1 = r2 = 0.5 for "derandomized" in the coefficient sense,
                # or perhaps alternate? 
                # A common derandomized approach uses r1 = r2 = 0.5.
                r1 = 0.5
                r2 = 0.5
                
                # Velocity update
                # v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (lbest - x)
                v_new = w * velocities[i] + c1 * r1 * (pbest_pos[i] - positions[i]) + c2 * r2 * (lbest_pos - positions[i])
                
                # Clip velocity to [-v_max, v_max]
                v_new = np.clip(v_new, -v_max, v_max)
                
                # Position update
                x_new = positions[i] + v_new
                
                # Clip position to box [-5, 5]
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate
                f_new = func(x_new)
                used_calls += 1
                
                # Update personal best
                if f_new < pbest_val[i]:
                    pbest_val[i] = f_new
                    pbest_pos[i] = x_new.copy()
                    # Update global best
                    if f_new < gbest_val:
                        gbest_val = f_new
                        gbest_pos = x_new.copy()
                
                # Update state for next iteration
                positions[i] = x_new
                velocities[i] = v_new
            
            iter_count += 1
        
        return gbest_pos, gbest_val
# EVOLVE-BLOCK-END
