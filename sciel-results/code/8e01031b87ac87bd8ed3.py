"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """PSO with Rastrigin-aware jumps for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        rng = np.random.default_rng(12345)
        
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        dim = self.dim
        n_particles = 15
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        
        # Initialize particles
        positions = np.random.uniform(-5, 5, size=(n_particles, dim))
        velocities = np.random.uniform(-2, 2, size=(n_particles, dim))
        pbest_positions = positions.copy()
        pbest_fits = np.full(n_particles, float('inf'))
        
        # Evaluate initial positions
        for i in range(n_particles):
            f = counted_func(positions[i])
            pbest_fits[i] = f
            pbest_positions[i] = positions[i].copy()
            if f < best_f:
                best_f = f
                best_x = positions[i].copy()
        
        gbest_position = best_x.copy()
        gbest_fit = best_f
        
        # PSO main loop
        max_iter = (max_calls - call_count) // n_particles
        stagnation_counter = 0
        stagnation_threshold = 5  # restart if no improvement for 5 iterations
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if call_count >= max_calls:
                break
            
            # Check for stagnation
            if gbest_fit < best_f - 1e-10:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # If stagnating, add Rastrigin-aware jumps to some particles
            if stagnation_counter >= stagnation_threshold:
                # Reset some particles with jumps aligned to Rastrigin periodicity
                n_reset = n_particles // 3
                for i in range(n_reset):
                    # Jump by multiples of 2*pi/sqrt(5) ≈ 2.808 in random dimensions
                    jump_dims = rng.choice(dim, size=dim, replace=False)
                    jump_mags = rng.choice([-2.808, 2.808], size=dim)
                    positions[i] = positions[i] + jump_mags * rng.choice([0, 1], size=dim)
                    positions[i] = np.clip(positions[i], -5, 5)
                    velocities[i] = np.random.uniform(-1, 1, size=dim)
                    pbest_positions[i] = positions[i].copy()
                    pbest_fits[i] = float('inf')
                stagnation_counter = 0
            
            # Update velocities and positions
            for i in range(n_particles):
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (pbest_positions[i] - positions[i]) + 
                               c2 * r2 * (gbest_position - positions[i]))
                
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -3, 3)
                
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)
            
            # Evaluate new positions
            for i in range(n_particles):
                f = counted_func(positions[i])
                
                # Update particle best
                if f < pbest_fits[i]:
                    pbest_fits[i] = f
                    pbest_positions[i] = positions[i].copy()
                
                # Update global best
                if f < gbest_fit:
                    gbest_fit = f
                    gbest_position = positions[i].copy()
                    best_f = f
                    best_x = positions[i].copy()
        
        # Final local refinement using coordinate descent
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 10:
                x = best_x.copy()
                f = best_f
                step = 0.1
                for _ in range(min(remaining, 50)):
                    if time.time() > deadline or call_count >= max_calls:
                        break
                    improved = False
                    for d in range(dim):
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            f_new = counted_func(x_new)
                            if f_new < f - 1e-10:
                                x = x_new
                                f = f_new
                                improved = True
                    if not improved:
                        step *= 0.5
                        if step < 0.01:
                            break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
