import numpy as np
import time

class Optimizer:
    """
    Stochastic global search with simulated annealing exploiting Rastrigin's periodicity.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Simulated annealing with periodic-aware moves
        # Rastrigin's minima are at integer coordinates (shifted). 
        # Use a cooling schedule to explore the landscape.
        
        # Initial point: random
        current_x = rng.uniform(-5, 5, size=dim)
        current_f = try_point(current_x)
        if current_f is None:
            return best_x, best_f
        
        # Simulated annealing parameters
        T_start = 10.0
        T_end = 0.01
        cooling_rate = 0.995  # Geometric cooling
        n_iterations = min(300, budget - calls_made)  # Leave budget for refinement
        
        T = T_start
        for it in range(n_iterations):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Generate new point: mix of local perturbation and periodic jump
            if rng.random() < 0.3:
                # Periodic jump: move to a nearby integer lattice point
                # This helps escape local minima by jumping between basins
                jump_dim = rng.integers(dim)
                current_x[jump_dim] = np.round(current_x[jump_dim]) + rng.choice([-1, 0, 1])
                current_x[jump_dim] = np.clip(current_x[jump_dim], -5, 5)
            else:
                # Local perturbation: Gaussian noise
                step_size = T * 0.5
                current_x = current_x + rng.normal(0, step_size, size=dim)
                current_x = np.clip(current_x, -5, 5)
            
            new_f = try_point(current_x)
            if new_f is None:
                break
            
            # Accept/reject based on simulated annealing criterion
            if new_f < current_f:
                current_f = new_f
            else:
                delta = new_f - current_f
                if delta > 0 and rng.random() < np.exp(-delta / T):
                    current_f = new_f
                else:
                    # Reject: revert
                    current_x = best_x if best_x is not None else current_x
            
            # Cool down
            T = max(T_end, T * cooling_rate)
        
        # Phase 2: Local refinement around the best point found
        if best_x is not None and calls_made < budget:
            # Coordinate-wise local search with decreasing step sizes
            step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
            for step in step_sizes:
                if calls_made >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and calls_made < budget and time.time() <= deadline:
                    improved = False
                    for i in range(dim):
                        for sign in [1, -1]:
                            if calls_made >= budget or time.time() > deadline:
                                break
                            x_new = best_x.copy()
                            x_new[i] += sign * step
                            x_new = np.clip(x_new, -5, 5)
                            f_new = try_point(x_new)
                            if f_new is not None and f_new < best_f:
                                improved = True
        
        # Phase 3: Random restarts if budget remains
        restarts = min(20, max(0, (budget - calls_made) // 10))
        for _ in range(restarts):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f = try_point(x)
            if f is not None and f < best_f:
                # Quick local refinement from this new point
                for step in [0.25, 0.125]:
                    if calls_made >= budget or time.time() > deadline:
                        break
                    for i in range(dim):
                        for sign in [1, -1]:
                            if calls_made >= budget or time.time() > deadline:
                                break
                            x_new = best_x.copy()
                            x_new[i] += sign * step
                            x_new = np.clip(x_new, -5, 5)
                            try_point(x_new)
        
        return best_x, best_f
