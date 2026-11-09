"""Rewritten optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 295.0
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            if self.calls > self.budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # ---- Phase 1: Coarse random sampling to find promising region ----
        # Use a modest number of random samples to get a rough sense of the landscape
        n_init = min(30, budget // 10)
        init_points = np.zeros((n_init, dim))
        init_points[0] = np.zeros(dim)
        init_points[1] = rng.uniform(-1, 1, dim)
        init_points[2] = rng.uniform(-2, 2, dim)
        init_points[3] = rng.uniform(-3, 3, dim)
        init_points[4] = rng.uniform(-4, 4, dim)
        init_points[5:] = rng.uniform(-5, 5, dim - 5)
        init_points[5:] = rng.uniform(-5, 5, dim)

        init_f = np.zeros(n_init)
        for i in range(n_init):
            if self.calls >= budget or time.time() > self.deadline:
                break
            init_f[i] = wrapped_func(init_points[i])

        valid_n = self.calls
        if valid_n < n_init:
            n_init = valid_n
            init_points = init_points[:n_init]
            init_f = init_f[:n_init]

        if n_init == 0:
            self.best_x = np.zeros(dim)
            self.best_f = func(self.best_x)
            self.calls = 1
            return self.best_x, self.best_f

        # Find the best initial point
        best_idx = np.argmin(init_f)
        best_init_x = init_points[best_idx].copy()
        best_init_f = init_f[best_idx]

        # ---- Phase 2: Coordinate descent / axis-wise local search ----
        # For each axis, do a fine-grained 1D search around the current best
        # This exploits the fact that Bent Cigar has one very narrow axis
        # and the others are relatively flat.
        
        remaining_budget = budget - self.calls
        if remaining_budget <= 0:
            return self.best_x, self.best_f

        # Adaptive step size for coordinate search
        # Start with moderate step sizes and refine
        step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
        
        current_x = best_init_x.copy()
        current_f = best_init_f

        # For each step size, try moving along each axis
        for step_size in step_sizes:
            if self.calls >= budget or time.time() > self.deadline:
                break
            
            improved = False
            for axis in range(dim):
                if self.calls >= budget or time.time() > self.deadline:
                    break
                
                # Try positive and negative directions
                for direction in [1, -1]:
                    if self.calls >= budget or time.time() > self.deadline:
                        break
                    
                    test_x = current_x.copy()
                    test_x[axis] += direction * step_size
                    test_x = np.clip(test_x, -5, 5)
                    
                    test_f = wrapped_func(test_x)
                    
                    if test_f < current_f:
                        current_x = test_x.copy()
                        current_f = test_f
                        improved = True
                    else:
                        # Try half step in the same direction
                        half_test_x = current_x.copy()
                        half_test_x[axis] += direction * step_size * 0.5
                        half_test_x = np.clip(half_test_x, -5, 5)
                        
                        half_test_f = wrapped_func(half_test_x)
                        
                        if half_test_f < current_f:
                            current_x = half_test_x.copy()
                            current_f = half_test_f
                            improved = True

            # If no improvement in this step size, stop refining
            if not improved and step_size < 0.1:
                break

        # ---- Phase 3: Fine-tuning with smaller steps ----
        if self.calls < budget:
            fine_steps = [0.0005, 0.0002, 0.0001, 0.00005]
            for step_size in fine_steps:
                if self.calls >= budget or time.time() > self.deadline:
                    break
                
                improved = False
                for axis in range(dim):
                    if self.calls >= budget or time.time() > self.deadline:
                        break
                    
                    for direction in [1, -1]:
                        if self.calls >= budget or time.time() > self.deadline:
                            break
                        
                        test_x = current_x.copy()
                        test_x[axis] += direction * step_size
                        test_x = np.clip(test_x, -5, 5)
                        
                        test_f = wrapped_func(test_x)
                        
                        if test_f < current_f:
                            current_x = test_x.copy()
                            current_f = test_f
                            improved = True

                if not improved:
                    break

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
