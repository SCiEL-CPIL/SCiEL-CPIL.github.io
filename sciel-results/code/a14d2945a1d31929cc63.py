"""Optimizer for BBOB-style continuous functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive grid-refinement with local descent."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Track calls
        call_count = [0]

        def safe_func(x):
            if call_count[0] >= budget:
                return float('inf')
            call_count[0] += 1
            try:
                return float(func(x))
            except Exception:
                return float('inf')

        # Phase 1: Coarse grid search to find promising region
        # Evaluate a structured set of points in [-5, 5]^dim
        # Use a small number of points per dimension for coarse search
        
        # First, evaluate center and corners
        x_center = np.zeros(dim)
        f_center = safe_func(x_center)
        
        best_x = x_center.copy()
        best_f = f_center

        # Evaluate corners of the box
        for i in range(2**dim):
            if call_count[0] >= budget or time.time() > deadline:
                break
            corner = np.array([(i >> d) & 1 for d in range(dim)], dtype=float)
            corner = corner * 10 - 5  # map {0,1} to {-5, 5}
            f_corner = safe_func(corner)
            if f_corner < best_f:
                best_f = f_corner
                best_x = corner.copy()

        # Phase 2: Adaptive refinement around the best point found
        # Use a local search with decreasing step sizes
        current_x = best_x.copy()
        current_f = best_f
        step_size = 2.0

        # Adaptive local search
        while call_count[0] < budget and time.time() < deadline:
            improved = False
            
            # Try moving in each dimension
            for d in range(dim):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                
                # Try positive and negative steps
                for direction in [1, -1]:
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    
                    trial_x = current_x.copy()
                    trial_x[d] += direction * step_size
                    trial_x = np.clip(trial_x, -5, 5)
                    
                    trial_f = safe_func(trial_x)
                    
                    if trial_f < current_f:
                        current_x = trial_x
                        current_f = trial_f
                        improved = True
                        # If we improved, continue trying this direction
                        if call_count[0] < budget and time.time() < deadline:
                            # Try to keep going in this direction
                            for _ in range(3):  # up to 3 more steps in same direction
                                if call_count[0] >= budget or time.time() > deadline:
                                    break
                                trial_x2 = current_x.copy()
                                trial_x2[d] += direction * step_size
                                trial_x2 = np.clip(trial_x2, -5, 5)
                                trial_f2 = safe_func(trial_x2)
                                if trial_f2 < current_f:
                                    current_x = trial_x2
                                    current_f = trial_f2
                                else:
                                    break

            if improved:
                # Keep step size or slightly increase
                step_size = min(step_size * 1.1, 5.0)
            else:
                # No improvement, reduce step size
                step_size *= 0.7
                if step_size < 0.01:
                    break

        # Phase 3: Fine-tuning with very small steps
        if call_count[0] < budget and time.time() < deadline:
            fine_step = 0.01
            for _ in range(min(50, budget - call_count[0])):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        if call_count[0] >= budget or time.time() > deadline:
                            break
                        trial_x = current_x.copy()
                        trial_x[d] += direction * fine_step
                        trial_x = np.clip(trial_x, -5, 5)
                        trial_f = safe_func(trial_x)
                        if trial_f < current_f:
                            current_x = trial_x
                            current_f = trial_f
                            improved = True
                if not improved:
                    break

        return current_x, current_f
# EVOLVE-BLOCK-END
