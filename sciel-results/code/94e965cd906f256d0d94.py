import numpy as np
import time
from itertools import product

class Optimizer:
    """
    Deterministic Grid + Fixed-Step Hill Climb for BBOB f21.
    Phase 1: Evaluate all 32 hypercube vertices + origin.
    Phase 2: Fixed-step coordinate-wise hill climb from best vertex.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        def time_left():
            return deadline - time.time()

        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        budget = self.budget

        # Phase 1: Evaluate all 2^dim vertices of [-5, 5]^dim plus the origin
        # This provides deterministic global coverage with 2^5 + 1 = 33 points
        vertices = []
        for bits in product([0, 1], repeat=dim):
            v = np.array([5.0 if b == 1 else -5.0 for b in bits])
            vertices.append(v)
        vertices.append(np.zeros(dim))  # origin

        for v in vertices:
            if calls_made >= budget or time_left() < 5:
                break
            evaluate(v)

        if best_x is None:
            return best_x, best_f

        # Phase 2: Fixed-step coordinate-wise hill climb
        # Start from the best vertex found
        current_x = best_x.copy()
        current_f = best_f
        
        # Use a fixed step size for coordinate-wise moves
        # Start with a moderate step and shrink if no improvement
        step_size = 1.0
        no_improve_count = 0
        max_no_improve = 5

        while calls_made < budget and time_left() > 2:
            improved = False
            
            # Try moving in each coordinate direction (positive and negative)
            for d in range(dim):
                if calls_made >= budget or time_left() < 2:
                    break
                
                for direction in [1, -1]:
                    if calls_made >= budget or time_left() < 2:
                        break
                    
                    x_try = current_x.copy()
                    x_try[d] += direction * step_size
                    x_try = np.clip(x_try, -5.0, 5.0)
                    
                    f_try = evaluate(x_try)
                    
                    if f_try < current_f:
                        current_x = x_try.copy()
                        current_f = f_try
                        improved = True
                        no_improve_count = 0
                    else:
                        # Try half step if full step didn't work
                        x_try_half = current_x.copy()
                        x_try_half[d] += direction * step_size * 0.5
                        x_try_half = np.clip(x_try_half, -5.0, 5.0)
                        
                        if calls_made < budget and time_left() > 2:
                            f_try_half = evaluate(x_try_half)
                            if f_try_half < current_f:
                                current_x = x_try_half.copy()
                                current_f = f_try_half
                                improved = True
                                no_improve_count = 0

            if not improved:
                no_improve_count += 1
                # Shrink step size if no improvement
                step_size *= 0.5
                if step_size < 1e-4 or no_improve_count > max_no_improve:
                    break
            else:
                # If we improved, keep step size or slightly increase
                step_size = min(step_size * 1.2, 2.0)

        return best_x, best_f
