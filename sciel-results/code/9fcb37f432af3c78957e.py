import numpy as np
import time

class Optimizer:
    """
    Multi-start Hooke-Jeeves with adaptive step size for BBOB f12 (Bent Cigar).
    
    Strategy:
    1. Multi-start initialization: Evaluate several random points to find promising regions.
    2. Hooke-Jeeves pattern search: Coordinate-wise local refinement with adaptive step sizes.
       - Step size doubles on improvement, halves on failure.
       - If a full cycle yields no improvement, all step sizes are halved.
       - If all step sizes drop below a threshold, the search terminates.
    3. Budget management: Strict tracking of evaluations and time to ensure validity.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            
            x = np.clip(np.asarray(x, dtype=float).flatten(), -5, 5)
            
            try:
                f = float(func(x))
            except Exception:
                return float('inf')
                
            if not np.isfinite(f):
                return float('inf')

            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        def hooke_jees(start_x, initial_f=None):
            """
            Hooke-Jeeves pattern search with adaptive step sizes.
            """
            nonlocal eval_count
            if eval_count >= budget - 10:
                return

            x = np.clip(start_x, -5, 5).copy()
            if initial_f is not None:
                f = initial_f
            else:
                f = eval_func(x)
                if eval_count >= budget:
                    return

            # Initialize step sizes. 
            # For Bent Cigar, the valley is narrow. A moderate initial step is good.
            step_sizes = np.ones(d) * 0.5
            min_step = 1e-5
            max_iter = 500  # Max iterations of the outer loop (full coordinate cycles)
            
            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                moved = False
                for i in range(d):
                    if eval_count >= budget or time.time() > deadline:
                        break

                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    f_plus = eval_func(x_plus)
                    if eval_count >= budget:
                        break

                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        step_sizes[i] *= 2.0
                        moved = True
                        # Continue exploring this direction? 
                        # Standard HJ often does a "pattern move" or just proceeds.
                        # Proceeding to next coordinate is simpler and robust.
                    else:
                        step_sizes[i] *= 0.5

                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= step_sizes[i]
                    f_minus = eval_func(x_minus)
                    if eval_count >= budget:
                        break

                    if f_minus < f:
                        x = x_minus
                        f = f_minus
                        step_sizes[i] *= 2.0
                        moved = True
                    else:
                        step_sizes[i] *= 0.5

                if np.all(step_sizes < min_step):
                    break
                
                if not moved:
                    # If no move was made in a full cycle, we might be stuck or converged.
                    # Shrink all step sizes to try to escape local minima or confirm convergence.
                    step_sizes *= 0.5
                    if np.all(step_sizes < min_step):
                        break
                    # If we've shrunk too much without moving, break to save budget
                    if np.all(step_sizes < 1e-4):
                        break

        # --- Phase 1: Multi-Start Initialization ---
        rng = np.random.default_rng(42)
        
        # Generate initial candidates
        # Include origin and random points in the likely range of the minimizer [-4, 4]
        init_points = [np.zeros(d)]
        for _ in range(9):
            init_points.append(rng.uniform(-4, 4, d))
            
        # Evaluate all initial points
        point_scores = []
        for p in init_points:
            if eval_count >= budget:
                break
            fp = eval_func(p)
            point_scores.append((fp, p))
            
        # Sort by function value
        point_scores.sort(key=lambda x: x[0])
        
        # --- Phase 2: Local Refinement on the best starting points ---
        # Refine the top K points. 
        # K=3 is a good balance between exploration and exploitation.
        top_k = min(3, len(point_scores))
        
        for i in range(top_k):
            if eval_count >= budget - 20:
                break
            
            # Refine the i-th best point
            # Pass the known function value to save one evaluation
            hooke_jees(point_scores[i][1], point_scores[i][0])
            
            # Update best if HJ found a better point
            # (eval_func already updates best_x, best_f)
            
        return best_x, best_f
