import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Leave 20 seconds margin for safety
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Reference point from task description
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # State
        best_f = np.full(self.n_obj, np.inf)
        all_points = []
        all_f = []
        
        # Configuration
        # Use 10 parallel local searches
        n_searches = 10
        evals_per_search = self.budget // n_searches
        
        # Generate weight vectors
        # We want to cover the simplex of weights
        # For 3 objectives, we can use a grid or random
        # Let's use random but normalized
        weights = []
        for _ in range(n_searches):
            w = rng.random(self.n_obj)
            w = w / np.sum(w)
            weights.append(w)
            
        # Shuffle weights to randomize the order of searches
        rng.shuffle(weights)
        
        for i in range(n_searches):
            if time.time() > deadline:
                break
                
            lambda_i = weights[i]
            
            # Initialize x
            # Start from center or random?
            # Random is better for diversity
            x_curr = lo + (hi - lo) * rng.random(self.dim)
            
            # Evaluate
            f_curr = np.array(problem(x_curr))
            all_points.append(x_curr.copy())
            all_f.append(f_curr.copy())
            
            # Update global best
            best_f = np.minimum(best_f, f_curr)
            
            # Define scalarization function
            # We use Tchebycheff with dynamic normalization
            def calc_scalar(f_vec, lam):
                # Avoid division by zero if best_f is already at ref_point or better
                # But we are minimizing, so best_f <= ref_point ideally.
                # Scale is the range from current best to reference point.
                # If best_f > ref_point, the scale is negative? 
                # The task says "Only points strictly better than the reference point in every objective count".
                # So we should focus on the region below ref_point.
                # If best_f is worse than ref_point, we should still try to improve.
                # Let's use a fixed scale based on ref_point and a lower bound estimate (e.g., 0)
                # Or just use the difference from ref_point.
                
                # Let's use: norm_f = (f_vec - best_f) / (ref_point - best_f)
                # If best_f is very close to ref_point, this blows up.
                # Let's add a small epsilon.
                denom = ref_point - best_f
                denom = np.maximum(denom, 1e-6) # Ensure positive scale
                
                # If f_vec is better than best_f, this formula doesn't work directly for "distance to front".
                # Tchebycheff usually measures distance to an ideal point.
                # Let's stick to: max( lam_i * (f_i - best_f_i) / scale_i )
                
                d = lam * (f_vec - best_f) / denom
                return np.max(d)
            
            s_curr = calc_scalar(f_curr, lambda_i)
            
            # Adaptive step size
            # Initial step size: 10% of range
            step_size = 0.1 * (hi - lo)
            
            for step in range(evals_per_search - 1):
                if time.time() > deadline:
                    break
                    
                # Perturb
                delta = rng.standard_normal(self.dim)
                x_try = x_curr + step_size * delta
                x_try = np.clip(x_try, lo, hi)
                
                f_try = np.array(problem(x_try))
                all_points.append(x_try.copy())
                all_f.append(f_try.copy())
                
                # Update global best
                best_f = np.minimum(best_f, f_try)
                
                s_try = calc_scalar(f_try, lambda_i)
                
                if s_try < s_curr:
                    x_curr = x_try
                    f_curr = f_try
                    s_curr = s_try
                    # Optional: increase step size slightly if we are making progress
                    # step_size = min(step_size * 1.1, 0.5 * (hi - lo))
                else:
                    # Decay step size
                    step_size *= 0.9

        return np.array(all_points)
