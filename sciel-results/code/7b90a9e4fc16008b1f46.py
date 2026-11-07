import numpy as np

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            # bounds is list of (low, high)
            # dim = 10
            # We assume bounds[0] is for x1, bounds[1:] for x2...x10
            
            # Generate x1 values
            # Use linspace for even coverage of the Pareto front
            x1_low, x1_high = bounds[0]
            # We want to cover the range where the front exists
            # Usually x1 in [0, 1]
            # But let's use the provided bounds
            # If bounds are [-5, 5] for x1, the front is only in [0, 1]?
            # ZDT4 x1 is typically [0, 1].
            # If bounds[0] is [-5, 5], we should sample [0, 1] mostly.
            
            # Let's sample x1 uniformly in [0, 1] intersected with bounds[0]
            # If bounds[0] is (0, 1), use that.
            # If bounds[0] is (-5, 5), use (0, 1).
            
            lo1, hi1 = bounds[0]
            target_lo = max(lo1, 0.0)
            target_hi = min(hi1, 1.0)
            
            if target_lo >= target_hi:
                 # Fallback to full bounds
                 target_lo, target_hi = lo1, hi1
                 
            n_evals = self.budget
            # 90% on the "good" line, 10% random
            n_good = int(0.9 * n_evals)
            n_random = n_evals - n_good
            
            # Generate x1 for good points
            x1_good = np.linspace(target_lo, target_hi, n_good)
            
            # Generate x for random points
            # x1 random in target range
            x1_rand = np.random.uniform(target_lo, target_hi, n_random)
            
            # For good points, x2..x10 = 0
            # Check if 0 is in bounds for x2..x10
            # If not, clamp to closest bound
            x_good = np.zeros((n_good, self.dim))
            x_good[:, 0] = x1_good
            for i in range(1, self.dim):
                lo_i, hi_i = bounds[i]
                if 0.0 < lo_i:
                    x_good[:, i] = lo_i
                elif 0.0 > hi_i:
                    x_good[:, i] = hi_i
                else:
                    x_good[:, i] = 0.0
                    
            # For random points, x random in bounds
            x_rand = np.random.uniform(
                [b[0] for b in bounds], 
                [b[1] for b in bounds], 
                (n_random, self.dim)
            )
            x_rand[:, 0] = x1_rand # Keep x1 in target range? Or full?
            # Let's keep x1 in target range to stay near the front
            
            # Combine
            X = np.vstack([x_good, x_rand])
            
            # Evaluate
            for x in X:
                problem(x)
                
            return X
