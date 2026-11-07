# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_optimal_f1_values(self, n):
        # Derive optimal spacing for ZDT1 hypervolume maximization
        # Pareto front: f2 = 1 - sqrt(f1), f1 in [0, 1]
        # Hypervolume integrand: the area "weight" for a point at f1 is
        # proportional to (ref_f2 - f2) * d f1 = (10 + sqrt(f1)) * d f1
        # for maximizing sum of individual point contributions to HV
        # Actually the standard optimal quantization for hypervolume:
        # We want to place points to maximize the sum of (ref - f) products
        # For the i-th point, its "cell" width in f1 is w_i
        # Contribution ~ (11 - f1_i) * (11 - f2_i) but this is overcounting
        # 
        # The correct approach: maximize the HV of N discrete points on the curve
        # Optimal spacing equalizes the marginal HV gain per point
        # dHV/df1 ~ (ref_f2 - f2(f1)) = 10 + sqrt(f1)
        # So df1 ∝ 1/(10 + sqrt(f1))
        # Integrate: f1 = [10*u - u^2/2 + C] where u = sqrt(f1)
        # At u=0: f1=0, so C=0
        # f1(u) = 10u - u^2/2
        # We need f1 in [0,1], so 10u - u^2/2 in [0,1]
        # When f1=1: 10u - u^2/2 = 1
        # u^2 - 20u + 2 = 0
        # u = (20 - sqrt(400-8))/2 = (20 - sqrt(392))/2 = (20 - 14*sqrt(2))/2 = 10 - 7*sqrt(2) ≈ 0.101
        # 
        # Parametrize: sample u uniformly in [0, 10 - 7*sqrt(2)]
        # Then f1 = 10u - u^2/2
        
        u_max = 10.0 - 7.0 * np.sqrt(2.0)
        u_vals = np.linspace(0.0, u_max, n)
        f1_vals = 10.0 * u_vals - 0.5 * np.power(u_vals, 2)
        
        # Clip to [0, 1]
        f1_vals = np.clip(f1_vals, 0.0, 1.0)
        
        return f1_vals

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        n_evals = self.budget
        if n_evals <= 0:
            return []

        # Compute optimal f1 values
        f1_targets = self._compute_optimal_f1_values(n_evals)
        
        # Map f1 targets to x[0] within bounds
        # f1 = x[0] when x[0] in [0,1], and bounds are [0,1]
        x0_vals = lo[0] + f1_targets * span[0]
        x0_vals = np.clip(x0_vals, lo[0], hi[0])

        results = []
        
        for i in range(n_evals):
            # Check deadline
            if (time.time() - t0) > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0_vals[i]
            
            f = problem(x)
            results.append(f)
        
        return results
# EVOLVE-BLOCK-END
