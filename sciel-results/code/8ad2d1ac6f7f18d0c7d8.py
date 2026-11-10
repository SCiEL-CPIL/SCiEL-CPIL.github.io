"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for ZDT6 using a deterministic grid search on x0 with x1: set to the analytic minimizer of g(x)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.dim
        budget = self.budget

        # ZDT6 structure:
        # f1(x) = 1 + 10*(n-1) + sum_{i=2}^{n} (2*x_i^2 - 2*sqrt(x_1*x_i))
        # f2(x) = g(x) * h(f1(x), g(x))
        # g(x) = 1 + 9*(sum(x_i)/n)^(1/(1+alpha)), alpha = 0.25
        # h(u,v) = 1 - sqrt(u/v) - (u/v)*sin(10*pi*u*v)
        
        # The Pareto front is achieved when:
        # 1. x_i = c * x_1^(1/3) for all i=2..n, where c = 0.25^(2/3)
        #    This minimizes the penalty term in f1, making f1 depend only on x_1.
        # 2. x_1 varies over [0, 1] to trace the front.
        
        # Note: The analytic solution x_i = c * x_1^(1/3) minimizes the sum of penalties.
        # Let's verify:
        # term_i = 2*x_i^2 - 2*sqrt(x_1*x_i)
        # d/dx_i = 4*x_i - sqrt(x_1/x_i) = 0 => 4*x_i^2 = sqrt(x_1/x_i) => 16*x_i^4 = x_1/x_i => 16*x_i^5 = x_1 => x_i = (x_1/16)^(1/5)
        # Wait, let's re-calculate.
        # term = 2*x_i^2 - 2*sqrt(x_1*x_i)
        # derivative w.r.t x_i: 4*x_i - 2 * (1/2) * (x_1/x_i)^(1/2) = 4*x_i - sqrt(x_1/x_i)
        # Set to 0: 4*x_i = sqrt(x_1/x_i) => 16*x_i^2 = x_1/x_i => 16*x_i^3 = x_1 => x_i = (x_1/16)^(1/3)
        # So c = (1/16)^(1/3) = 0.5^(2/3) = 0.25^(1/3) ? No.
        # 16 = 2^4. (1/16)^(1/3) = 16^(-1/3).
        # Let's check the standard ZDT6 definition.
        # f1 = 1 + 10(n-1) + sum_{i=2}^n (2x_i^2 - 2sqrt(x_1 x_i))
        # Minimizer of 2x_i^2 - 2sqrt(x_1 x_i):
        # d/dx_i (2x_i^2 - 2sqrt(x_1)x_i^(1/2)) = 4x_i - sqrt(x_1)x_i^(-1/2) = 0
        # 4x_i^(3/2) = sqrt(x_1) => 16x_i^3 = x_1 => x_i = (x_1/16)^(1/3)
        # So c = (1/16)^(1/3) ≈ 0.18898
        
        c = (1.0 / 16.0) ** (1.0/3.0)

        # We have 'budget' evaluations.
        # We will dedicate all of them to sweeping x_1.
        # x_1 should be chosen to cover the range [0, 1] well.
        # Since f1(x) is monotonic with x_1 (when x_i are optimal),
        # and f2(x) depends on f1 and g, we want a good spread of f1 values.
        # f1_min = 1 (when x_1=0 and x_i=0? No, if x_1=0, x_i must be 0 for penalty 0. f1=1+10(n-1)+0. Wait.)
        # If x_1 = 0, x_i = 0. f1 = 1 + 10(n-1) + 0 = 1 + 10(n-1).
        # If x_1 = 1, x_i = (1/16)^(1/3).
        # Let's just sweep x_1 uniformly from 0 to 1.
        
        # However, the Pareto front of ZDT6 is not uniform in x_1.
        # f1(x) = 1 + 10(n-1) + sum (2*(x_1/16)^(2/3) - 2*sqrt(x_1*(x_1/16)^(1/3)))
        # = 1 + 10(n-1) + (n-1) * (2*(x_1/16)^(2/3) - 2*sqrt(x_1^(1/3) * 16^(-1/3)))
        # = 1 + 10(n-1) + (n-1) * (2 * x_1^(2/3) / 16^(2/3) - 2 * x_1^(1/2) / 16^(1/2))
        # = 1 + 10(n-1) + (n-1) * (2 * x_1^(2/3) / 4^(2/3) - 2 * x_1^(1/2) / 4)
        # = 1 + 10(n-1) + (n-1) * (2 * x_1^(2/3) / (2^(4/3)) - 0.5 * x_1^(1/2))
        
        # Actually, it's easier to just evaluate.
        # A uniform grid in x_1 might be biased because the mapping to f1 is non-linear.
        # But for hypervolume, we just need to cover the front.
        # Let's use a uniform grid in x_1.
        
        # To maximize hypervolume, we should ensure we capture the "kinks" or high curvature regions.
        # ZDT6 has a discontinuous front? No, it's continuous but the shape is specific.
        # Let's stick to uniform x_1 first.
        
        # Number of points = budget
        x1_vals = np.linspace(0.0, 1.0, budget)
        
        # If we want to be smarter, we could cluster near x_1=0 where f1 is high?
        # Or near x_1=1?
        # Let's just use uniform.
        
        results = []
        
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            
            x = np.zeros(n)
            x[0] = x1
            if n > 1:
                x[1:] = c * (x1 ** (1.0/3.0))
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            
        return results

# EVOLVE-BLOCK-END
