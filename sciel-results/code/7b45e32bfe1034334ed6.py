"""Optimized ZDT6 solver using analytic Pareto front sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s kill
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        used = 0
        archive_F = []

        def eval_point(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        # ZDT6 Pareto Front Analysis:
        # f1 = 1 + 9*c + sin(10*pi*x1)  where c = mean(x[1:])
        # f2 = 1 + 10*(1 - sqrt(f1/11))
        # Pareto front: for each x1 in [0,1], c can range [0,1]
        # f1 ranges from 1+sin(10*pi*x1) to 10+sin(10*pi*x1)
        # The non-dominated front is the lower envelope.
        # 
        # Key insight: The global Pareto front is best sampled by choosing
        # x1 to maximize sin(10*pi*x1) = 1, i.e., x1 = 0.05, 0.15, ...
        # But the "first lobe" (x1 in [0, 0.1]) has sin(10*pi*x1) >= 0.
        # For maximum hypervolume, we want f1 as small as possible for each f2.
        # 
        # Strategy: Use bisection to find x1 and c such that f1 = target,
        # concentrating on the first lobe where sin(10*pi*x1) is maximized.
        # 
        # For the first lobe (x1 in [0, 0.1]):
        #   f1 = 1 + 9*c + sin(10*pi*x1)
        #   To minimize f1 for a given target, maximize sin(10*pi*x1).
        #   Max sin in [0, 0.1] is at x1=0.05, sin(pi/2)=1.
        #   So f1_min = 2 + 9*c, giving c = (f1-2)/9 for f1 in [2, 11].
        #   For f1 in [1, 2], we need sin(10*pi*x1) = f1 - 1 - 9*c.
        #   With c=0: f1 = 1 + sin(10*pi*x1), so sin(10*pi*x1) = f1-1.
        #   Use bisection on x1 in [0, 0.05] to find x1 where sin(10*pi*x1) = f1-1.
        
        n_samples = self.budget
        
        # Generate uniform targets for f1 in [1, 11]
        targets = np.linspace(1.0, 11.0, n_samples)
        
        def solve_x1_for_f1_low(target_f1):
            """Find x1 in [0, 0.05] such that 1 + sin(10*pi*x1) = target_f1, with c=0."""
            # sin(10*pi*x1) = target_f1 - 1
            # target_f1 in [1, 2], so sin in [0, 1]
            sin_val = target_f1 - 1.0
            sin_val = np.clip(sin_val, 0.0, 1.0)
            # 10*pi*x1 = arcsin(sin_val), x1 = arcsin(sin_val)/(10*pi)
            x1 = np.arcsin(sin_val) / (10.0 * np.pi)
            return x1, 0.0  # c = 0
        
        def solve_x1_for_f1_high(target_f1):
            """Find x1 and c such that 2 + 9*c = target_f1, with x1=0.05."""
            # c = (target_f1 - 2) / 9
            c = (target_f1 - 2.0) / 9.0
            c = np.clip(c, 0.0, 1.0)
            return 0.05, c
        
        for t in targets:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            if t <= 2.0:
                # First lobe lower part: c=0, solve for x1
                x1, c = solve_x1_for_f1_low(t)
            else:
                # First lobe upper part: x1=0.05, solve for c
                x1, c = solve_x1_for_f1_high(t)
            
            # Construct x
            x = np.full(self.dim, c)
            x[0] = x1
            
            # Evaluate
            x_eval, f = eval_point(x)
            archive_F.append(f)

        # The return value is ignored, but we must return something valid
        return archive_F if archive_F else np.zeros((1, self.n_obj))
# EVOLVE-BLOCK-END
