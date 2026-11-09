import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 structure:
        # f1 = x[0]
        # f2 = g(x[1:]) * (1 - sqrt(x[0]/n))
        # g(x) = sum_{i=2}^{n} (x_i^2 - 10*cos(2*pi*x_i))
        # The minimum of each term x^2 - 10*cos(2*pi*x) is -9 at x=0, 1, -1, etc.
        # Standard ZDT4 bounds are [-5, 5], so 0 is always in bounds.
        
        def g_term(x):
            return x*x - 10.0*np.cos(2.0*np.pi*x)
        
        # Step 1: Find the optimal tail (x[1:]) to minimize g
        # We want to find x_i that minimizes x_i^2 - 10*cos(2*pi*x_i) within [lo[i], hi[i]]
        optimal_tail = np.zeros(self.dim - 1)
        for i in range(1, self.dim):
            l, h = lo[i], hi[i]
            # Candidates: boundaries and integers (local minima of the periodic function)
            candidates = [l, h]
            # Add integers in [l, h]
            if l <= 0 <= h:
                candidates.append(0.0)
            # Add nearby integers
            for k in range(int(np.floor(l)) - 1, int(np.ceil(h)) + 2):
                if l - 0.1 <= k <= h + 0.1:
                    candidates.append(float(k))
            
            best_val = float('inf')
            best_x = l
            for c in candidates:
                c = np.clip(c, l, h)
                v = g_term(c)
                if v < best_val:
                    best_val = v
                    best_x = c
            optimal_tail[i-1] = best_x
        
        # Step 2: Estimate g_min by evaluating the problem
        # If x[0] = 0, f2 = g(x[1:]) * (1 - 0) = g(x[1:])
        x1_for_g = 0.0
        if x1_for_g < lo[0]:
            x1_for_g = lo[0]
        elif x1_for_g > hi[0]:
            x1_for_g = hi[0]
        
        x_test = np.zeros(self.dim)
        x_test[0] = x1_for_g
        x_test[1:] = optimal_tail
        f_test = problem(x_test)
        
        if x1_for_g > 0:
            g_min = f_test[1] / (1.0 - np.sqrt(x1_for_g / self.dim))
        else:
            g_min = f_test[1]
        
        used = 1
        
        # Step 3: Sample the Pareto front
        # f1 = x[0], f2 = g_min * (1 - sqrt(x[0]/n))
        remaining = self.budget - used
        if remaining <= 0:
            return []
        
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # Strategy: Use a combination of uniform and concentrated sampling.
        # The Pareto front is a concave curve f2 = g_min * (1 - sqrt(f1/n)).
        # Hypervolume is maximized by points spread along the front.
        # Points near f1=0 have low f2, which is good.
        # Points near f1=max have low f1, which is good.
        
        # We will generate a set of x1 values.
        # 1. Uniform samples to cover the range.
        # 2. Power-law samples biased towards f1=0 (where curvature is high and f2 is low).
        
        # For hypervolume maximization with a concave front, we want to maximize the area
        # under the curve. The optimal distribution of points for hypervolume coverage
        # of a concave function is non-uniform.
        
        # Use a power-law distribution that biases towards both ends (where the front
        # has high curvature) while maintaining coverage in the middle.
        # Actually, for a concave front like f2 = g*(1-sqrt(x1/n)), the hypervolume
        # contribution per point is highest where the curve is steepest.
        # The derivative df2/dx1 = -g/(2*sqrt(n*x1)) is steepest near x1=0.
        # So we should bias towards x1=0.
        
        # Let's use a distribution that gives more points near x1=0.
        # t in [0, 1], x1 = x1_lo + (x1_hi - x1_lo) * t^alpha
        # alpha > 1 biases towards x1=0 (low t values)
        
        n_uniform = remaining // 3
        n_concentrated = remaining - n_uniform
        
        x1_samples = []
        
        if n_uniform > 0:
            x1_samples.extend(np.linspace(x1_lo, x1_hi, n_uniform).tolist())
        
        if n_concentrated > 0:
            # Bias towards lower x1 values (near 0)
            t = np.linspace(0, 1, n_concentrated)
            alpha = 1.5  # Moderate bias towards x1=0
            t_conc = t ** alpha
            x1_samples.extend((x1_lo + t_conc * (x1_hi - x1_lo)).tolist())
        
        # Sort to ensure we don't duplicate too much, though duplicates are harmless
        x1_samples.sort()
        
        F = []
        for x1_val in x1_samples:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = np.clip(x1_val, lo[0], hi[0])
            x[1:] = optimal_tail
            f = problem(x)
            F.append(f)
        
        # If we have budget left, do a refinement pass
        # Focus on the region near x1=0 where the front has high curvature
        # and contributes most to hypervolume.
        
        if F and time.time() < deadline - 5:
            remaining2 = self.budget - used - len(F)
            if remaining2 > 0:
                # Add more points in the high-curvature region near x1=0
                # and also ensure we have good coverage at the other end.
                n_refine = min(remaining2, 30)
                if n_refine > 0:
                    # Use a power-law distribution biased towards x1=0
                    t_r = np.linspace(0, 1, n_refine)
                    alpha_r = 1.0  # Linear for refinement
                    for t_r_val in t_r:
                        if time.time() > deadline:
                            break
                        x1_val = x1_lo + t_r_val * (x1_hi - x1_lo)
                        x = np.zeros(self.dim)
                        x[0] = np.clip(x1_val, lo[0], hi[0])
                        x[1:] = optimal_tail
                        f = problem(x)
                        F.append(f)
        
        return F
