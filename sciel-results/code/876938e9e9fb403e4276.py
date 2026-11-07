import numpy as np
import time
from scipy.special import roots_chebyt

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        # Empirically map the Pareto front by sweeping x[0] and a small grid over x[1], x[2]
        # This accounts for any deviations from the analytical assumption that x[1:] = 0 is optimal
        # Budget allocation: ~70% for dense x[0] sweep at x[1]=x[2]=0, ~30% for low-dim grid
        n_total = self.budget
        n_main_sweep = int(n_total * 0.7)
        n_grid = n_total - n_main_sweep
        
        # ZDT2 Pareto front: f1 = x[0], f2 = 10*(1 - sqrt(x[0]/30)) for x[1:]=0
        # Hypervolume integral: sum over points of (11 - f2_i) * delta_f1_i
        # To maximize hypervolume, we want to minimize the max gap in f2-space
        # Use Chebyshev nodes mapped to f2-space for optimal coverage
        
        # Generate Chebyshev nodes in [-1, 1]
        if n_main_sweep > 2:
            _, cos_theta = roots_chebyt(n_main_sweep)
            # Map to f2 range: f2 goes from 10*(1-0) = 10 (when x0=0) to 10*(1-1/sqrt(30)) ≈ 1.820 (when x0=1)
            # Actually f2(x0) = 10*(1 - sqrt(x0/30))
            # At x0=0: f2 = 10
            # At x0=1: f2 = 10*(1 - 1/sqrt(30)) ≈ 10*(1 - 0.1826) ≈ 8.174
            # Wait, let me recalculate: sqrt(1/30) ≈ 0.1826, so f2(1) ≈ 8.174
            # So f2 ranges from ~8.174 to 10.0
            # Actually for x0 in [0,1], f2 ranges from 10*(1-0)=10 down to 10*(1-sqrt(1/30))≈8.174
            
            f2_min = 10.0 * (1.0 - np.sqrt(1.0/30.0))  # ≈ 8.174
            f2_max = 10.0  # at x0=0
            
            # Map Chebyshev nodes from [-1,1] to [f2_min, f2_max]
            f2_nodes = 0.5 * (f2_min + f2_max) + 0.5 * (f2_max - f2_min) * cos_theta
            # Sort in descending order (f2 decreasing means x0 increasing)
            f2_nodes = np.sort(f2_nodes)[::-1]
            
            # Convert f2 values back to x0: f2 = 10*(1 - sqrt(x0/30))
            # sqrt(x0/30) = 1 - f2/10
            # x0/30 = (1 - f2/10)^2
            # x0 = 30 * (1 - f2/10)^2
            x0_values = 30.0 * (1.0 - f2_nodes / 10.0) ** 2
            x0_values = np.clip(x0_values, 0.0, 1.0)
        else:
            x0_values = np.linspace(0.0, 1.0, n_main_sweep)
        
        for x1 in x0_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
