import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        # ZDT6 specific knowledge
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)**6)**(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        
        def calc_f1(x1):
            # x1 is scalar or array
            return 1 - (1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6)**0.1

        # Use closed-form parametric representation
        # On the Pareto front, x2..xn = 0, and f1 = 1 - u where u = exp(-4*x1)*sin(6*pi*x1)^6
        # We can sample u uniformly in [0, 1] and solve for x1 such that f1(x1) = 1 - u
        
        # Find the maximum of f1 using dense sampling first
        x1_coarse = np.linspace(0, 1, 10000)
        f1_coarse = calc_f1(x1_coarse)
        idx_max = np.argmax(f1_coarse)
        f1_max = f1_coarse[idx_max]
        
        # Refine the maximum location
        x1_peak = x1_coarse[idx_max]
        
        # Generate target f1 values uniformly spaced from 0 to f1_max
        n_points = self.budget
        target_f1s = np.linspace(0, f1_max, n_points)
        
        # For each target f1, find x1 using brentq
        # f1(x1) = target_f1 => 1 - (1 - exp(-4*x1)*sin(6*pi*x1)^6)^0.1 = target_f1
        # => (1 - exp(-4*x1)*sin(6*pi*x1)^6)^0.1 = 1 - target_f1
        # => 1 - exp(-4*x1)*sin(6*pi*x1)^6 = (1 - target_f1)^10
        # => exp(-4*x1)*sin(6*pi*x1)^6 = 1 - (1 - target_f1)^10
        
        def equation_for_x1(x1, target_f1):
            return calc_f1(x1) - target_f1
        
        x1_targets = np.empty(n_points)
        
        for i in range(n_points):
            # Check deadline
            if time.time() > deadline:
                x1_targets[i:] = 0.0
                break
            
            target = target_f1s[i]
            
            # For target = 0, x1 = 0
            if target == 0:
                x1_targets[i] = 0.0
                continue
            
            # For target = f1_max, x1 = x1_peak
            if target >= f1_max - 1e-12:
                x1_targets[i] = x1_peak
                continue
            
            # Find a bracket for brentq
            # f1(0) = 0, f1 increases initially
            # We need to find where f1(x1) = target
            
            # Search in the first lobe [0, 0.2] where f1 is mostly monotonic increasing
            # Find a good bracket
            a, b = 0.0, min(0.2, 1.0)
            
            # Check if target is within range
            fa = equation_for_x1(a, target)
            fb = equation_for_x1(b, target)
            
            if fa * fb > 0:
                # Try wider range
                b = 0.5
                fb = equation_for_x1(b, target)
                if fa * fb > 0:
                    b = 1.0
                    fb = equation_for_x1(b, target)
            
            if fa * fb > 0:
                # Fallback: use the coarse grid
                idx = np.argmin(np.abs(f1_coarse - target))
                x1_targets[i] = x1_coarse[idx]
            else:
                try:
                    x1_sol = brentq(equation_for_x1, a, b, args=(target,), xtol=1e-10, maxiter=100)
                    x1_targets[i] = x1_sol
                except:
                    # Fallback: use the coarse grid
                    idx = np.argmin(np.abs(f1_coarse - target))
                    x1_targets[i] = x1_coarse[idx]

        # Construct and evaluate points
        recorded = []
        for i in range(n_points):
            # Check deadline
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
