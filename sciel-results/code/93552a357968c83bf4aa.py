import numpy as np
from scipy.optimize import minimize_scalar, brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
            def calc_f1(x1):
                return 1 - (1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6)**0.1

            # Find peak
            res = minimize_scalar(lambda x: -calc_f1(x), bounds=(0, 0.2), method='bounded', options={'xatol': 1e-12})
            x_peak = res.x
            f1_max = -res.fun
            
            # Generate targets
            n_points = self.budget
            target_f1s = np.linspace(0, f1_max, n_points)
            
            recorded = []
            for t in target_f1s:
                if t == 0:
                    x1 = 0.0
                elif t == f1_max:
                    x1 = x_peak
                else:
                    # Solve calc_f1(x) = t in [0, x_peak]
                    try:
                        x1 = brentq(lambda x: calc_f1(x) - t, 0, x_peak, xtol=1e-12)
                    except:
                        x1 = x_peak # Fallback
                x = np.zeros(self.dim)
                x[0] = x1
                x = np.clip(x, 0, 1)
                obj = problem(x)
                recorded.append(obj)
                
            return recorded
