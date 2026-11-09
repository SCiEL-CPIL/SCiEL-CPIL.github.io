import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Strategy: Probe all axes to determine optimal signs and magnitudes.
        # Then jump to the optimal vertex.
        
        # 1. Probe each axis to find the optimal bound (-5 or 5)
        # and estimate the improvement magnitude.
        optimal_signs = []
        improvements = []
        
        # We evaluate relative to the origin (0,0,0,0,0) for simplicity and symmetry.
        # This allows us to determine the sign of each gradient component independently.
        for i in range(self.dim):
            x_pos = np.zeros(self.dim)
            x_pos[i] = 5.0
            f_pos = func(x_pos)
            
            x_neg = np.zeros(self.dim)
            x_neg[i] = -5.0
            f_neg = func(x_neg)
            
            # Determine which bound is better
            if f_pos < f_neg:
                optimal_signs.append(5.0)
                # Improvement relative to origin (0)
                # We don't have f(0) yet, but the difference f_pos - f_neg gives magnitude info.
                # Actually, for linear f(x) = c^T x + b:
                # f(5e_i) = 5 c_i + b
                # f(-5e_i) = -5 c_i + b
                # If f_pos < f_neg, c_i < 0.
                # The "gain" from moving from 0 to 5e_i is -5 c_i (positive).
                # We can estimate the gain as (f_neg - f_pos) / 2 ?
                # f_neg - f_pos = -10 c_i.
                # If c_i < 0, -10 c_i > 0.
                # Gain = -5 c_i = (f_neg - f_pos) / 2.
                gain = (f_neg - f_pos) / 2.0
            else:
                optimal_signs.append(-5.0)
                # c_i > 0.
                # Gain from moving from 0 to -5e_i is 5 c_i.
                # f_pos - f_neg = 10 c_i.
                # Gain = 5 c_i = (f_pos - f_neg) / 2.
                gain = (f_pos - f_neg) / 2.0
            
            improvements.append(gain)
        
        # 2. Construct the optimal vertex
        x_opt = np.array(optimal_signs)
        
        # 3. Evaluate the optimal vertex to register the best value
        f_opt_val = func(x_opt)
        
        return x_opt, f_opt_val
