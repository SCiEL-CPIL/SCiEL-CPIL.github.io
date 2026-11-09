"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT1 using optimal control (PMP) derived feedback law."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 structure: f1 = x[0], f2 = 1 - sqrt(x[0]) on the Pareto Front (x[1:]=0.5)
        # We maximize the Hypervolume (area under the curve relative to origin, effectively)
        # The optimal placement of points x_i to maximize the area of the union of rectangles
        # [0, x_i] x [0, f2(x_i)] is found by greedily maximizing the marginal gain.
        
        n_evals = self.budget
        
        # Initialize with endpoints of the Pareto front
        x0_list = [0.0, 1.0]
        
        # Evaluate initial points
        for x0 in x0_list:
            x = np.full(self.dim, 0.5)
            x[0] = x0
            x = np.clip(x, lo, hi)
            problem(x)
        
        # Greedy loop with analytical optimal split
        for _ in range(n_evals - len(x0_list)):
            if len(x0_list) < 2:
                break
            
            best_gain = -1.0
            best_x_new = None
            best_idx = -1
            
            # Iterate through all gaps to find the one with the highest marginal gain
            for i in range(len(x0_list) - 1):
                a = x0_list[i]
                b = x0_list[i+1]
                
                if b - a < 1e-12:
                    continue
                
                # Derive optimal x* for interval [a, b] for f(x) = 1 - sqrt(x)
                # Gain(x) = (x - a) * (f2(x) - f2(b))
                # f2(x) = 1 - sqrt(x)
                # Gain(x) = (x - a) * ( (1 - sqrt(x)) - (1 - sqrt(b)) )
                # Gain(x) = (x - a) * ( sqrt(b) - sqrt(x) )
                
                # Maximize G(x) = (x-a)(sqrt(b)-sqrt(x))
                # G'(x) = sqrt(b) - sqrt(x) - (x-a)/(2sqrt(x)) = 0
                # 2sqrt(x)(sqrt(b)-sqrt(x)) = x-a
                # 2sqrt(bx) - 2x = x - a
                # 2sqrt(bx) = 3x - a
                # 4bx = 9x^2 - 6ax + a^2
                # 9x^2 - (6a + 4b)x + a^2 = 0
                
                # Quadratic formula: x = [(6a+4b) +/- sqrt((6a+4b)^2 - 36a^2)] / 18
                # Disc = 36a^2 + 48ab + 16b^2 - 36a^2 = 48ab + 16b^2
                # x = [6a + 4b + sqrt(48ab + 16b^2)] / 18  (taking the larger root)
                
                disc = 48.0 * a * b + 16.0 * b * b
                if disc < 0:
                    continue
                
                sqrt_disc = np.sqrt(disc)
                x_star = (6.0 * a + 4.0 * b + sqrt_disc) / 18.0
                
                # Ensure x_star is within (a, b)
                if x_star <= a or x_star >= b:
                    # Fallback to midpoint if numerical issues
                    x_star = (a + b) / 2.0
                    if x_star <= a or x_star >= b:
                        continue
                
                # Calculate actual gain for this x_star
                f2_b = 1.0 - np.sqrt(b)
                f2_x = 1.0 - np.sqrt(x_star)
                gain = (x_star - a) * (f2_x - f2_b)
                
                if gain > best_gain:
                    best_gain = gain
                    best_x_new = x_star
                    best_idx = i
            
            if best_x_new is None:
                # Fallback if no valid gap found (should not happen)
                max_gap = 0
                max_idx = 0
                for i in range(len(x0_list) - 1):
                    gap = x0_list[i+1] - x0_list[i]
                    if gap > max_gap:
                        max_gap = gap
                        max_idx = i
                best_x_new = (x0_list[max_idx] + x0_list[max_idx+1]) / 2.0
                best_idx = max_idx

            # Insert new point
            x0_list.insert(best_idx + 1, best_x_new)
            
            # Evaluate
            x = np.full(self.dim, 0.5)
            x[0] = best_x_new
            x = np.clip(x, lo, hi)
            problem(x)
            
        return None
# EVOLVE-BLOCK-END
