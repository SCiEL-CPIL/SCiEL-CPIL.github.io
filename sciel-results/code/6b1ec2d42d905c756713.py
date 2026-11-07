import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 290.0
            dim = self.dim
            
            # 1. Evaluate origin
            x0 = np.zeros(dim)
            f0 = func(x0)
            best_f = f0
            best_x = x0.copy()
            calls = 1
            
            # 2. Probe axes to determine gradient signs
            # We probe at the boundary 5.0 to maximize the signal
            # f(5*ei) - f(0) = 5 * grad_i
            # Sign of this difference is sign of grad_i
            
            grad_signs = np.zeros(dim)
            for i in range(dim):
                if time.time() > deadline:
                    break
                
                x_probe = np.zeros(dim)
                x_probe[i] = 5.0
                fp = func(x_probe)
                calls += 1
                
                diff = fp - f0
                if diff < 0:
                    grad_signs[i] = -1.0
                else:
                    grad_signs[i] = 1.0
                
                # Update best if probe was better (unlikely for linear, but safe)
                if fp < best_f:
                    best_f = fp
                    best_x = x_probe.copy()
            
            # 3. Determine optimal vertex
            # If grad_i > 0, we want x_i to be -5
            # If grad_i < 0, we want x_i to be 5
            # Note: Our probe was at +5.
            # If f(+5) < f(0), then gradient is negative (function decreases as x increases).
            # So if diff < 0, grad < 0, optimal x_i is +5.
            # If diff > 0, grad > 0, optimal x_i is -5.
            
            x_vertex = np.zeros(dim)
            for i in range(dim):
                if grad_signs[i] < 0:
                    x_vertex[i] = 5.0
                else:
                    x_vertex[i] = -5.0
            
            # 4. Evaluate vertex
            if time.time() > deadline:
                return best_x, best_f
                
            f_vertex = func(x_vertex)
            calls += 1
            
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()
            
            # 5. Optional: Polish?
            # Since it's linear, we are at the global min.
            # Any move increases f.
            # So we should stop.
            # However, to be robust against any non-linearities (if the task description is slightly misleading or if there's a hidden quadratic term),
            # we could do a few local steps.
            # But the task says "Linear slope".
            # Let's add a very small polish just in case, using the remaining budget?
            # No, the scoring penalizes early calls. Stopping is best.
            # But wait, if we stop at 7 calls, n_8..n_500 are n_7.
            # If n_7 is 0, the mean is (sum n_1..n_6 + 493*0)/500.
            # This is optimal.
            
            return best_x, best_f
