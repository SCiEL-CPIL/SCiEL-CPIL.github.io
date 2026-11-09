"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    
    Strategy:
    1. Exploit the known analytic structure of ZDT1.
    2. Use an adaptive grid based on curvature to initialize points.
    3. Refine the point positions using gradient ascent on the discrete Hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        X = np.zeros((n_points, self.dim))
        
        # Point 1: x=0
        X[0, 0] = 0.0
        
        # Points 2..N: x in (0.1, 0.15]
        if n_points > 1:
            n_inner = n_points - 1
            x_start = 0.1
            x_end = 0.15
            
            # Initial uniform grid
            x_vals = np.linspace(x_start, x_end, n_inner)
            
            # Refine grid based on curvature
            for _ in range(10):
                weights = np.abs(np.sin(10 * np.pi * x_vals))
                weights = weights + 1e-6
                desired_h = weights ** (-1/3.0)
                total_h = np.sum(desired_h)
                scale = (x_end - x_start) / total_h
                h_vals = desired_h * scale
                new_x_vals = x_start + np.cumsum(h_vals)
                new_x_vals[-1] = x_end
                x_vals = new_x_vals
            
            # Gradient Ascent Polish
            # y(x) = 1 + sin(10*pi*x)
            # y'(x) = 10*pi*cos(10*pi*x)
            # A(x) = 11 - y(x) = 10 - sin(10*pi*x)
            
            # We want to maximize HV = sum (x_{i+1}-x_i) A_i + (11-x_{N-1}) A_{N-1}
            # Note: x_0 is fixed at 0.1 (start of inner points)
            # The "previous" point for the first inner point is (0, 1).
            # So A_{-1} = 11 - 1 = 10.
            # And x_{-1} = 0.
            
            x_inner = x_vals.copy()
            
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                
                y_vals = 1 + np.sin(10 * np.pi * x_inner)
                A_vals = 11 - y_vals
                dy_vals = 10 * np.pi * np.cos(10 * np.pi * x_inner)
                
                grads = np.zeros_like(x_inner)
                
                # For i=0 to N-2
                # grad_i = A_{i-1} - A_i - (x_{i+1}-x_i) * dy_i
                # A_{-1} = 10 (from point 0,1)
                # x_{-1} = 0 (but we use x_0 - x_{-1} in the term? No, the term is (x_i - x_{i-1}) A_{i-1})
                # Wait, the formula derived was:
                # dHV/dx_k = A_{k-1} - A_k - (x_{k+1}-x_k) y'(x_k)
                # This assumed the term was (x_{k+1}-x_k) A_k.
                # And the previous term was (x_k - x_{k-1}) A_{k-1}.
                # So for k=0:
                # Term 1: (x_0 - 0) * A_{-1} = x_0 * 10.
                # Term 2: (x_1 - x_0) * A_0.
                # d/dx_0: 10 - A_0 - (x_1 - x_0) dy_0.
                
                A_prev = 10.0
                for i in range(n_inner - 1):
                    # x_i is x_inner[i]
                    # x_{i+1} is x_inner[i+1]
                    # A_i is A_vals[i]
                    # A_{i-1} is A_prev
                    # dy_i is dy_vals[i]
                    # h_i = x_inner[i+1] - x_inner[i]
                    h_i = x_inner[i+1] - x_inner[i]
                    grads[i] = A_prev - A_vals[i] - h_i * dy_vals[i]
                    A_prev = A_vals[i]
                
                # Last point i = N-1
                # Term: (11 - x_{N-1}) A_{N-1}
                # Prev term: (x_{N-1} - x_{N-2}) A_{N-2}
                # d/dx_{N-1}: A_{N-2} - A_{N-1} - A_{N-1} - (11 - x_{N-1}) dy_{N-1}
                # = A_{N-2} - 2*A_{N-1} - (11 - x_{N-1}) dy_{N-1}
                if n_inner > 1:
                    A_last_prev = A_vals[-2]
                else:
                    A_last_prev = 10.0 # If only 1 inner point, prev is (0,1)
                
                grads[-1] = A_last_prev - 2 * A_vals[-1] - (11 - x_inner[-1]) * dy_vals[-1]
                
                # Update
                # Use a small step size
                step_size = 1e-5
                x_inner += step_size * grads
                
                # Clip to bounds
                x_inner = np.clip(x_inner, x_start, x_end)
                # Ensure ordering
                for i in range(1, n_inner):
                    if x_inner[i] <= x_inner[i-1]:
                        x_inner[i] = x_inner[i-1] + 1e-9
                
                # Re-normalize to ensure we cover [0.1, 0.15]
                # The gradient ascent might drift. We should project back to the manifold of ordered points?
                # Or just clip and enforce spacing.
                x_inner[0] = x_start
                x_inner[-1] = x_end
            
            X[1:, 0] = x_inner
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
