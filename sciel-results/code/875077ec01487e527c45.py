import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        B = self.budget
        calls = 0
        
        # 1. Chebyshev Grid Initialization
        # Use 3 points per dimension to stay within budget (3^5 = 243)
        # Chebyshev nodes of the first kind (roots) scaled to [-5, 5]
        # Roots of T_3(x) are cos(pi/6), cos(3pi/6), cos(5pi/6)
        # Values: sqrt(3)/2, 0, -sqrt(3)/2
        # Scaled to [-5, 5]: 5*sqrt(3)/2, 0, -5*sqrt(3)/2
        nodes = np.array([-5 * np.sqrt(3) / 2, 0, 5 * np.sqrt(3) / 2])
        
        # Generate all combinations
        grid_points = []
        for x0 in nodes:
            for x1 in nodes:
                for x2 in nodes:
                    for x3 in nodes:
                        for x4 in nodes:
                            grid_points.append([x0, x1, x2, x3, x4])
        
        # Ensure we don't exceed budget
        # If n > 5, this logic breaks, but task says dim=5.
        # If budget is very small, we might need to reduce grid size.
        # 3^5 = 243. Budget 500. OK.
        
        best_x = None
        best_f = float('inf')
        
        for p in grid_points:
            if calls >= B:
                break
            f_val = func(p)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(p)
                
        if best_x is None:
            # Fallback if no calls made or something went wrong
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        # 2. Adaptive Bisection / Plateau Search
        # We will iterate a few times to refine the center of the minimum plateau.
        # Each iteration updates one dimension at a time (Coordinate Descent).
        
        # Parameters for bisection
        bisection_iter = 15  # Number of bisection steps per boundary
        epsilon = 1e-8       # Tolerance for "equal" function values
        
        # We have B - calls remaining.
        # Each dimension update costs ~ 2 * bisection_iter calls (left and right).
        # 5 dims * 30 calls = 150 calls per full cycle.
        # We can afford 2-3 cycles.
        
        cycles = 0
        max_cycles = 3
        
        while cycles < max_cycles and calls < B - (2 * bisection_iter * n):
            cycles += 1
            for i in range(n):
                # Find Left Boundary
                lo = -5.0
                hi = best_x[i]
                f_hi = best_f
                
                # We assume the function is flat or lower to the left until the boundary.
                # Actually, we want to find the extent of the region where f(x) <= best_f + tol.
                # Bisection logic:
                # If f(mid) <= f_hi, the boundary is further left (lo = mid).
                # If f(mid) > f_hi, the boundary is to the right (hi = mid).
                
                for _ in range(bisection_iter):
                    if calls >= B:
                        break
                    mid = (lo + hi) / 2.0
                    x_test = best_x.copy()
                    x_test[i] = mid
                    f_mid = func(x_test)
                    calls += 1
                    
                    if f_mid <= f_hi + epsilon:
                        lo = mid
                    else:
                        hi = mid
                        
                left_edge = (lo + hi) / 2.0
                
                # Find Right Boundary
                lo = best_x[i]
                hi = 5.0
                f_lo = best_f
                
                for _ in range(bisection_iter):
                    if calls >= B:
                        break
                    mid = (lo + hi) / 2.0
                    x_test = best_x.copy()
                    x_test[i] = mid
                    f_mid = func(x_test)
                    calls += 1
                    
                    if f_mid <= f_lo + epsilon:
                        hi = mid
                    else:
                        lo = mid
                        
                right_edge = (lo + hi) / 2.0
                
                # Update best_x[i] to the center of the found plateau
                best_x[i] = (left_edge + right_edge) / 2.0
                
                # Evaluate the new point to update best_f
                if calls < B:
                    new_f = func(best_x)
                    calls += 1
                    if new_f < best_f:
                        best_f = new_f
        
        return best_x, best_f
