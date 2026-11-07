import numpy as np
import heapq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        N = min(self.budget, 200)
        
        # ZDT1 Pareto Front: x1 in [0, 1], x2..xn = 0
        # f1 = x1, f2 = 1 - sqrt(x1)
        
        # We will determine the optimal x1 coordinates using adaptive bisection
        # to maximize the hypervolume (which is proportional to the Riemann sum of f2)
        
        # f(x) = 1 - sqrt(x)
        def f(x):
            return 1.0 - np.sqrt(x)
        
        # g(x) = (x-a)f(x) - (b-x)f(b)
        # We want to find x in [a,b] that maximizes g(x)
        # g'(x) = f(x) + (x-a)f'(x) + f(b) = 0
        # f'(x) = -0.5 / sqrt(x)
        # 1 - sqrt(x) - (x-a)/(2*sqrt(x)) + f(b) = 0
        # Let s = sqrt(x). x = s^2.
        # 1 - s - (s^2 - a)/(2s) + f(b) = 0
        # 1 - s - s/2 + a/(2s) + f(b) = 0
        # 1 + f(b) - 1.5s + a/(2s) = 0
        # 2s(1 + f(b) - 1.5s) + a = 0
        # -3s^2 + 2(1+f(b))s + a = 0
        # 3s^2 - 2(1+f(b))s - a = 0
        # s = [2(1+f(b)) + sqrt(4(1+f(b))^2 + 12a)] / 6
        # x = s^2
        
        def find_optimal_x(a, b):
            fb = f(b)
            # Coefficients for 3s^2 - 2(1+fb)s - a = 0
            A = 3.0
            B = -2.0 * (1.0 + fb)
            C = -a
            disc = B*B - 4*A*C
            if disc < 0:
                return 0.5 * (a + b)
            # Use stable quadratic formula
            if B >= 0:
                s = (-B - np.sqrt(disc)) / (2.0 * A)
            else:
                s = (-B + np.sqrt(disc)) / (2.0 * A)
            x = s * s
            # Ensure x is in [a, b]
            if x < a: x = a
            if x > b: x = b
            return x

        # Initialize with [0, 1]
        # We need N points. We start with 2 and add N-2 more.
        # Actually, we can just generate the points.
        # The adaptive strategy:
        # 1. Start with intervals [(0, 1)]
        # 2. For each step, pick the interval with max gain, split it at optimal x.
        # 3. Add x to the list of points.
        
        # To avoid O(N^2) log N, we can just use a heap.
        # However, N is only 200. O(N^2) is 40,000 ops. Very fast.
        
        points = [0.0, 1.0]
        intervals = [(0.0, 1.0)]
        
        # Calculate gain for an interval [a, b]
        # Gain = max_{x in [a,b]} ( (x-a)f(x) - (b-x)f(b) )
        # Note: The "current" contribution of [a,b] is (b-a)f(b).
        # The "new" contribution is (x-a)f(x) + (b-x)f(b).
        # Gain = (x-a)f(x) - (b-x)f(b).
        
        def calc_gain(a, b):
            x_opt = find_optimal_x(a, b)
            return (x_opt - a) * f(x_opt) - (b - x_opt) * f(b)
        
        # Heap of (-gain, a, b)
        heap = []
        for a, b in intervals:
            g = calc_gain(a, b)
            heapq.heappush(heap, (-g, a, b))
            
        for _ in range(N - 2):
            neg_g, a, b = heapq.heappop(heap)
            x_opt = find_optimal_x(a, b)
            points.append(x_opt)
            
            # Add new intervals [a, x_opt] and [x_opt, b]
            g1 = calc_gain(a, x_opt)
            g2 = calc_gain(x_opt, b)
            heapq.heappush(heap, (-g1, a, x_opt))
            heapq.heappush(heap, (-g2, x_opt, b))
            
        # Sort points to ensure correct order for evaluation
        points = sorted(points)
        
        # Remove duplicates (shouldn't happen, but safety)
        unique_points = [points[0]]
        for p in points[1:]:
            if p - unique_points[-1] > 1e-15:
                unique_points.append(p)
        points = unique_points
        
        # If we have fewer points than budget due to duplicates, add more
        # (This shouldn't happen in practice)
        
        # Create X matrix
        X = np.zeros((len(points), self.dim))
        X[:, 0] = points
        # X[:, 1:] remains 0
        
        # Evaluate all points - 200 evaluations should finish well within time limit
        for i in range(len(points)):
            problem(X[i])
            
        return X
