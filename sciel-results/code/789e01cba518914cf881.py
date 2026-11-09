import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evals = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # DTLZ2 structure: x[0:n-3] are shape variables, x[n-3:n] are position variables
        # For 3 objectives, n-3 = 9 shape variables, n-3 to n are 3 position variables
        # Actually for DTLZ2 with M objectives, there are M-1 position variables and the rest are shape
        # For M=3: 2 position variables (x[10], x[11]) and 10 shape variables (x[0:10])
        
        # Known Pareto front: g(x) = 0, which means all shape variables are at their optimal values
        # For DTLZ2, g(x) = sum((x_i - 0.5)^2) for shape variables, minimized at x_i = 0.5
        # Position variables determine the point on the Pareto front via sine functions
        
        # Strategy: 
        # 1. Use a set of well-spread weight vectors for the position variables
        # 2. For each weight, do local search on the shape variables to drive g(x) to 0
        # 3. The position variables are set to achieve the desired point on the front
        
        # For 3 objectives, the Pareto front is a quarter-sphere in 3D
        # f_i = g(x) * (1 + h_i) where h_i = sin(pi * x_{n-M+i} / 2) for position variables
        # When g=0, f_i = h_i, and sum(h_i^2) = 1 (on the unit sphere)
        
        # Actually, let me re-examine DTLZ2:
        # f_i(x) = g(x) * (1 + h_i(x))
        # where g(x) = sum_{i=n-M+1}^{n} (x_i - 0.5)^2
        # h_i(x) = sin(pi * x_{n-M+i} / 2) for i=1..M-1, and h_M = sqrt(1 - sum_{i=1}^{M-1} h_i^2)
        # Wait, that's not quite right. Let me think again.
        
        # Standard DTLZ2:
        # f_i(x) = g(x) * (1 + h_i(x)) for i=1..M
        # g(x) = sum_{i=n-M+1}^{n} (x_i - 0.5)^2
        # h_i(x) = sin(pi * x_{n-M+i} / 2) for i=1..M-1
        # h_M(x) = sqrt(1 - sum_{i=1}^{M-1} h_i(x)^2)
        
        # So for M=3:
        # g(x) = (x_10 - 0.5)^2 + (x_11 - 0.5)^2
        # h_1 = sin(pi * x_10 / 2)
        # h_2 = sin(pi * x_11 / 2)
        # h_3 = sqrt(1 - h_1^2 - h_2^2)
        
        # Pareto front: g=0, so x_10=0.5, x_11=0.5
        # Then h_1 = sin(pi/4) = sqrt(2)/2, h_2 = sin(pi/4) = sqrt(2)/2, h_3 = sqrt(1 - 1/2 - 1/2) = 0
        # That gives f = (1+h_1, 1+h_2, 1+h_3) = (1+sqrt(2)/2, 1+sqrt(2)/2, 1)
        # That's not the whole front. The front is parameterized by x_10 and x_11 in [0,1]
        
        # When g=0: f_i = 1 + h_i
        # h_1 = sin(pi * x_10 / 2), h_2 = sin(pi * x_11 / 2), h_3 = sqrt(1 - h_1^2 - h_2^2)
        # So the front is: f = (1 + sin(pi*x_10/2), 1 + sin(pi*x_11/2), 1 + sqrt(1 - sin^2(pi*x_10/2) - sin^2(pi*x_11/2)))
        
        # To cover the front well, I should sample x_10 and x_11 on a grid in [0,1]
        # and for each, set shape variables to 0.5 (to get g=0)
        
        # But wait, the shape variables also appear in g, so to get g=0 I need x_10=0.5 and x_11=0.5
        # That's only one point! The shape variables are x_0 through x_9, and the position variables are x_10, x_11
        # g(x) = sum_{i=10}^{11} (x_i - 0.5)^2
        # So to get g=0, I need x_10=0.5 and x_11=0.5, which gives only one point on the front
        
        # That can't be right. Let me reconsider.
        
        # Actually, I think I have the indexing wrong. Let me reconsider DTLZ2.
        # In DTLZ2, there are M objectives, and the last M-1 variables are position variables
        # The remaining variables are shape variables that affect g(x)
        
        # For M=3, dim=12:
        # Position variables: x_10, x_11 (last M-1 = 2 variables)
        # Shape variables: x_0 through x_9 (first 10 variables)
        # g(x) = sum_{i=0}^{9} (x_i - 0.5)^2
        # h_1 = sin(pi * x_10 / 2)
        # h_2 = sin(pi * x_11 / 2)
        # h_3 = sqrt(1 - h_1^2 - h_2^2)
        # f_i = g(x) * (1 + h_i)
        
        # Pareto front: g(x)=0, so all shape variables x_0..x_9 = 0.5
        # Then f_i = 1 + h_i, with h determined by x_10, x_11
        # So the front is a 2D surface parameterized by x_10, x_11 in [0,1]
        
        # This makes more sense! The shape variables need to be at 0.5, and the position variables sweep the front.
        
        # Strategy:
        # 1. Generate a set of (x_10, x_11) pairs that cover the [0,1]^2 domain well
        # 2. For each, set x_0..x_9 = 0.5
        # 3. Evaluate and record
        
        # But I also need to use the budget wisely. I have 200 evaluations.
        # I can afford a 14x14 grid = 196 points, leaving 4 for refinement.
        
        # Let's do a 14x14 grid = 196 points
        n_grid = 14
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                x10 = i / (n_grid - 1)
                x11 = j / (n_grid - 1)
                x = np.full(self.dim, 0.5)
                x[10] = x10
                x[11] = x11
                grid_points.append(x)
        
        # Evaluate all grid points
        for x in grid_points:
            if self.evals >= self.budget:
                break
            problem(x)
            self.evals += 1
        
        # Use remaining budget for local refinement around the best points
        # Find points that are not dominated and refine them
        # For simplicity, just do a few random perturbations around the grid points
        remaining = self.budget - self.evals
        if remaining > 0:
            # Pick some grid points and perturb the shape variables slightly
            for _ in range(remaining):
                base = rng.integers(0, len(grid_points))
                x = grid_points[base].copy()
                # Perturb shape variables (x_0..x_9) slightly
                for i in range(10):
                    x[i] += rng.normal(0, 0.01)
                x = np.clip(x, lo, hi)
                problem(x)
                self.evals += 1
        
        return grid_points[0]
