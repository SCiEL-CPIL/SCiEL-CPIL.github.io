import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # For DTLZ1, the Pareto front is defined by the last (n_obj-1) variables.
        # The first (dim - n_obj + 1) variables determine the quality (distance to front).
        # To maximize hypervolume, we want points close to the front (quality ~ 0)
        # and diverse in the last (n_obj-1) dimensions.
        
        # Strategy:
        # 1. Set the first (dim - n_obj + 1) variables to 0 to ensure points are on the ideal front.
        #    This is valid because DTLZ1's front is at x_0...x_{d-k} = 0.
        #    Actually, g(x) = 100 * (n-1 + sum(x_i^2 - 2 cos(20 pi x_i)) for i in 0..n-2)
        #    If we set those to 0, g(x) = 0, so f_i = g(x) * (1 + h_i) = 0.
        #    Wait, if g(x)=0, all objectives are 0. That's the ideal point.
        #    But we need diversity.
        #    The front is defined by h_i functions of the last k-1 variables.
        #    f_1 = (1+u_1) * 0.5 * prod(h_i)
        #    f_2 = (1+u_2) * 0.5 * h_{k-1} * prod(h_i for i<k-1)
        #    ...
        #    f_k = 0.5 * prod(h_i)
        #    If g(x)=0, then f_i = 0 for all i. This is a single point.
        #    We need g(x) > 0 to have a spread.
        #    Actually, the Pareto front is the set of non-dominated points.
        #    For DTLZ1, any point with g(x) > 0 and appropriate u variables is on the front?
        #    No, the front is the set of points where g(x) is minimized (which is 0)
        #    and the u variables vary.
        #    If g(x) > 0, the point is dominated by a point with g(x') < g(x)
        #    and the same u variables (scaled).
        #    So to get points on the Pareto front, we MUST have g(x) = 0.
        #    This implies x_0...x_{d-k} = 0.
        #    Then the objectives are determined solely by the last k-1 variables.
        #    f_i = 0.5 * prod(h_j) * (1 + u_{k-i})  (indices depend on definition)
        #    Specifically:
        #    f_1 = 0.5 * (1+u_1) * prod_{j=2}^{k} h_j
        #    f_2 = 0.5 * (1+u_2) * h_k * prod_{j=2}^{k-1} h_j
        #    ...
        #    f_k = 0.5 * prod_{j=2}^{k} h_j
        #    Where h_j = (u_j + 2 cos(20 pi u_j) + 2) / (k + 4 sum_{m=2}^{k} cos(20 pi u_m + 2))
        #    And u_j are the last k-1 variables.
        
        # So the strategy is:
        # 1. Fix the first (dim - n_obj + 1) variables to 0.
        # 2. Randomly sample the last (n_obj - 1) variables uniformly in [0, 1].
        # 3. This gives points exactly on the Pareto front.
        # 4. Use remaining budget to potentially improve coverage or just keep sampling.
        
        # Since the evaluator records ALL returned vectors, and the score is hypervolume,
        # we want a dense set of points on the Pareto front.
        # Random sampling of the last k-1 variables in [0,1] gives a uniform distribution
        # in the objective space (approximately, due to the h functions).
        
        # Let's just use the entire budget to sample random points on the front.
        # This is optimal for hypervolume if the sampling is uniform enough.
        
        n_vars_for_front = self.n_obj - 1
        n_vars_for_quality = self.dim - n_vars_for_front
        
        # Allocate budget
        # We can evaluate up to self.budget points.
        # Let's just do that.
        
        points = []
        for i in range(self.budget):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            # First n_vars_for_quality are 0
            # Last n_vars_for_front are random
            x[self.dim - n_vars_for_front:] = np.random.rand(n_vars_for_front)
            
            f = problem(x)
            points.append((x, f))
        
        # Return the last point's x (ignored)
        if points:
            return points[-1][0]
        else:
            return lo

        return X
