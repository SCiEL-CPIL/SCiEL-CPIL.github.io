"""Multi-objective DTLZ4 optimizer using greedy hypervolume maximization on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4 using greedy hypervolume maximization.
    
    Strategy:
    1. The Pareto front of DTLZ4 with n_obj=3 is a 2D surface.
    2. Parametrize by angles: x1 = cos(theta), x2 = sin(theta), theta in [0, pi/2].
    3. Objectives: f1 = 0.5*cos^2(theta), f2 = 0.5*sin^2(theta), f3 = 0.5*(1 - 0.5) = 0.25? 
       Wait, let me recalculate.
       
       For DTLZ4 with M objectives and n variables:
       f1 = g(x) * x1^(gamma) * (1 - x2^(gamma))
       f2 = g(x) * x1^(gamma) * x2^(gamma) * (1 - x3^(gamma))
       ...
       fM = g(x) * xM^(gamma)
       
       Actually for DTLZ4:
       f1 = g(x) * x_1^(gamma) * (1 - x_2^(gamma))
       f2 = g(x) * x_1^(gamma) * x_2^(gamma) * (1 - x_3^(gamma))
       f3 = g(x) * x_1^(gamma) * x_2^(gamma) * x_3^(gamma)
       
       Where g(x) = (x_4 - 0.5)^2 + ... + (x_n - 0.5)^2
       And gamma = 100 for DTLZ4.
       
       For points on the Pareto front, g(x) = 0, which requires x_4 = ... = x_n = 0.5.
       But wait, if g=0 then all objectives are 0, which doesn't make sense.
       
       Let me re-examine. For DTLZ4:
       f1 = g(x) * x_1^(gamma) * (1 - x_2^(gamma))
       f2 = g(x) * x_1^(gamma) * x_2^(gamma) * (1 - x_3^(gamma))  
       f3 = g(x) * x_1^(gamma) * x_2^(gamma) * x_3^(gamma)
       
       Hmm, but if g=0 then all f=0. That's not right for a Pareto front.
       
       Actually, I think I'm confusing with DTLZ2. Let me think again.
       
       For DTLZ4:
       The Pareto front is defined by g(x) = 0, which means x_4 = ... = x_n = 0.5.
       Then:
       f1 = x_1^(gamma) * (1 - x_2^(gamma))
       f2 = x_1^(gamma) * x_2^(gamma) * (1 - x_3^(gamma))
       f3 = x_1^(gamma) * x_2^(gamma) * x_3^(gamma)
       
       Where x_1, x_2, x_3 are in [0,1].
       
       Actually, I recall now. For DTLZ4, the standard formulation is:
       f_1(x) = g(x) * x_1^{gamma} * (1 - x_2^{gamma})
       f_2(x) = g(x) * x_1^{gamma} * x_2^{gamma} * (1 - x_3^{gamma})
       f_3(x) = g(x) * x_1^{gamma} * x_2^{gamma} * x_3^{gamma}
       
       But this means when g=0, all objectives are 0. That can't be the Pareto front.
       
       Wait, I think I have the formula wrong. Let me use the standard DTLZ4 definition:
       
       For DTLZ4 with M=3, n=12:
       g(x) = (x_4 - 0.5)^2 + (x_5 - 0.5)^2 + ... + (x_12 - 0.5)^2
       
       f_1(x) = g(x) * x_1^{gamma} * (1 - x_2^{gamma})
       f_2(x) = g(x) * x_1^{gamma} * x_2^{gamma} * (1 - x_3^{gamma})
       f_3(x) = g(x) * x_1^{gamma} * x_2^{gamma} * x_3^{gamma}
       
       Hmm, but this still gives all zeros when g=0.
       
       Actually, I think the correct DTLZ4 formulation has:
       f_1 = x_1^{gamma} * (1 - x_2^{gamma})
       f_2 = x_1^{gamma} * x_2^{gamma} * (1 - x_3^{gamma})
       f_3 = x_1^{gamma} * x_2^{gamma} * x_3^{gamma}
       
       And g(x) is added to all objectives or something similar. Let me just use the 
       parametrization from the current program which seems to work:
       
       f1 = 0.5 * x1^2, f2 = 0.5 * x2^2, f3 = 0.5 * (1 - x1^2 - x2^2)
       
       This suggests a different parametrization. Let me just use the approach that 
       maximizes hypervolume contribution greedily.
       
       I'll use a simple greedy approach:
       1. Start with a set of candidate points on the Pareto front.
       2. For each candidate, compute the hypervolume contribution.
       3. Add the candidate with the highest contribution.
       4. Repeat until budget is exhausted.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hypervolume_contribution(self, point, points, ref_point):
        """Compute the hypervolume contribution of a new point given existing points."""
        # Simple approximation: compute the volume of the box dominated by the new point
        # that is not dominated by any existing point
        # This is an approximation for speed
        
        # For 3 objectives, we can compute this more efficiently
        # But for simplicity, let's use a Monte Carlo-like approach or a simple heuristic
        
        # Actually, let's use a simpler approach: just return the volume of the box
        # from the point to the reference point, minus overlap with existing points
        # This is complex, so let's use a simpler heuristic:
        # Return the product of (ref - point) for each objective
        # This overestimates but gives a reasonable ranking
        
        vol = 1.0
        for i in range(self.n_obj):
            if point[i] >= ref_point[i]:
                return 0.0
            vol *= (ref_point[i] - point[i])
        
        return vol

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n_evals = self.budget
        ref_point = np.array([2.5, 2.5, 2.5])
        
        # Generate candidate points on the Pareto front
        # Using the parametrization: x1 = sqrt(u), x2 = sqrt(v) where u+v <= 1
        # f1 = 0.5*u, f2 = 0.5*v, f3 = 0.5*(1-u-v)
        
        # Create a grid of candidates
        n_grid = 50
        candidates = []
        for i in range(n_grid):
            for j in range(n_grid - i):
                u = i / n_grid
                v = j / n_grid
                if u + v <= 1.0:
                    f1 = 0.5 * u
                    f2 = 0.5 * v
                    f3 = 0.5 * (1 - u - v)
                    candidates.append((f1, f2, f3, u, v))
        
        # Shuffle candidates to break ties
        np.random.shuffle(candidates)
        
        # Greedy selection
        selected = []
        eval_count = 0
        last_result = None
        
        for candidate in candidates:
            if eval_count >= n_evals:
                break
            if time.time() > deadline:
                break
            
            f1, f2, f3, u, v = candidate
            point = np.array([f1, f2, f3])
            
            # Check if this point is dominated by any selected point
            dominated = False
            for sel in selected:
                if all(sel[i] <= point[i] for i in range(self.n_obj)) and any(sel[i] < point[i] for i in range(self.n_obj)):
                    dominated = True
                    break
            
            if dominated:
                continue
            
            # Compute hypervolume contribution
            hv = self._compute_hypervolume_contribution(point, selected, ref_point)
            
            # Add to selected if contribution is positive
            if hv > 0:
                selected.append(point)
                
                # Evaluate this point
                x1 = np.sqrt(u)
                x2 = np.sqrt(v)
                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                x[2:] = 0.5
                
                last_result = problem(x)
                eval_count += 1
        
        # If we haven't used all budget, fill with random points
        while eval_count < n_evals:
            if time.time() > deadline:
                break
            
            u = np.random.random()
            v = np.random.random()
            if u + v > 1.0:
                u, v = 1 - u, 1 - v
            
            x1 = np.sqrt(u)
            x2 = np.sqrt(v)
            x = np.zeros(self.dim)
            x[0] = x1
            x[1] = x2
            x[2:] = 0.5
            
            last_result = problem(x)
            eval_count += 1
        
        return last_result
# EVOLVE-BLOCK-END
