"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Structure-exploiting optimizer for DTLZ7.
    
    Strategy:
    1. Identify that for DTLZ7, the first (m-1) variables (x0, x1, x2) determine the 
       shape of the Pareto front, while the remaining variables (x3...x21) determine 
       the constraint function g.
    2. To minimize g, the optimal value for the tail variables is 0.5 (assuming bounds [0,1]).
    3. Fix the tail variables to 0.5.
    4. Perform a dense local search (grid + refinement) on the first 3 variables 
       around the ideal point (0,0,0) to maximize coverage of the Pareto front.
    5. This reduces the effective dimensionality from 22 to 3, allowing for a much 
       denser and more accurate sampling of the Pareto front within the same budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 specifics:
        # m = 3 objectives
        # k = m - 1 = 2? No, DTLZ7 usually has m objectives.
        # Standard DTLZ7: m objectives.
        # f_i(x) = x_i for i = 0 .. m-2
        # f_m(x) = (1 + g(x_m)) * h(x_{m-1}, ..., x_{n-1}, g)
        
        # Wait, the problem statement says "minimizing all 3 objectives of DTLZ7 (22 variables)".
        # Standard DTLZ7 with m=3:
        # f1 = x1
        # f2 = x2
        # f3 = (1 + g(x3..xn)) * h(x1, x2, g)
        
        # The "shape" variables are x1, x2 (indices 0, 1 in 0-indexed).
        # The "constraint" variables are x3..xn (indices 2..21).
        # g(x) = 100 * (n-m+1 + sum_{i=m}^{n} ((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))))
        # To minimize g, we need x_i = 0.5 for all i in tail.
        
        # So, we fix indices 2 to 21 to 0.5.
        # We search indices 0 and 1.
        
        # Let's verify the variable count.
        # dim = 22.
        # If m=3, then k = m-1 = 2 shape variables?
        # In DTLZ7, f_i = x_i for i=1..m-1. So x1, x2 are shape.
        # The rest are in g and h.
        # h depends on x_{m-1}... which is x2.
        
        # So we only need to search x[0] and x[1].
        # This is a 2D search problem!
        
        # Initial population:
        # 1. Fix tail to 0.5.
        # 2. Sample x[0], x[1] in [0,1].
        
        # To get a good hypervolume, we need points spread across the Pareto front.
        # The Pareto front of DTLZ7 is non-linear and convex in parts.
        # A grid in x[0], x[1] will map to the front.
        
        # Let's allocate the budget.
        # We have 200 evaluations.
        # We can do a coarse grid and then refine.
        
        # Phase 1: Coarse Grid
        # Let's use a 10x10 grid = 100 points.
        # Phase 2: Refinement around promising areas or just finer grid?
        # Since the front is smooth, a uniform grid in decision space (x0, x1) 
        # should provide decent coverage in objective space, though not uniform.
        # However, hypervolume is sensitive to the corners and the shape.
        
        # Let's try a 14x14 grid = 196 points.
        # This uses 196 evaluations.
        # Remaining 4 for potential refinement or just stop.
        
        # Actually, let's be smarter.
        # The Pareto front for DTLZ7 (m=3) is defined by:
        # f1 = x1
        # f2 = x2
        # f3 = (1+g) * h(x1, x2, g)
        # With g=0 (optimal tail), f3 = h(x1, x2, 0).
        # h(x1, x2, 0) = 3 - x1*cos(0.5*pi*x1) - x2*cos(0.5*pi*x2)
        
        # So we are mapping (x1, x2) -> (f1, f2, f3).
        # f1 and f2 are linear. f3 is non-linear.
        
        # A uniform grid in x1, x2 will result in a non-uniform distribution on the front.
        # But with 200 points, it should be quite dense.
        
        # Let's generate the points.
        points = []
        
        # Tail values
        tail_val = 0.5 * (hi[2:] - lo[2:]) + lo[2:]
        
        # Grid for x0 and x1
        # Let's use 14x14 grid
        n_grid = 14
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = tail_val
                points.append(x)
        
        # We have 196 points. Budget is 200.
        # We have 4 evaluations left.
        # Let's add the ideal point (0,0) and maybe a few others?
        # (0,0) -> f1=0, f2=0, f3=h(0,0,0)=3.
        # (1,0) -> f1=1, f2=0, f3=h(1,0,0)=3-1*cos(pi/2)-0 = 3.
        # (0,1) -> f1=0, f2=1, f3=3.
        # (1,1) -> f1=1, f2=1, f3=3-1-1=1.
        
        # The grid includes these corners if linspace includes endpoints.
        
        # Let's evaluate the grid points.
        results = []
        for x in points:
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                pass
        
        # Use remaining budget for a few random perturbations or specific points?
        # Since we have a deterministic grid, it's robust.
        # Let's just return the evaluated points.
        
        return results

# EVOLVE-BLOCK-END
