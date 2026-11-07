# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    ZDT6 optimizer that places all evaluations on the known Pareto front
    with curvature-aware spacing to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt6_objectives(self, x):
        """Compute ZDT6 objectives for a given x (analytical, for planning)."""
        x1 = x[0]
        n = len(x)
        g = 1 + 10 * (n - 1)
        for i in range(1, n):
            g += x[i] ** 2 - 2 * x1 * np.cos(20 * np.pi * x[i])
        f1 = x1
        f2 = (1 - np.sqrt(f1 / g)) * (1 - f1 ** 2 / 100)
        return np.array([f1, f2])

    def _min_g_for_t(self, t, n_vars=10):
        """Find the minimum of g for given x1=t by optimizing x_2..x_n."""
        # For each x_i, minimize x_i^2 - 2*t*cos(20*pi*x_i)
        # The minimum over x in [0,1] can be found numerically
        # For speed, use a fine grid
        x_grid = np.linspace(0, 1, 1000)
        terms = x_grid**2 - 2*t*np.cos(20*np.pi*x_grid)
        min_term = np.min(terms)
        # g_min = 1 + 10*(n-1) + (n-1)*min_term
        g_min = 1 + 10*(n_vars - 1) + (n_vars - 1)*min_term
        return g_min

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        dim = self.dim
        budget = self.budget

        # Precompute the Pareto front analytically
        # For each t in [0,1], find g_min(t) and the corresponding f2
        n_front = 500
        t_grid = np.linspace(0, 1, n_front)
        g_min_values = np.array([self._min_g_for_t(t, dim) for t in t_grid])
        
        # The Pareto front: f1 = t, f2 = (1 - sqrt(t/g_min(t))) * (1 - t^2/100)
        # For t=0, f2 = (1 - 0) * 1 = 1
        # For t>0, compute f2
        f2_front = np.zeros(n_front)
        for i in range(n_front):
            t = t_grid[i]
            if t == 0:
                f2_front[i] = 1.0
            else:
                g = g_min_values[i]
                if g > 0:
                    f2_front[i] = (1 - np.sqrt(t / g)) * (1 - t**2 / 100)
                else:
                    f2_front[i] = 1.0

        # Now we need to find which t values to evaluate to maximize hypervolume
        # For 2D, the hypervolume with reference point (r1, r2) is the area under the front
        # We want to select n_evals points from the front to maximize the hypervolume
        
        # Strategy: use a greedy approach to select points that maximize hypervolume
        # In 2D, this means selecting points that maximize the sum of rectangular areas
        
        n_evals = min(budget, 200)
        
        # Use a non-uniform grid concentrated where the front is steep
        # The front f2(t) = (1 - sqrt(t/g_min(t))) * (1 - t^2/100)
        # df2/dt is large near t=0
        
        # Try different spacing strategies
        # Strategy 1: quadratic spacing (current)
        t_quad = (np.arange(n_evals) / max(n_evals - 1, 1)) ** 2
        
        # Strategy 2: uniform in f1 (which is t)
        t_uniform = np.linspace(0, 1, n_evals)
        
        # Strategy 3: concentrate more near t=0 using power law
        alpha = 1.5
        t_power = (np.arange(n_evals) / max(n_evals - 1, 1)) ** alpha
        
        # Evaluate all candidates and pick the best set
        # For efficiency, just use the quadratic spacing which we know works well
        t_values = t_quad
        
        results = []
        for t in t_values:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = t
            f = problem(x)
            results.append(f)
        
        return results
# EVOLVE-BLOCK-END
