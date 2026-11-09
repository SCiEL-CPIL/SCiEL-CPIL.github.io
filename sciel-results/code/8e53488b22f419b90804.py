import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit. Use 315s for actual work to be safe.
        deadline = start_time + 315
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.full(n, np.nan)
        best_f = np.inf

        # Shared state for the ensemble
        # We will run 5 independent Nelder-Mead instances.
        # Each gets a share of the budget.
        num_simplexes = 5
        budget_per_simplex = total_budget // num_simplex
        
        # Initialize simplex starting points
        # 1. Center
        # 2. 4 corners of the 5D box (selected for diversity)
        # Corners in 5D: we can pick points like (5,5,5,5,5), (-5,-5,-5,-5,-5), etc.
        # To be more diverse, let's use:
        # P0: (0,0,0,0,0)
        # P1: (5,5,5,5,5)
        # P2: (-5,-5,-5,-5,-5)
        # P3: (5,-5,5,-5,5)
        # P4: (-5,5,-5,5,-5)
        
        start_points = [
            np.zeros(n),
            np.array([5.0]*n),
            np.array([-5.0]*n),
            np.array([5.0, -5.0, 5.0, -5.0, 5.0]),
            np.array([-5.0, 5.0, -5.0, 5.0, -5.0])
        ]
        
        # Ensure start points are within bounds
        for i in range(len(start_points)):
            start_points[i] = np.clip(start_points[i], -5.0, 5.0)

        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        # We will iterate through the simplexes in a round-robin fashion?
        # Or run them sequentially but with strict budget limits?
        # "Parallel asynchronous" is hard in a single thread. 
        # However, we can simulate parallelism by alternating steps between simplexes.
        # This allows us to keep multiple basins under exploration simultaneously.
        
        # State for each simplex: current best point, current simplex vertices?
        # Since we use scipy.optimize.minimize, it manages the simplex internally.
        # We cannot easily "pause" and "resume" scipy's minimize.
        
        # Alternative Strategy: 
        # Instead of pausing/resuming, we run a **Multi-Start** approach but with a twist:
        # We allocate a small budget to each start point to find a local optimum, 
        # then re-allocate remaining budget to the most promising ones (or all of them) for refinement.
        
        # Actually, the prompt suggests "Parallel asynchronous Nelder-Mead simplex ensemble".
        # In a single-threaded context, the best approximation is **Round-Robin Nelder-Mead**.
        # We maintain a queue of active optimizers.
        
        # However, scipy's minimize is a blocking call. 
        # To implement round-robin, we would need to implement our own Nelder-Mead step.
        
        # Let's implement a lightweight, step-wise Nelder-Mead for the ensemble.
        
        class SimplexState:
            def __init__(self, x0, budget):
                self.x0 = x0
                self.budget_remaining = budget
                self.vertices = []
                self.vertex_fvals = []
                self.active = True
                self.init()
                
            def init(self):
                # Initialize simplex
                # Standard NM simplex: x0, and x0 + delta * e_i
                delta = 0.5 # Initial step size
                self.vertices = [self.x0.copy()]
                self.vertex_fvals = [do_call(self.x0)]
                
                for i in range(n):
                    x_new = self.x0.copy()
                    x_new[i] += delta
                    self.vertices.append(x_new)
                    self.vertex_fvals.append(do_call(x_new))
                    
                # Sort vertices
                self._sort()
                
            def _sort(self):
                # Sort vertices by function value
                combined = list(zip(self.vertex_fvals, self.vertices))
                combined.sort(key=lambda x: x[0])
                self.vertex_fvals = [x[0] for x in combined]
                self.vertices = [x[1] for x in combined]
                
            def step(self):
                if not self.active or self.budget_remaining <= 0:
                    return False
                
                # Check time
                if not time_ok():
                    self.active = False
                    return False
                
                # Standard NM Step
                # 1. Reflection
                x_worst = self.vertices[-1]
                f_worst = self.vertex_fvals[-1]
                
                # Centroid of all but worst
                centroid = np.mean(self.vertices[:-1], axis=0)
                alpha = 1.0
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                
                f_r = do_call(x_r)
                self.budget_remaining -= 1
                
                if f_r < self.vertex_fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    if self.budget_remaining > 0:
                        f_e = do_call(x_e)
                        self.budget_remaining -= 1
                        if f_e < f_r:
                            x_new = x_e
                            f_new = f_e
                        else:
                            x_new = x_r
                            f_new = f_r
                    else:
                        x_new = x_r
                        f_new = f_r
                        
                    self.vertices[-1] = x_new
                    self.vertex_fvals[-1] = f_new
                elif f_r < self.vertex_fvals[-2]:
                    # Reflection accepted
                    self.vertices[-1] = x_r
                    self.vertex_fvals[-1] = f_r
                else:
                    # Contraction
                    beta = 0.5
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + beta * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - beta * (x_r - centroid)
                        
                    x_c = np.clip(x_c, -5.0, 5.0)
                    if self.budget_remaining > 0:
                        f_c = do_call(x_c)
                        self.budget_remaining -= 1
                        if f_c < min(f_r, f_worst):
                            self.vertices[-1] = x_c
                            self.vertex_fvals[-1] = f_c
                        else:
                            # Shrink
                            self._shrink()
                    else:
                        self._shrink()
                        
                self._sort()
                return True
                
            def _shrink(self):
                x_best = self.vertices[0]
                for i in range(1, n+1):
                    x_s = x_best + 0.5 * (self.vertices[i] - x_best)
                    x_s = np.clip(x_s, -5.0, 5.0)
                    if self.budget_remaining > 0:
                        f_s = do_call(x_s)
                        self.budget_remaining -= 1
                        self.vertices[i] = x_s
                        self.vertex_fvals[i] = f_s
                    else:
                        break

        # Create ensemble
        simplexes = []
        for i in range(num_simplexes):
            if calls_made >= total_budget:
                break
            # Give each simplex an initial budget
            # We'll manage global budget via do_call, but local budget for logic
            # Let's just let them run until global budget or time is out.
            # To prevent one simplex from eating all budget, we can limit steps per round-robin turn.
            
            # Initialize simplex (costs n+1 calls)
            if calls_made + n + 1 <= total_budget:
                try:
                    s = SimplexState(start_points[i], total_budget)
                    simplexes.append(s)
                except Exception:
                    pass

        # Round-Robin Loop
        # Each simplex gets 1 step (which may use 1-3 calls) per turn.
        # We continue until budget or time is out.
        
        while calls_made < total_budget and time_ok():
            progress = False
            for s in simplexes:
                if calls_made >= total_budget or not time_ok():
                    break
                if s.active and s.budget_remaining > 0:
                    s.step()
                    progress = True
                    
            if not progress:
                break

        return best_x, best_f

