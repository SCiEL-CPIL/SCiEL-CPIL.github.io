import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def _nelder_mead(self, func, x0, budget, tol=1e-8):
        """
        Nelder-Mead simplex optimization.
        Returns best_x, best_f.
        """
        dim = self.dim
        # Initialize simplex
        # x0 is the initial point
        # Create simplex: x0, x0 + delta * e_i
        delta = 0.1
        simplex = [x0.copy()]
        for i in range(dim):
            x = x0.copy()
            x[i] += delta
            simplex.append(x)
        
        # Evaluate initial simplex
        f_vals = []
        for x in simplex:
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            self.calls += 1
            f_vals.append(f)
            
            if f < self.best_f:
                self.best_f = f
                self.best_x = x_clipped.copy()
                
        if len(f_vals) < dim + 1:
            return self.best_x, self.best_f

        # Sort simplex by function value
        # We need to keep track of which x corresponds to which f
        # For efficiency, we'll maintain a list of (f, x) pairs
        simplex_points = [(f_vals[i], simplex[i]) for i in range(len(f_vals))]
        simplex_points.sort(key=lambda x: x[0])
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        while self.calls < self.budget and time.time() < self.deadline:
            # Check convergence
            f_best = simplex_points[0][0]
            f_worst = simplex_points[-1][0]
            if abs(f_worst - f_best) < tol:
                break
            
            # Calculate centroid of all points except the worst
            # Worst is the last one in the sorted list
            centroid = np.mean([p[1] for p in simplex_points[:-1]], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex_points[-1][1])
            x_ref = np.clip(x_ref, -5, 5)
            
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            f_ref = func(x_ref)
            self.calls += 1
            if f_ref < self.best_f:
                self.best_f = f_ref
                self.best_x = x_ref.copy()
            
            f_worst = simplex_points[-1][0]
            f_second_worst = simplex_points[-2][0]
            f_best = simplex_points[0][0]
            
            if f_ref < f_best:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                if self.calls >= self.budget or time.time() > self.deadline:
                    break
                f_exp = func(x_exp)
                self.calls += 1
                if f_exp < self.best_f:
                    self.best_f = f_exp
                    self.best_x = x_exp.copy()
                
                if f_exp < f_ref:
                    simplex_points[-1] = (f_exp, x_exp)
                else:
                    simplex_points[-1] = (f_ref, x_ref)
            elif f_ref < f_second_worst:
                # Accept reflection
                simplex_points[-1] = (f_ref, x_ref)
            else:
                # Contraction
                if f_ref < f_worst:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_ref - centroid)
                
                x_con = np.clip(x_con, -5, 5)
                if self.calls >= self.budget or time.time() > self.deadline:
                    break
                f_con = func(x_con)
                self.calls += 1
                if f_con < self.best_f:
                    self.best_f = f_con
                    self.best_x = x_con.copy()
                
                if f_con < f_worst:
                    simplex_points[-1] = (f_con, x_con)
                else:
                    # Shrink
                    x_worst = simplex_points[-1][1]
                    for i in range(1, len(simplex_points)):
                        simplex_points[i][1] = simplex_points[0][1] + sigma * (simplex_points[i][1] - simplex_points[0][1])
                        simplex_points[i][1] = np.clip(simplex_points[i][1], -5, 5)
                        
                        if self.calls >= self.budget or time.time() > self.deadline:
                            break
                        f_shrink = func(simplex_points[i][1])
                        self.calls += 1
                        if f_shrink < self.best_f:
                            self.best_f = f_shrink
                            self.best_x = simplex_points[i][1].copy()
                        simplex_points[i][0] = f_shrink
                        
                    if self.calls >= self.budget or time.time() > self.deadline:
                        break

            # Re-sort
            simplex_points.sort(key=lambda x: x[0])

        return self.best_x, self.best_f

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 280
        
        # Strategy: Random Restarts with Nelder-Mead
        # 1. Try a few random starts
        # 2. Try a start near the center
        # 3. Final refinement on the best point
        
        # Phase 1: 3 restarts, ~100 evals each
        restarts = 3
        budget_per_restart = 100
        
        # Initializations
        # 1. Center
        x0_center = np.zeros(self.dim)
        # 2. Random
        x0_rand1 = np.random.uniform(-5, 5, self.dim)
        # 3. Random
        x0_rand2 = np.random.uniform(-5, 5, self.dim)
        
        initializations = [x0_center, x0_rand1, x0_rand2]
        
        for i in range(restarts):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            
            x0 = initializations[i]
            self._nelder_mead(func, x0, self.budget)
            
        # Phase 2: Final refinement
        if self.calls < self.budget and time.time() < self.deadline and self.best_x is not None:
            # Use best point found so far
            # Add a small perturbation to escape if stuck?
            # Or just run NM again from best_x
            x0_best = self.best_x.copy()
            # Maybe perturb slightly to ensure simplex doesn't collapse if it already converged
            # But NM from a good point is fine.
            self._nelder_mead(func, x0_best, self.budget)
            
        return self.best_x, self.best_f
