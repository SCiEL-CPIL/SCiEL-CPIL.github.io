"""CMA-ES optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        # 1. Initial Exploration to find a good starting point
        # The step ellipsoidal function has a flat minimum, so we need to be
        # careful not to get stuck in a "step" region that is far from the
        # true minimum but has the same value (if any). However, f7 is usually
        # defined such that the minimum is unique or at least the function
        # value increases as you move away from the center.
        # Actually, BBOB f7 is "Step Ellipsoidal". It is non-separable.
        # f(x) = 10 * max(0, |x - z|_inf - 1) + f(x - z) * 10? No, standard definition:
        # f(x) = 10 * max(0, |x - z|_inf - 1) + f_ellipsoidal(x - z)
        # Wait, the "Step" part creates a plateau or a jump.
        # Standard BBOB f7:
        # f(x) = 10 * max(0, |x - z|_inf - 1) + f(x - z)
        # Where f is the ellipsoidal function.
        # The term 10 * max(0, |x-z|_inf - 1) is 0 if x is within the box [-1,1] around z.
        # So inside the box, it's just the ellipsoidal function.
        # Outside, there's a constant offset of at least 10? No, it adds 10.
        # So the global minimum is inside the box.
        
        # We start with a few random points to get a sense of the scale and location.
        rng = np.random.default_rng(123)
        
        best_x = None
        best_f = np.inf
        n_init = min(10, budget // 10)
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if time.time() > deadline:
            return best_x, best_f

        # 2. Run CMA-ES
        # We use the 'cma' library if available because it is robust and handles
        # constraints and restarts well. If not, we implement a simple version.
        
        if HAS_CMA:
            # Configure CMA-ES
            # Population size: default is 4 + floor(3 ln n) -> 4 + 8 = 12 for n=5
            # We have ~480 calls left.
            # Each generation uses popsize evaluations.
            # 480 / 12 = 40 generations. This is quite few for CMA-ES to converge
            # from scratch, but we have a good starting point.
            
            # We can increase popsize if we want faster convergence in fewer generations?
            # No, larger popsize = more evals per gen.
            # We want to use all budget.
            
            popsize = 4 + int(3 * np.log(dim)) # 12
            # Allow larger popsize if budget permits?
            # Let's stick to standard.
            
            # Bounds: [-5, 5]
            # CMA-ES in 'cma' library handles bounds via resampling.
            
            opts = {
                'bounds': [-5, 5],
                'maxiter': 1000, # We will break manually based on budget
                'verbose': -9,
                'seed': 42,
                'CMA_popsize': popsize,
                # 'CMA_stopsamefit': 10, # Stop if no improvement for 10 gens?
                # 'CMA_maxfevals': budget - n_init,
            }
            
            # Create problem wrapper to track calls and enforce deadline
            class Problem:
                def __init__(self, func, deadline, budget_left):
                    self.func = func
                    self.deadline = deadline
                    self.calls = 0
                    self.budget_left = budget_left
                    self.best_f = best_f
                    self.best_x = best_x
                    self.stopped = False
                    
                def __call__(self, x):
                    if self.stopped:
                        return self.best_f
                    if time.time() > self.deadline:
                        self.stopped = True
                        return self.func(x) # One last eval? Or just return best?
                        # Better to just return a value and let cma stop.
                        # But we must not exceed budget.
                        
                    if self.calls >= self.budget_left:
                        self.stopped = True
                        return self.best_f
                    
                    fx = self.func(x)
                    self.calls += 1
                    if fx < self.best_f:
                        self.best_f = fx
                        self.best_x = x.copy()
                    return fx
                    
                def get_best(self):
                    return self.best_x, self.best_f

            problem = Problem(func, deadline, budget - n_init)
            
            try:
                es = cma.CMAEvolutionStrategy(best_x, 1.0, opts)
                while not es.stop() and not problem.stopped:
                    # Check time and budget before each generation
                    if time.time() > deadline or problem.calls >= problem.budget_left:
                        break
                        
                    X = es.ask()
                    fvals = [problem(x) for x in X]
                    
                    if problem.stopped:
                        break
                        
                    es.tell(X, fvals)
                    
            except Exception:
                # Fallback if cma library fails
                pass
            
            # After CMA-ES, if we have budget left, do a local refinement
            # CMA-ES might stop early due to convergence criteria.
            if problem.calls < problem.budget_left and time.time() < deadline:
                # Simple (1+1) ES refinement around the best found
                x = problem.best_x
                f = problem.best_f
                sigma = 0.1
                remaining = problem.budget_left - problem.calls
                
                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    problem.calls += 1
                    if fy < f:
                        x, f = y, fy
                        sigma = min(sigma * 1.5, 5.0)
                    else:
                        sigma = max(sigma * 0.7, 0.01)
                
                problem.best_x = x
                problem.best_f = f

            return problem.best_x, problem.best_f

        else:
            # Fallback: Manual CMA-ES-like or just robust (1+1) with restarts
            # Since 'cma' is standard in many environments, this is a backup.
            # We'll use a robust (1+1) ES with adaptive step and restarts.
            
            x = best_x
            f = best_f
            sigma = 1.0
            calls_left = budget - n_init
            
            # Simple (1+1) ES
            success_count = 0
            total_count = 0
            
            for _ in range(calls_left):
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < f:
                    x, f = y, fy
                    success_count += 1
                    sigma = min(sigma * 1.4, 10.0)
                else:
                    sigma = max(sigma * 0.7, 0.01)
                
                total_count += 1
                
                # Restart logic: if stuck for a long time, restart from best
                # with a large step
                if total_count > 50 and success_count < total_count * 0.1:
                    x = best_x + sigma * 2 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    success_count = 0
                    total_count = 0
                    sigma = 1.0

            return x, f

# EVOLVE-BLOCK-END
