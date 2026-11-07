import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0 # 5s safety margin
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        best_x_global = None
        best_fx_global = np.inf
        calls = 0
        
        # Helper to evaluate and update global best
        def evaluate(x):
            nonlocal calls, best_fx_global, best_x_global
            if calls >= budget:
                return np.inf
            fx = func(x)
            calls += 1
            if fx < best_fx_global:
                best_fx_global = fx
                best_x_global = x.copy()
            return fx

        # Nelder-Mead Simplex Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # Initial Step Size for Simplex
        initial_sigma = 1.0
        
        # Determine number of restarts based on budget
        # A standard NM in 5D takes roughly 2-3 * (dim+1) * dim evaluations to converge locally.
        # Let's allocate a fixed budget per restart to ensure diversity.
        # With budget 500, we can afford about 5-10 restarts if we use ~50-100 evals each.
        # However, early progress is key for AOCC. 
        # Strategy: Start with a few random points, then run NM from the best.
        # If NM stagnates, restart from a new random point or perturb the current best.
        
        # Phase 1: Initial Exploration
        # Evaluate a small set of random points to find a good starting basin.
        n_initial = 10
        initial_points = []
        for _ in range(n_initial):
            if calls >= budget: break
            x = np.random.uniform(-4, 4, dim)
            fx = evaluate(x)
            initial_points.append((fx, x))
        
        if not initial_points:
            x0 = np.zeros(dim)
            evaluate(x0)
            return best_x_global, best_fx_global
        
        # Sort initial points by fitness
        initial_points.sort(key=lambda t: t[0])
        
        # Phase 2: Iterative Nelder-Mead with Restarts
        # We will run NM instances. 
        # To maximize AOCC, we want fast convergence in the first few hundred calls.
        # We'll use a "budget per instance" approach.
        # If an instance doesn't improve significantly, we restart.
        
        # List of starting points to try. 
        # 1. The best initial point.
        # 2. Other good initial points.
        # 3. Random restarts.
        
        # We'll manage a queue of start points.
        start_queue = [p[1].copy() for p in initial_points[:5]]
        
        # Also add a few random points to the queue for later diversity
        for _ in range(5):
            start_queue.append(np.random.uniform(-4, 4, dim))
            
        idx = 0
        while calls < budget and time.time() < deadline:
            if idx >= len(start_queue):
                # Generate more random starts if needed
                start_queue.append(np.random.uniform(-4, 4, dim))
                
            x_start = start_queue.pop(0)
            
            # Run Nelder-Mead from x_start
            # Initialize Simplex
            # Vertex 0: x_start
            # Vertices 1..N: x_start + initial_sigma * e_i
            simplex = [x_start.copy()]
            for i in range(dim):
                v = x_start.copy()
                v[i] += initial_sigma
                simplex.append(v)
                
            # Evaluate initial simplex
            f_values = []
            for i in range(dim + 1):
                if calls >= budget or time.time() > deadline:
                    break
                fx = evaluate(simplex[i])
                f_values.append(fx)
                
            if len(f_values) < dim + 1:
                break # Out of budget during init
                
            # Main NM Loop
            # We limit the number of iterations per NM instance to avoid getting stuck
            # in a local minimum for too long, allowing other basins to be explored.
            # However, since we have a global best tracker, we can let it run until
            # it stops improving or budget runs out.
            # To ensure we explore other basins, we'll set a soft limit on evals per instance
            # if it doesn't find a new global best quickly.
            
            local_best_fx = min(f_values)
            local_best_x = simplex[np.argmin(f_values)].copy()
            evals_in_this_run = dim + 1
            max_evals_per_run = 100 # Cap per instance to ensure diversity
            
            while calls < budget and time.time() < deadline and evals_in_this_run < max_evals_per_run:
                # Sort simplex by fitness
                order = np.argsort(f_values)
                simplex = [simplex[i].copy() for i in order]
                f_values = [f_values[i] for i in order]
                
                # Check for convergence (simplex size)
                centroid = np.mean(simplex[:-1], axis=0)
                # Simple convergence check: if the best point is very close to centroid
                # or if the range of function values is very small
                if np.max(f_values) - np.min(f_values) < 1e-10:
                    break
                    
                # Calculate centroid of all points except the worst
                # centroid = np.mean(simplex[:-1], axis=0) # Already calculated above
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                fx_r = evaluate(x_r)
                evals_in_this_run += 1
                
                if fx_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    fx_e = evaluate(x_e)
                    evals_in_this_run += 1
                    
                    if fx_e < fx_r:
                        simplex[-1] = x_e
                        f_values[-1] = fx_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = fx_r
                elif fx_r < f_values[-2]:
                    # Accept Reflection
                    simplex[-1] = x_r
                    f_values[-1] = fx_r
                else:
                    # Contraction
                    # Outside Contraction if fx_r > f_values[-1]
                    if fx_r < f_values[-1]:
                        # Inside Contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Outside Contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        
                    x_c = np.clip(x_c, -5, 5)
                    fx_c = evaluate(x_c)
                    evals_in_this_run += 1
                    
                    if fx_c < min(fx_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = fx_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fx_s = evaluate(simplex[i])
                            f_values[i] = fx_s
                            evals_in_this_run += 1
                            if calls >= budget or time.time() > deadline:
                                break
                        if calls >= budget or time.time() > deadline:
                            break

            # After NM run, if we haven't improved global best significantly,
            # the next iteration will pick a new start point from the queue.

        return best_x_global, best_fx_global
