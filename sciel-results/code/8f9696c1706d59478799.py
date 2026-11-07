import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        
        start_time = time.time()
        deadline = start_time + 290 # 10s safety margin
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        def wrap_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Multi-Start Initialization
        K = 5
        starts = []
        # Start 0: Origin
        starts.append(np.zeros(dim))
        # Starts 1-4: Random
        for _ in range(K-1):
            starts.append(rng.uniform(-5, 5, dim))
            
        # Evaluate starts
        for x in starts:
            wrap_func(x)
            
        # 2. Setup Nelder-Mead instances
        # We will run NM on the best K starting points
        # Sort starts by fitness (we don't have individual fitnesses stored separately, 
        # but we can just re-evaluate or assume the random ones are okay. 
        # Actually, we should track which start was best).
        
        # Let's re-evaluate the starts to get their fitnesses for sorting?
        # No, that wastes calls.
        # We can just pick the first K starts.
        # Or, we can keep a list of (x, f) from the evaluation.
        
        # Let's redo the initialization to track fitness
        # Reset calls? No, we already used K calls.
        # We need to know which of the K starts was best.
        # I will store them in a list.
        
        # Re-implementing init to track fitness
        # The code above already called wrap_func. 
        # I need to know the fitness of each start.
        # I can modify wrap_func to return f, and store it.
        
        # Let's restart the logic inside __call__ to be cleaner.
        
        # --- Revised Logic ---
        
        # 1. Generate K starts
        K = 5
        starts = [np.zeros(dim)] + [rng.uniform(-5, 5, dim) for _ in range(K-1)]
        
        # 2. Evaluate and sort
        start_stats = []
        for x in starts:
            fx = wrap_func(x)
            start_stats.append((fx, x))
            
        start_stats.sort(key=lambda item: item[0])
        
        # 3. Initialize NMs on the best K starts
        nms = []
        for fx, x0 in start_stats:
            # Create simplex
            simplex = [x0.copy()]
            for j in range(dim):
                xp = x0.copy()
                # Adaptive step size? Start with 1.0
                step = 1.0
                xp[j] += step
                # Clip to box
                xp = np.clip(xp, -5, 5)
                simplex.append(xp)
            
            simplex_f = [wrap_func(p) for p in simplex]
            
            nms.append({
                'simplex': simplex,
                'fitness': simplex_f,
                'stagnant': 0
            })

        # 4. Optimization Loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Cycle through NMs
            for nm in nms:
                if calls >= budget or time.time() > deadline:
                    break
                    
                self._nelder_mead_step(nm, wrap_func, dim, budget, calls)
                
                # Check for stagnation/restart
                # If simplex volume is tiny or no improvement for 20 steps
                # We can check the range of fitness in the simplex
                f_range = max(nm['fitness']) - min(nm['fitness'])
                if f_range < 1e-6:
                    nm['stagnant'] += 1
                else:
                    nm['stagnant'] = 0
                    
                if nm['stagnant'] > 20:
                    # Restart this NM with a new random point
                    x_new = rng.uniform(-5, 5, dim)
                    fx_new = wrap_func(x_new)
                    # Rebuild simplex
                    simplex = [x_new.copy()]
                    for j in range(dim):
                        xp = x_new.copy()
                        xp[j] += 1.0
                        xp = np.clip(xp, -5, 5)
                        simplex.append(xp)
                    simplex_f = [wrap_func(p) for p in simplex]
                    nm['simplex'] = simplex
                    nm['fitness'] = simplex_f
                    nm['stagnant'] = 0

        return best_x, best_f

    def _nelder_mead_step(self, nm, func, dim, budget, current_calls):
        # Perform one iteration of Nelder-Mead
        simplex = nm['simplex']
        fitness = nm['fitness']
        
        # Sort simplex
        indices = np.argsort(fitness)
        simplex = [simplex[i] for i in indices]
        fitness = [fitness[i] for i in indices]
        
        # Update nm
        nm['simplex'] = simplex
        nm['fitness'] = fitness
        
        # 1. Centroid of best D points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # 2. Reflection
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, -5, 5)
        fr = func(xr)
        
        if fr < fitness[0]:
            # Expand
            xe = centroid + 2 * (centroid - simplex[-1])
            xe = np.clip(xe, -5, 5)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fitness[-1] = fe
            else:
                simplex[-1] = xr
                fitness[-1] = fr
        elif fr < fitness[-2]:
            # Accept Reflection
            simplex[-1] = xr
            fitness[-1] = fr
        else:
            # Contract
            # Inside contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, -5, 5)
            fc = func(xc)
            if fc < fitness[-1]:
                simplex[-1] = xc
                fitness[-1] = fc
            else:
                # Shrink
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], -5, 5)
                    fitness[i] = func(simplex[i])
