import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # 280s safety margin out of 300s limit

        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            f_val = func(x)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)

        # Phase 1: Initial random sampling to find a good starting region
        # Use ~15% of budget for initialization
        n_init = min(50, self.budget // 6)
        init_points = []
        init_vals = []

        for _ in range(n_init):
            if time.time() > deadline or self.calls >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = safe_func(x)
            init_points.append(x)
            init_vals.append(f)

        if not init_points:
            x = np.zeros(self.dim)
            safe_func(x)
            return self.best_x, self.best_f

        # Find the best initial point
        best_idx = np.argmin(init_vals)
        x0 = init_points[best_idx].copy()

        # Phase 2: Deterministic Nelder-Mead simplex method
        # Initialize with a vertex at origin and others offset along axes by large magnitude
        # This provides immediate geometric progress in the first few calls
        if self.calls < self.budget - 10 and time.time() < deadline:
            # Build initial simplex: v0 = best point, v_i = v0 + 4.0 * e_i
            simplex = [x0.copy()]
            for i in range(self.dim):
                v = x0.copy()
                v[i] += 4.0
                simplex.append(v)
            
            # Evaluate all vertices (v0 already evaluated, but we need consistent tracking)
            # Actually, let's just run a custom Nelder-Mead implementation
            # to have full control over the budget and time limits
            
            def nelder_mead(func_eval, x0, dim, max_calls, deadline):
                # Initialize simplex
                # v0 = x0, v_i = x0 + alpha * e_i
                alpha = 4.0
                simplex = [x0.copy()]
                for i in range(dim):
                    v = x0.copy()
                    v[i] += alpha
                    simplex.append(v)
                
                # Evaluate all vertices
                values = []
                for i, v in enumerate(simplex):
                    if max_calls <= 0 or time.time() > deadline:
                        break
                    f_val = func_eval(v)
                    values.append(f_val)
                    max_calls -= 1
                
                # If we couldn't evaluate all, return what we have
                if len(values) < len(simplex):
                    # Pad with infinity
                    while len(values) < len(simplex):
                        values.append(float('inf'))
                
                # Nelder-Mead parameters
                gamma = 1.0  # reflection
                rho = 2.0    # expansion
                sigma = 0.5  # contraction
                xi = 0.5     # shrink
                
                while max_calls > 0 and time.time() <= deadline:
                    # Sort simplex by function value
                    idx = np.argsort(values)
                    simplex = [simplex[i] for i in idx]
                    values = [values[i] for i in idx]
                    
                    # Check convergence
                    if len(values) >= 2:
                        if abs(values[-1] - values[0]) < 1e-10:
                            break
                    
                    # Compute centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + gamma * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = func_eval(xr)
                    max_calls -= 1
                    
                    if fr < values[0]:
                        # Expansion
                        xe = centroid + rho * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        if max_calls > 0 and time.time() <= deadline:
                            fe = func_eval(xe)
                            max_calls -= 1
                            if fe < fr:
                                simplex[-1] = xe
                                values[-1] = fe
                            else:
                                simplex[-1] = xr
                                values[-1] = fr
                        else:
                            simplex[-1] = xr
                            values[-1] = fr
                    elif fr < values[-2]:
                        # Accept reflection
                        simplex[-1] = xr
                        values[-1] = fr
                    else:
                        # Contraction
                        xc = centroid + sigma * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        if max_calls > 0 and time.time() <= deadline:
                            fc = func_eval(xc)
                            max_calls -= 1
                            if fc <= fr:
                                simplex[-1] = xc
                                values[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, len(simplex)):
                                    simplex[i] = simplex[0] + xi * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], -5, 5)
                                    if max_calls > 0 and time.time() <= deadline:
                                        values[i] = func_eval(simplex[i])
                                        max_calls -= 1
                                    else:
                                        values[i] = float('inf')
                                break
                        else:
                            break
                
                return simplex[0], values[0]

            nm_budget = min(200, self.budget - self.calls - 10)
            try:
                nm_x, nm_f = nelder_mead(safe_func, x0, self.dim, nm_budget, deadline)
                if nm_f < self.best_f:
                    self.best_f = nm_f
                    self.best_x = nm_x.copy()
            except Exception:
                pass

        # Phase 3: Multi-scale coordinate descent with geometrically shrinking steps
        # This provides predictable early progress on rotated landscapes
        step_sizes = [10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 
                      0.0390625, 0.01953125, 0.009765625, 0.0048828125, 
                      0.00244140625, 0.001220703125, 0.0006103515625, 
                      0.00030517578125, 0.000152587890625, 7.62939453125e-05,
                      3.814697265625e-05, 1.9073486328125e-05, 9.5367431640625e-06,
                      4.76837158203125e-06, 2.384185791015625e-06, 1.1920928955078125e-06,
                      5.9604644775390625e-07, 2.9802322387695312e-07, 1.4901161193847656e-07,
                      7.450580596923828e-08, 3.725290298461914e-08, 1.862645149230957e-08]
        
        dim_order = list(range(self.dim))
        
        current_x = self.best_x.copy() if self.best_x is not None else x0
        current_f = self.best_f
        
        for scale_idx, step in enumerate(step_sizes):
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            # Rotate the dimension order for this scale
            offset = scale_idx % self.dim
            rotated_dims = dim_order[offset:] + dim_order[:offset]
            
            for d in rotated_dims:
                if self.calls >= self.budget or time.time() > deadline:
                    break
                
                # Try positive and negative steps
                for direction in [1, -1]:
                    if self.calls >= self.budget or time.time() > deadline:
                        break
                    
                    new_x = current_x.copy()
                    new_x[d] += direction * step
                    new_x = np.clip(new_x, -5, 5)
                    
                    f_new = safe_func(new_x)
                    
                    if f_new < current_f:
                        current_x = new_x
                        current_f = f_new
        
        # Phase 4: Random restarts with local refinement to escape local minima
        if self.calls < self.budget:
            remaining_budget = self.budget - self.calls
            # Each restart uses ~30 calls for local search
            n_restarts = min(8, remaining_budget // 30)
            
            for r in range(n_restarts):
                if self.calls >= self.budget or time.time() > deadline:
                    break
                
                # Random restart point
                x_r = rng.uniform(-5, 5, self.dim)
                f_r = safe_func(x_r)
                
                if f_r < self.best_f:
                    current_x = x_r.copy()
                    current_f = f_r
                else:
                    current_x = self.best_x.copy()
                    current_f = self.best_f
                
                # Small local search around this point
                local_steps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]
                for step in local_steps:
                    if self.calls >= self.budget or time.time() > deadline:
                        break
                    
                    for d in range(self.dim):
                        if self.calls >= self.budget or time.time() > deadline:
                            break
                        
                        for direction in [1, -1]:
                            if self.calls >= self.budget or time.time() > deadline:
                                break
                            
                            new_x = current_x.copy()
                            new_x[d] += direction * step
                            new_x = np.clip(new_x, -5, 5)
                            
                            f_new = safe_func(new_x)
                            
                            if f_new < current_f:
                                current_x = new_x
                                current_f = f_new
        
        return self.best_x, self.best_f
