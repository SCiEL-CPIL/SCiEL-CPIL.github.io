import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        
        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)
        
        # --- Phase 1: Initialization ---
        # We will launch multiple Nelder-Mead optimizers from diverse points.
        # Number of starts: 5
        n_starts = 5
        
        # Generate initial points
        # 1 point at origin
        # n_starts - 1 points via LHS or random
        init_points = []
        init_points.append(np.zeros(dim))
        
        # LHS for the rest to ensure coverage
        n_lhs = n_starts - 1
        for i in range(n_lhs):
            x = np.zeros(dim)
            for d in range(dim):
                # Stratified sampling
                frac = (i + rng.random()) / n_lhs
                x[d] = -5.0 + 10.0 * frac
            init_points.append(x)

        # Evaluate initial points
        for x in init_points:
            if calls >= budget or not time_ok():
                break
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            record(x, f)

        if best_x is None:
            return best_x, best_f

        # --- Phase 2: Multi-Start Nelder-Mead ---
        # Each simplex is represented by a list of (point, value) pairs
        # We store them in a list of dicts: {'points': [x0, ..., xN], 'values': [f0, ..., fN]}
        # N = dim, so simplex has dim+1 points.
        
        active_simplexes = []
        
        # Initialize simplexes around each starting point
        # We already evaluated the center. We need to generate the other dim points.
        # To save budget, we can generate the simplex points on the fly and evaluate them.
        # However, NM requires a valid simplex to start.
        # Let's construct simplexes.
        
        for i in range(n_starts):
            if calls >= budget or not time_ok():
                break
                
            center = init_points[i]
            # We have already evaluated center.
            # We need dim more points to form the simplex.
            # Use a small step size for initial simplex, e.g., 0.5
            step = 0.5
            
            simplex_points = [center.copy()]
            simplex_values = [func(center)] # Wait, we already evaluated this. 
            # But `record` was called. We need to be careful not to double count.
            # Actually, `calls` was incremented. `func` was called.
            # We can reuse the value. But we don't have a map from x to f.
            # Let's just re-evaluate or store it. 
            # Better: Store the initial evaluation results.
            
            # Let's restructure: 
            # 1. Generate n_starts centers.
            # 2. Evaluate all centers.
            # 3. For each center, build simplex by evaluating dim neighbors.
            
            # This is getting complicated with the budget.
            # Alternative: 
            # Just start with a small (1+1)-ES to find a local minimum, then switch to NM?
            # No, NM is better if we have a good start.
            
            # Let's stick to the plan:
            # We have evaluated `init_points`. 
            # Let's assume we can afford to evaluate the simplex vertices.
            # Total cost: n_starts * (1 + dim) evaluations for initialization.
            # 5 * (1 + 5) = 30 evaluations. Budget is 500. This is fine.
            
            # But we already evaluated the centers. 
            # Let's just create a new simplex for each center.
            
            # We need to store the value of the center to avoid re-evaluation.
            # Since we don't have a cache, and `func` is a black box, we just re-evaluate?
            # No, that wastes budget. 
            # Let's modify the init loop to store values.
            
            pass # Placeholder, let's rewrite the init logic properly.
        
        # Reset and do it properly
        calls = 0
        best_x = None
        best_f = np.inf
        
        init_points = [np.zeros(dim)]
        for i in range(1, n_starts):
            x = np.zeros(dim)
            for d in range(dim):
                frac = (i + rng.random()) / n_starts
                x[d] = -5.0 + 10.0 * frac
            init_points.append(x)
        
        init_values = []
        for x in init_points:
            if calls >= budget or not time_ok():
                break
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            record(x, f)
            init_values.append(f)
            
        if not init_values:
            return best_x, best_f

        active_simplexes = []
        step_size = 0.5
        
        for i in range(n_starts):
            if calls >= budget or not time_ok():
                break
                
            center = init_points[i]
            f_center = init_values[i]
            
            # Build simplex
            # Point 0 is center
            simplex_points = [center.copy()]
            simplex_values = [f_center]
            
            # Generate dim neighbors
            for d in range(dim):
                x_neighbor = center.copy()
                # Perturb dimension d
                # If center is 0, step to 0.5. If center is positive, step to center + 0.5?
                # Standard NM: use a small step from the starting point.
                # If starting point is 0, neighbors are at +/- step? 
                # Usually, simplex is formed by moving along axes.
                # Let's move in positive direction for simplicity, or symmetric?
                # Standard: x_0 = x_start, x_i = x_start + step * e_i
                x_neighbor[d] += step_size
                x_neighbor = np.clip(x_neighbor, -5.0, 5.0)
                
                f_neighbor = func(x_neighbor)
                calls += 1
                record(x_neighbor, f_neighbor)
                
                simplex_points.append(x_neighbor.copy())
                simplex_values.append(f_neighbor)
                
                if calls >= budget or not time_ok():
                    break
            
            if len(simplex_points) == dim + 1:
                active_simplexes.append({
                    'points': simplex_points,
                    'values': simplex_values
                })
        
        if not active_simplexes:
            # Fallback to (1+1)-ES if NM init failed
            return self._fallback_es(func, best_x, best_f, calls, budget, time_ok, record, rng)

        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main Loop
        while calls < budget and time_ok() and active_simplexes:
            # Select Best-First Simplex
            best_si = 0
            best_val = np.inf
            for i, s in enumerate(active_simplexes):
                # Best value in this simplex
                s_best = min(s['values'])
                if s_best < best_val:
                    best_val = s_best
                    best_si = i
            
            simplex = active_simplexes[best_si]
            points = simplex['points']
            values = simplex['values']
            
            # Sort points by value (ascending)
            # We need to keep track of indices or just sort the lists
            # Sorting lists of numpy arrays is expensive. 
            # Let's use indices.
            order = np.argsort(values)
            sorted_points = [points[i] for i in order]
            sorted_values = [values[i] for i in order]
            
            # x_0 is best, x_N is worst
            x_0 = sorted_points[0]
            f_0 = sorted_values[0]
            x_N = sorted_points[dim]
            f_N = sorted_values[dim]
            
            # Centroid of all points except worst
            x_c = np.mean(sorted_points[:-1], axis=0)
            
            # 1. Reflection
            x_r = x_c + alpha * (x_c - x_N)
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = func(x_r)
            calls += 1
            record(x_r, f_r)
            
            new_points = sorted_points[:-1]
            new_values = sorted_values[:-1]
            
            if f_r < f_0:
                # 2. Expansion
                x_e = x_c + gamma * (x_r - x_N)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = func(x_e)
                calls += 1
                record(x_e, f_e)
                
                if f_e < f_r:
                    new_points.append(x_e)
                    new_values.append(f_e)
                else:
                    new_points.append(x_r)
                    new_values.append(f_r)
            elif f_r < f_N:
                # Accept Reflection
                new_points.append(x_r)
                new_values.append(f_r)
            else:
                # 3. Contraction
                # Check if it's an outside or inside contraction
                # If f_r >= f_N, it's outside contraction (x_r is worse than worst)
                # If f_r < f_N but f_r >= f_0, it's inside contraction? 
                # Standard NM:
                # If f_r >= f_N: Outside Contraction
                # If f_N > f_r >= f_0: Accept Reflection (handled above)
                # If f_r < f_0: Expansion (handled above)
                
                # So here f_r >= f_N
                x_c_out = x_c + rho * (x_r - x_N)
                x_c_out = np.clip(x_c_out, -5.0, 5.0)
                f_c_out = func(x_c_out)
                calls += 1
                record(x_c_out, f_c_out)
                
                if f_c_out < f_r:
                    new_points.append(x_c_out)
                    new_values.append(f_c_out)
                else:
                    # 4. Shrink
                    # Shrink all points except x_0 towards x_0
                    for i in range(1, len(new_points)):
                        x_shrink = x_0 + sigma * (new_points[i] - x_0)
                        x_shrink = np.clip(x_shrink, -5.0, 5.0)
                        f_shrink = func(x_shrink)
                        calls += 1
                        record(x_shrink, f_shrink)
                        new_points[i] = x_shrink
                        new_values[i] = f_shrink
                    
                    # Update simplex
                    simplex['points'] = new_points
                    simplex['values'] = new_values
                    
                    # Check convergence
                    # If simplex is very small, remove it
                    spread = np.max(np.std(new_points, axis=0))
                    if spread < 1e-6:
                        active_simplexes.pop(best_si)
                    continue # Skip normal update if we shrunk and converged? 
                    # Actually, we updated simplex in the else block. 
                    # We need to break out of the if/else chain properly.
                    # Let's refactor the control flow.
                    
            # If we didn't shrink/converge, update simplex
            simplex['points'] = new_points
            simplex['values'] = new_values
            
            # Check convergence
            spread = np.max(np.std(new_points, axis=0))
            if spread < 1e-6:
                active_simplexes.pop(best_si)

        # --- Phase 3: Final Refinement ---
        # If budget remains, do a quick (1+1)-ES on the global best
        if calls < budget and best_x is not None and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            
            while calls < budget and time_ok():
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = func(x_trial)
                calls += 1
                
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    sigma = min(sigma * 1.1, 0.5)
                    record(x, f)
                else:
                    sigma = max(sigma * 0.9, 1e-4)
                    if sigma < 1e-5:
                        break

        return best_x, best_f

    def _fallback_es(self, func, best_x, best_f, calls, budget, time_ok, record, rng):
        # Simple (1+1)-ES fallback
        if best_x is None:
            x = np.zeros(self.dim)
            f = func(x)
            calls += 1
            record(x, f)
        else:
            x = best_x.copy()
            f = best_f
        
        sigma = 1.0
        while calls < budget and time_ok():
            z = rng.standard_normal(self.dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5.0, 5.0)
            f_trial = func(x_trial)
            calls += 1
            
            if f_trial < f:
                x = x_trial
                f = f_trial
                sigma = min(sigma * 1.05, 5.0)
                record(x, f)
            else:
                sigma = max(sigma * 0.95, 1e-6)
        
        return x, f
