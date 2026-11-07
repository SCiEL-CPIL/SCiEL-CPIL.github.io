import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Define bounds
        lb = -5.0
        ub = 5.0
        
        # Number of parallel simplexes
        # Each simplex needs dim+1 points.
        # Let's try 5 simplexes. 5 * 6 = 30 initial calls.
        # This leaves 470 calls for steps.
        n_simplexes = 5
        
        # Store simplex data: list of dicts or arrays
        # Each simplex: points (dim+1, dim), values (dim+1,)
        simplexes = []
        
        # Global best
        best_f = float('inf')
        best_x = None
        
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
            
        # Initialize simplexes using Latin Hypercube Sampling for diversity
        # We need n_simplexes starting points, then expand to simplex.
        # LHS for n_simplexes points in dim dimensions.
        # Simple approach: stratified random.
        starts = []
        for i in range(n_simplexes):
            # Random point in the box, slightly shrunk to avoid boundaries initially?
            # Or just uniform.
            x0 = rng.uniform(lb, ub, dim)
            starts.append(x0)
            
        # Construct simplexes
        for i in range(n_simplexes):
            x0 = starts[i]
            # Create simplex points
            # Point 0 is x0
            # Points 1..dim are x0 + delta * e_j
            # Choose delta based on distance to bounds or fixed small step
            # Fixed step 0.5 is reasonable for [-5,5]
            delta = 0.5
            points = np.zeros((dim + 1, dim))
            points[0] = x0
            for j in range(dim):
                points[j + 1] = x0
                points[j + 1, j] += delta
                # Clip to bounds
                points[j + 1] = np.clip(points[j + 1], lb, ub)
            
            values = np.zeros(dim + 1)
            for k in range(dim + 1):
                val = evaluate(points[k])
                if val is None:
                    # Out of budget, return current best
                    return best_x, best_f
                values[k] = val
            
            simplexes.append({
                'points': points,
                'values': values,
                'active': True
            })
            
        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrink
        
        # Main Loop
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Cycle through active simplexes
            for idx in range(n_simplexes):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                s = simplexes[idx]
                if not s['active']:
                    continue
                    
                points = s['points']
                values = s['values']
                
                # Sort points by value
                order = np.argsort(values)
                points = points[order]
                values = values[order]
                s['points'] = points
                s['values'] = values
                
                # Check for convergence/stagnation
                # If the spread is very small, we might be stuck
                # Or if values are all very close
                val_range = values[-1] - values[0]
                if val_range < 1e-12:
                    s['active'] = False
                    continue
                    
                # Centroid of all but worst point
                centroid = np.mean(points[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - points[-1])
                x_r = np.clip(x_r, lb, ub)
                f_r = evaluate(x_r)
                if f_r is None:
                    break
                    
                if f_r < values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lb, ub)
                    f_e = evaluate(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        # Replace worst with expansion
                        points[-1] = x_e
                        values[-1] = f_e
                    else:
                        # Replace worst with reflection
                        points[-1] = x_r
                        values[-1] = f_r
                elif f_r < values[-2]:
                    # Replace worst with reflection
                    points[-1] = x_r
                    values[-1] = f_r
                else:
                    # Contraction
                    if f_r < values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, lb, ub)
                    f_c = evaluate(x_c)
                    if f_c is None:
                        break
                        
                    if f_c < min(f_r, values[-1]):
                        # Replace worst with contraction
                        points[-1] = x_c
                        values[-1] = f_c
                    else:
                        # Shrink
                        # Shrink all points except the best one towards the best point
                        for i in range(1, dim + 1):
                            points[i] = points[0] + sigma * (points[i] - points[0])
                            points[i] = np.clip(points[i], lb, ub)
                            val = evaluate(points[i])
                            if val is None:
                                break
                            values[i] = val
                        if val is None:
                            break
                        
                        # Re-sort after shrink? Usually shrink keeps order roughly, but let's be safe
                        # Actually, standard NM doesn't re-sort immediately after shrink, 
                        # but since we sort at the start of the loop, it's fine.
                        
                # Update stored points/values
                s['points'] = points
                s['values'] = values

        return best_x, best_f
