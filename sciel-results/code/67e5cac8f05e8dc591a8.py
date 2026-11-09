import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin for 300s limit
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf
        all_evaluated = []

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
            all_evaluated.append((val, x_clipped.copy()))
            return val

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        # Phase 1: Broad exploration with random points to find promising basins
        # We'll use 50 random points to get a good initial spread
        num_initial = 50
        initial_points = []
        for _ in range(num_initial):
            # Sample from a slightly smaller box to avoid boundary issues
            p = rng.uniform(-4.5, 4.5, n)
            initial_points.append(p)
        
        for p in initial_points:
            if not time_ok() or calls_made >= total_budget:
                break
            do_call(p)
        
        # Sort by function value and pick top 5 diverse points
        # Diversity: ensure they are not too close to each other
        initial_points_sorted = sorted(initial_points, key=lambda x: min(f for f, xp in all_evaluated if np.allclose(xp, x)))
        
        # Actually, let's just sort all_evaluated by value and pick top K diverse ones
        # Better approach: sort all evaluated points by value, then greedily pick diverse ones
        evaluated_sorted = sorted(all_evaluated, key=lambda x: x[0])
        
        selected_starts = []
        min_distance = 1.0  # Minimum distance between starts
        
        for f_val, x_val in evaluated_sorted:
            if len(selected_starts) >= 5:
                break
            # Check if this point is sufficiently far from all selected starts
            is_diverse = True
            for s in selected_starts:
                if np.linalg.norm(x_val - s) < min_distance:
                    is_diverse = False
                    break
            if is_diverse:
                selected_starts.append(x_val.copy())
        
        # If we don't have 5, fill with random points
        while len(selected_starts) < 5:
            selected_starts.append(rng.uniform(-4.5, 4.5, n))

        # Phase 2: Run 5 independent Nelder-Mead simplexes
        # Each simplex gets a portion of the remaining budget
        remaining_budget = total_budget - calls_made
        num_simplexes = len(selected_starts)
        
        # Allocate budget: give each simplex roughly equal share, but allow early termination
        budget_per_simplex = max(10, remaining_budget // num_simplexes)
        
        simplex_states = []
        for i in range(num_simplexes):
            center = selected_starts[i]
            step = 0.3  # Initial step size
            
            simplex = []
            values = []
            
            # Evaluate center
            f0 = do_call(center)
            simplex.append(center.copy())
            values.append(f0)
            
            # Evaluate n neighbors
            for j in range(n):
                pt = center.copy()
                if pt[j] + step > 5.0:
                    pt[j] = 5.0 - 0.05
                elif pt[j] - step < -5.0:
                    pt[j] = -5.0 + 0.05
                else:
                    pt[j] += step
                fj = do_call(pt)
                simplex.append(pt.copy())
                values.append(fj)
                
            simplex_states.append({
                'points': simplex,
                'values': values,
                'calls': n + 1,
                'converged': False
            })
        
        # If we ran out of budget during initialization
        if calls_made >= total_budget or not time_ok():
            return best_x, best_f

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_iter_per_simplex = 100
        
        # Round-robin optimization
        while calls_made < total_budget and time_ok():
            progress_made = False
            for idx, state in enumerate(simplex_states):
                if not time_ok() or calls_made >= total_budget:
                    break
                if state['converged'] or state['calls'] >= max_iter_per_simplex:
                    continue
                
                points = state['points']
                values = state['values']
                
                # Sort points by value
                order = np.argsort(values)
                points = [points[i] for i in order]
                values = [values[i] for i in order]
                
                # Check convergence
                if values[-1] - values[0] < 1e-6:
                    state['converged'] = True
                    continue
                
                # Centroid of all but worst
                centroid = np.sum(points[:-1], axis=0) / n
                worst_point = points[-1]
                
                # Reflection
                xr = centroid + alpha * (centroid - worst_point)
                xr = np.clip(xr, -5.0, 5.0)
                fr = do_call(xr)
                state['calls'] += 1
                
                if fr < values[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst_point)
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = do_call(xe)
                    state['calls'] += 1
                    
                    if fe < fr:
                        points[-1] = xe
                        values[-1] = fe
                    else:
                        points[-1] = xr
                        values[-1] = fr
                elif fr < values[-2]:
                    # Accept reflection
                    points[-1] = xr
                    values[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (worst_point - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = do_call(xc)
                    state['calls'] += 1
                    
                    if fc < fr:
                        points[-1] = xc
                        values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            points[i] = points[0] + sigma * (points[i] - points[0])
                            fi = do_call(points[i])
                            state['calls'] += 1
                            values[i] = fi
                
                state['points'] = points
                state['values'] = values
                progress_made = True
                
                if calls_made >= total_budget or not time_ok():
                    break
            
            # If no progress was made (all converged or budget exhausted), break
            if not progress_made:
                break
        
        return best_x, best_f
