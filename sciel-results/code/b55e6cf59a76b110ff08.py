import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        # Multi-start Nelder-Mead strategy
        # Initialize at origin
        x0 = np.zeros(d)
        try:
            f_x0 = safe_func(x0)
        except StopIteration:
            return best_x, best_f

        # Generate diverse starting points
        n_starts = 10
        starts = []
        for i in range(n_starts):
            # Mix of random points and points near known good regions
            if i == 0:
                x_start = x0.copy()
            elif i < 5:
                # Random in [-4, 4]
                x_start = rng.uniform(-4, 4, d)
            else:
                # Points near the typical Schwefel minimum region (4.20968)
                # but with some randomness
                x_start = 4.20968 + rng.normal(0, 1.0, d)
                x_start = np.clip(x_start, -5, 5)
            
            starts.append(x_start)
        
        # Track state for each start
        # Each entry: [simplex_points, simplex_fvals, best_f, best_x, step_size, active]
        states = []
        
        # Allocate initial budget: 50 calls per start, but we'll do it dynamically
        # First, evaluate all starts
        for x_start in starts:
            try:
                f = safe_func(x_start)
            except StopIteration:
                break
        
        # If we've used up budget, return
        if calls >= budget:
            return best_x, best_f
        
        # Initialize simplexes
        for i, x_start in enumerate(starts):
            simplex = [x_start.copy()]
            fvals = []
            try:
                f = safe_func(x_start)
                fvals.append(f)
            except StopIteration:
                break
            
            # Generate remaining simplex points
            for j in range(d):
                x_new = x_start.copy()
                # Perturb in direction j
                x_new[j] += 0.5 * (1 + abs(x_start[j]))
                x_new = np.clip(x_new, -5, 5)
                try:
                    f_new = safe_func(x_new)
                except StopIteration:
                    break
                simplex.append(x_new)
                fvals.append(f_new)
            
            if len(fvals) == d + 1:
                states.append({
                    'simplex': simplex,
                    'fvals': fvals,
                    'best_f': min(fvals),
                    'best_x': simplex[np.argmin(fvals)].copy(),
                    'active': True,
                    'improvements': 0,
                    'last_improvement_call': calls
                })
        
        # If no valid states, return
        if not states:
            return best_x, best_f
        
        # Main loop: run Nelder-Mead on active states
        # We'll use a round-robin approach with dynamic budget allocation
        while calls < budget and time.time() < deadline:
            # Get active states
            active = [s for s in states if s['active']]
            if not active:
                break
            
            # Select which state to update based on convergence rate
            # States that are improving get more budget
            # Simple approach: round-robin among active states
            for state in active:
                if calls >= budget or time.time() > deadline:
                    break
                
                simplex = state['simplex']
                fvals = state['fvals']
                
                # Sort simplex by fvals
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = fvals[-1]
                
                # Reflection
                alpha = 1.0
                x_reflect = centroid + alpha * (centroid - worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                
                try:
                    f_reflect = safe_func(x_reflect)
                except StopIteration:
                    break
                
                if f_reflect < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_expand = centroid + gamma * (centroid - worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    try:
                        f_expand = safe_func(x_expand)
                    except StopIteration:
                        # Use reflection
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                        break
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    # Reflection
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contraction
                    rho = 0.5
                    x_contract = centroid + rho * (worst - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    try:
                        f_contract = safe_func(x_contract)
                    except StopIteration:
                        break
                    
                    if f_contract < worst_f:
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            try:
                                fvals[i] = safe_func(simplex[i])
                            except StopIteration:
                                break
                        
                        if calls >= budget:
                            break
                
                # Update state
                state['simplex'] = simplex
                state['fvals'] = fvals
                min_f = min(fvals)
                min_idx = np.argmin(fvals)
                
                if min_f < state['best_f']:
                    state['best_f'] = min_f
                    state['best_x'] = simplex[min_idx].copy()
                    state['improvements'] += 1
                    state['last_improvement_call'] = calls
                
                # Check for convergence
                if max(fvals) - min(fvals) < 1e-8:
                    state['active'] = False
        
        # Final polish with (1+1)-ES on the best found
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.05
            while calls < budget and time.time() < deadline:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                x_candidate = np.clip(x_candidate, -5, 5)
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break

        return best_x, best_f
