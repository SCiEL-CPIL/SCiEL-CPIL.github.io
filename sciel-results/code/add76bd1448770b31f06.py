"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start Nelder-Mead with adaptive restarts.
    Uses a deadline guard to ensure we finish within the time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, sigma, max_iter, deadline, state):
        """
        Perform Nelder-Mead optimization starting from x0.
        state: dict with 'best_x', 'best_f', 'calls'
        """
        dim = self.dim
        # Initialize simplex
        simplex = [x0.copy() + sigma * np.random.default_rng(state['rng_seed']).standard_normal(dim) for _ in range(dim)]
        simplex[0] = x0.copy()
        
        # Evaluate initial simplex
        f_values = []
        for i in range(dim + 1):
            if state['calls'] >= self.budget or time.time() > deadline:
                break
            fx = func(simplex[i])
            f_values.append(fx)
            state['calls'] += 1
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = simplex[i].copy()
        
        if len(f_values) < dim + 1:
            return
        
        # Sort simplex
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_shrink = 0.5  # shrink
        
        for _ in range(max_iter):
            if state['calls'] >= self.budget or time.time() > deadline:
                break
            
            # Check convergence
            if abs(f_values[0] - f_values[-1]) < 1e-8:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            
            if state['calls'] >= self.budget or time.time() > deadline:
                break
            f_ref = func(x_ref)
            state['calls'] += 1
            if f_ref < state['best_f']:
                state['best_f'] = f_ref
                state['best_x'] = x_ref.copy()
            
            if f_ref < f_values[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                if state['calls'] >= self.budget or time.time() > deadline:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                    break
                f_exp = func(x_exp)
                state['calls'] += 1
                if f_exp < state['best_f']:
                    state['best_f'] = f_exp
                    state['best_x'] = x_exp.copy()
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_values[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
            elif f_ref < f_values[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                f_values[-1] = f_ref
            else:
                # Contraction
                if f_ref < f_values[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_ref - centroid)
                x_con = np.clip(x_con, -5, 5)
                
                if state['calls'] >= self.budget or time.time() > deadline:
                    break
                f_con = func(x_con)
                state['calls'] += 1
                if f_con < state['best_f']:
                    state['best_f'] = f_con
                    state['best_x'] = x_con.copy()
                
                if f_con < min(f_ref, f_values[-1]):
                    simplex[-1] = x_con
                    f_values[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        if state['calls'] >= self.budget or time.time() > deadline:
                            break
                        f_values[i] = func(simplex[i])
                        state['calls'] += 1
                        if f_values[i] < state['best_f']:
                            state['best_f'] = f_values[i]
                            state['best_x'] = simplex[i].copy()
                    if state['calls'] >= self.budget or time.time() > deadline:
                        break
            
            # Re-sort simplex
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
        
        return state['best_x'], state['best_f']

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10 second safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        state = {
            'best_x': None,
            'best_f': float('inf'),
            'calls': 0,
            'rng_seed': 42
        }
        
        # Multi-start strategy
        # Start with a few initial points spread across the domain
        num_starts = 5
        max_iter_per_start = 200
        
        # First, do a coarse random search to find promising regions
        # Use 20% of budget for random sampling
        random_budget = min(50, self.budget // 5)
        
        for _ in range(random_budget):
            if state['calls'] >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            state['calls'] += 1
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x.copy()
        
        if state['best_x'] is None:
            # Fallback: just return the first point
            x = np.zeros(self.dim)
            fx = func(x)
            state['calls'] += 1
            state['best_x'] = x
            state['best_f'] = fx
            return x, fx
        
        # Now do local refinement from the best point found
        # And from a few other good points
        
        # Sort the random points by fitness (we didn't store them, so just use best)
        # Do Nelder-Mead from the best point
        x0 = state['best_x'].copy()
        self._nelder_mead(func, x0, 1.0, max_iter_per_start, deadline, state)
        
        # If we have budget left, try from a few other random points
        remaining_starts = 3
        for _ in range(remaining_starts):
            if state['calls'] >= self.budget or time.time() > deadline:
                break
            
            # Generate a new random starting point
            x0 = rng.uniform(-5, 5, self.dim)
            fx = func(x0)
            state['calls'] += 1
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x0.copy()
            
            # Do Nelder-Mead from this point
            self._nelder_mead(func, x0, 1.0, max_iter_per_start, deadline, state)
        
        # Final polish: one more Nelder-Mead from the global best with smaller sigma
        if state['calls'] < self.budget and time.time() < deadline:
            self._nelder_mead(func, state['best_x'].copy(), 0.1, 100, deadline, state)
        
        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
