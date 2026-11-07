"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def count_func(x):
            self._call_count += 1
            fx = func(x)
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        initial_step = 0.5

        def nelder_mead(start_x):
            """Run Nelder-Mead from start_x until budget is exhausted."""
            # Create initial simplex
            simplex = [start_x.copy()]
            f_values = [count_func(start_x)]
            
            # Add other vertices
            for i in range(self.dim):
                if self._call_count >= self.budget:
                    return
                vertex = start_x.copy()
                if vertex[i] < 4.5:
                    vertex[i] += initial_step
                else:
                    vertex[i] -= initial_step
                vertex = np.clip(vertex, -5, 5)
                f_val = count_func(vertex)
                simplex.append(vertex)
                f_values.append(f_val)
                
            if self._call_count >= self.budget:
                return

            # Main loop
            while self._call_count < self.budget:
                # Sort by fitness
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Check for convergence (optional, but good for efficiency)
                if max(f_values) - min(f_values) < 1e-8:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = count_func(x_ref)
                
                if self._call_count >= self.budget:
                    return

                # If reflection is better than worst but not best
                if f_values[0] <= f_ref < f_values[-2]:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                # If reflection is better than best
                elif f_ref < f_values[0]:
                    # Expansion
                    if self._call_count >= self.budget:
                        return
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = count_func(x_exp)
                    if self._call_count >= self.budget:
                        return
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                # If reflection is worse than worst
                else:
                    # Contraction
                    if self._call_count >= self.budget:
                        return
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = count_func(x_con)
                    if self._call_count >= self.budget:
                        return
                    if f_con < f_values[-1]:
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if self._call_count >= self.budget:
                                return
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = count_func(simplex[i])

        # Multi-start strategy
        # We have 500 budget. 
        # Let's do 5 restarts. Each restart gets ~100 evals.
        
        num_restarts = 5
        budget_per_restart = self.budget // num_restarts
        
        for r in range(num_restarts):
            if self._call_count >= self.budget:
                break
            
            # Choose start point
            if r == 0:
                # Start at origin or random
                start_x = rng.uniform(-2, 2, self.dim)
            else:
                # Random restart
                start_x = rng.uniform(-4.5, 4.5, self.dim)
            
            # Run Nelder-Mead
            nelder_mead(start_x)

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
