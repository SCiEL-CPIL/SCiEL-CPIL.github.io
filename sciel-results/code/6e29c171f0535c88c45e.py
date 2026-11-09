import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def _halton(self, index, base):
        """Generate the Halton sequence for a given index and base."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i = i // base
        return result

    def _halton_sequence(self, count, dim, bases):
        """Generate a sequence of 'count' points in 'dim' dimensions."""
        points = np.zeros((count, dim))
        for i in range(1, count + 1):
            for j, base in enumerate(bases):
                points[i - 1, j] = self._halton(i, base)
        return points

    def _count_call(self, func, x):
        """Wrapper to count function calls."""
        self.calls += 1
        return func(x)

    def __call__(self, func):
        # Define bases for 5 dimensions (first 5 primes)
        bases = [2, 3, 5, 7, 11]
        
        # Phase 1: Exploration with Halton sequence
        # Use 150 points for exploration to ensure good coverage and early progress
        n_explore = min(150, self.budget)
        
        # Generate Halton points in [0, 1]^dim
        halton_points = self._halton_sequence(n_explore, self.dim, bases)
        
        # Map to [-5, 5]^dim
        halton_points = halton_points * 10.0 - 5.0
        
        best_x = None
        best_f = float('inf')
        
        for i in range(n_explore):
            if self.calls >= self.budget:
                break
            x = halton_points[i]
            fx = self._count_call(func, x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Exploitation with Nelder-Mead
        if self.calls < self.budget and best_x is not None:
            # Initialize Nelder-Mead simplex
            # Start with best_x and perturbations
            # Use a small initial step size
            delta = 0.2
            simplex = [best_x.copy()]
            for i in range(self.dim):
                v = best_x.copy()
                v[i] += delta
                simplex.append(v)
            
            # Evaluate initial simplex
            # We already have the value for best_x (simplex[0])
            fvals = [best_f]
            for i in range(1, len(simplex)):
                if self.calls >= self.budget:
                    break
                fx = self._count_call(func, simplex[i])
                fvals.append(fx)
            
            # If we ran out of budget during simplex init, return best found
            if self.calls >= self.budget:
                return best_x, best_f

            # Nelder-Mead Iteration
            # Parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            while self.calls < self.budget:
                # Sort simplex by function value
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]
                
                # 1. Reflection
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Centroid of best n points
                x_c = np.mean(simplex[:-1], axis=0)
                
                x_r = x_c + alpha * (x_c - x_worst)
                x_r = np.clip(x_r, -5, 5)
                
                if self.calls >= self.budget:
                    break
                f_r = self._count_call(func, x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = x_c + gamma * (x_r - x_c)
                    x_e = np.clip(x_e, -5, 5)
                    if self.calls >= self.budget:
                        break
                    f_e = self._count_call(func, x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Reflection accepted
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        x_c_r = x_c + rho * (x_r - x_c)
                    else:
                        x_c_r = x_c - rho * (x_worst - x_c)
                    
                    x_c_r = np.clip(x_c_r, -5, 5)
                    if self.calls >= self.budget:
                        break
                    f_c = self._count_call(func, x_c_r)
                    
                    if f_c < f_r:
                        simplex[-1] = x_c_r
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        # Shrink all points except the best one towards the best one
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if self.calls >= self.budget:
                                break
                            fvals[i] = self._count_call(func, simplex[i])
                        
                        if self.calls >= self.budget:
                            break

                # Update best
                current_best_f = min(fvals)
                if current_best_f < best_f:
                    best_f = current_best_f
                    best_idx = np.argmin(fvals)
                    best_x = simplex[best_idx].copy()

        return best_x, best_f
