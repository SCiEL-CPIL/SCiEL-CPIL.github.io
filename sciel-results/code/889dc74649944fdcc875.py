"""Hybrid Optimizer: LHS Exploration + Nelder-Mead Exploitation for BBOB f19."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Latin Hypercube Sampling for global exploration
    followed by Nelder-Mead for local exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # 1. Global Exploration with LHS
        # Use a small number of LHS points to cover the space efficiently.
        n_lhs = 20
        x_lhs = self._generate_lhs(n_lhs, self.dim, -5.0, 5.0)
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        for x in x_lhs:
            if time.time() > deadline or calls_used >= self.budget:
                break
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # 2. Local Exploitation with Nelder-Mead
        # If we still have budget, refine the best point found.
        if best_x is not None and calls_used < self.budget:
            remaining_budget = self.budget - calls_used
            best_x, best_f = self._nelder_mead(func, best_x, best_f, remaining_budget, deadline)

        return best_x, best_f

    def _generate_lhs(self, n, dim, low, high):
        """Generate n points using Latin Hypercube Sampling."""
        # Create a grid of indices
        indices = np.array([np.random.permutation(n) for _ in range(dim)])
        # Normalize to [0, 1)
        x = (indices + 0.5) / n
        # Scale to [low, high]
        x = low + x * (high - low)
        return x.T

    def _nelder_mead(self, func, x0, f0, budget, deadline):
        """
        Implement Nelder-Mead simplex algorithm.
        x0: initial point
        f0: function value at x0
        budget: max number of function evaluations allowed
        deadline: time limit
        """
        dim = self.dim
        # Initialize simplex
        # Start with x0, then create other vertices by perturbing one dimension at a time
        simplex = [x0.copy()]
        f_vals = [f0]
        
        # Initial step size for simplex expansion
        step = 0.1
        for i in range(dim):
            x_new = x0.copy()
            x_new[i] += step
            # Clip to box
            x_new = np.clip(x_new, -5, 5)
            simplex.append(x_new)
            f_new = func(x_new)
            f_vals.append(f_new)
            budget -= 1
            
            if budget <= 0 or time.time() > deadline:
                break

        # Ensure we have a valid simplex (dim+1 points)
        # If budget ran out during init, we just return the best so far
        if len(simplex) < dim + 1:
            best_idx = np.argmin(f_vals)
            return simplex[best_idx], f_vals[best_idx]

        # Main Nelder-Mead loop
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        while budget > 0 and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            # Check convergence
            if len(f_vals) > 1 and abs(f_vals[0] - f_vals[-1]) < 1e-10:
                break

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Worst point
            x_worst = simplex[-1]
            f_worst = f_vals[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            budget -= 1
            if budget <= 0 or time.time() > deadline:
                if f_r < f_worst:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                break

            if f_r < f_vals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                budget -= 1
                if budget <= 0 or time.time() > deadline:
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                    break
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_vals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                budget -= 1
                if budget <= 0 or time.time() > deadline:
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    break

                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    # Shrinkage
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = func(simplex[i])
                        budget -= 1
                        if budget <= 0 or time.time() > deadline:
                            break
                    
                    if budget <= 0 or time.time() > deadline:
                        break

        # Return the best point
        best_idx = np.argmin(f_vals)
        return simplex[best_idx], f_vals[best_idx]
# EVOLVE-BLOCK-END
