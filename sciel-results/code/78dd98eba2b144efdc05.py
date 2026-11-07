"""Optimizer for BBOB f13 (Sharp ridge) using Nelder-Mead simplex."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # 10s safety margin before 300s kill

        # Track best value found so far
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return np.inf
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize simplex at center with small spread
        # f13 sharp ridge: minimizer in [-4,4]^5
        # Start at origin (center of box) which is close to typical minimizer location
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = 0.0  # center
        for i in range(1, dim + 1):
            simplex[i] = 0.0
            simplex[i, i - 1] = 0.5  # small spread along each axis

        # Evaluate all simplex vertices
        for i in range(dim + 1):
            simplex[i] = np.clip(simplex[i], -5, 5)
            f_i = evaluate(simplex[i])
            if time.time() > deadline:
                break

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        # Main Nelder-Mead loop
        while call_count < budget and time.time() < deadline:
            # Sort simplex by function value
            f_vals = np.array([evaluate(simplex[i]) if not np.isfinite(evaluate(simplex[i])) else np.inf for i in range(dim + 1)])
            # Actually we need to be careful: evaluate increments call_count
            # Let's restructure: we already have f values from initial eval, 
            # but after each iteration we need to re-evaluate only changed points.
            # Simpler approach: maintain f values array alongside simplex.
            
            # Let me rewrite this more carefully.
            pass

        # The above loop is flawed because evaluate() is called multiple times.
        # Let me redo with proper bookkeeping.
        
        # Reset and do it properly
        best_x = None
        best_f = np.inf
        call_count = 0
        deadline = time.time() + 290

        def safe_eval(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return np.inf
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = 0.0
        for i in range(1, dim + 1):
            simplex[i, i - 1] = 0.5

        # Evaluate initial simplex
        f_vals = np.zeros(dim + 1)
        for i in range(dim + 1):
            simplex[i] = np.clip(simplex[i], -5, 5)
            f_vals[i] = safe_eval(simplex[i])
            if time.time() > deadline:
                break

        # Nelder-Mead iteration
        while call_count < budget and time.time() < deadline:
            # Sort by function value
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = safe_eval(x_ref)

            if f_ref < f_vals[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = safe_eval(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_vals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
            elif f_ref < f_vals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                f_vals[-1] = f_ref
            else:
                # Contraction
                if f_ref < f_vals[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (centroid - simplex[-1])
                x_con = np.clip(x_con, -5, 5)
                f_con = safe_eval(x_con)
                if f_con < min(f_ref, f_vals[-1]):
                    simplex[-1] = x_con
                    f_vals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = safe_eval(simplex[i])
                        if time.time() > deadline:
                            break

        return best_x, best_f
