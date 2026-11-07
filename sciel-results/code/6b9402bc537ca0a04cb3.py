"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT1's separable structure.
    
    Strategy:
    1. Initialize a diverse set of points with x1 spread across [0,1] and x[1:] at 0.5.
    2. For each point, use Nelder-Mead to minimize g(x[1:]) (which is sum of squares of x[1:])
       while keeping x1 fixed, driving points toward the Pareto front.
    3. Use remaining budget for a simple evolutionary refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Number of initial points to place along the Pareto front
        n_init = min(20, budget // 5)
        
        # Initialize: spread x1 across [0,1], set x[1:] = 0.5
        x1_values = np.linspace(0.0, 1.0, n_init)
        X = np.zeros((n_init, dim))
        for i in range(n_init):
            X[i, 0] = x1_values[i]
            X[i, 1:] = 0.5
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # For each point, run Nelder-Mead on x[1:] to minimize g
        # g(x[1:]) = sum(x[1:]^2), so we want to drive x[1:] toward 0
        # We minimize f2 = (1 + g) * (1 - sqrt(f1/(1+g)))
        # But since we fix x1, f1 = x1 is fixed, so minimizing f2 is equivalent to minimizing g
        
        for i in range(n_init):
            if used >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            
            x1_val = X[i, 0]
            
            # Define objective: given x_tail (dim-1 vars), compute f2
            def obj_tail(x_tail):
                x = np.concatenate(([x1_val], x_tail))
                x = np.clip(x, lo, hi)
                f = problem(x)
                return f[1]
            
            # Nelder-Mead on x_tail
            x0_tail = X[i, 1:].copy()
            x_tail_opt, f2_opt, n_iter = self._nelder_mead(obj_tail, x0_tail, lo[1:], hi[1:], budget_remaining=budget - used)
            
            if n_iter > 0:
                used += n_iter
                X[i, 1:] = x_tail_opt
                F[i, 1] = f2_opt
        
        # Check time
        if time.time() > deadline - 5.0 or used >= budget:
            return X
        
        # Use remaining budget for evolutionary refinement
        while used < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Find non-dominated points
            dominated = np.zeros(n_init, dtype=bool)
            for i in range(n_init):
                for j in range(n_init):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            parents_idx = np.where(~dominated)[0]
            if len(parents_idx) == 0:
                parents_idx = np.arange(n_init)
            
            parent_idx = parents_idx[rng.integers(len(parents_idx))]
            child = X[parent_idx].copy()
            
            # Mutate
            sigma = 0.1 * (hi - lo)
            child = child + sigma * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc
        
        return X

    def _nelder_mead(self, func, x0, lo, hi, budget_remaining=50):
        """Simple Nelder-Mead minimization with bounds."""
        dim = len(x0)
        # Initial simplex
        simplex = [x0.copy()]
        for i in range(dim):
            point = x0.copy()
            step = 0.05 * (hi[i] - lo[i]) if (hi[i] - lo[i]) > 0 else 0.05
            point[i] += step
            point = np.clip(point, lo, hi)
            simplex.append(point)
        
        simplex = np.array(simplex)
        fvals = np.array([func(x) for x in simplex])
        n_evals = dim + 1
        
        max_iter = min(budget_remaining, 50)
        
        for _ in range(max_iter):
            if n_evals >= budget_remaining:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, lo, hi)
            f_ref = func(x_ref)
            n_evals += 1
            
            if f_ref < fvals[-2]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, lo, hi)
                f_exp = func(x_exp)
                n_evals += 1
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[0]:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (simplex[-1] - centroid)
                x_con = np.clip(x_con, lo, hi)
                f_con = func(x_con)
                n_evals += 1
                
                if f_con < f_ref:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                    simplex[1:] = np.clip(simplex[1:], lo, hi)
                    fvals[1:] = np.array([func(x) for x in simplex[1:]])
                    n_evals += (dim - 1)
            else:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (simplex[-1] - centroid)
                x_con = np.clip(x_con, lo, hi)
                f_con = func(x_con)
                n_evals += 1
                
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                    simplex[1:] = np.clip(simplex[1:], lo, hi)
                    fvals[1:] = np.array([func(x) for x in simplex[1:]])
                    n_evals += (dim - 1)
            
            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break
        
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx], n_evals
# EVOLVE-BLOCK-END
