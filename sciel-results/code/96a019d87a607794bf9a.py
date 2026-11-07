"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    """Model-and-optimize approach: Weighted Sum Local Search with diverse weights."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        self.eval_count = 0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Define weight vectors on the simplex for n_obj objectives
        # For 3 objectives, we can sample points on the 2D simplex
        n_weights = 10
        rng = np.random.default_rng(42)
        weights = []
        for _ in range(n_weights):
            w = rng.dirichlet([1, 1, 1])
            weights.append(w)
            
        # Ensure we also have vertices and edge midpoints for better coverage
        weights.append([1, 0, 0])
        weights.append([0, 1, 0])
        weights.append([0, 0, 1])
        weights.append([0.5, 0.5, 0])
        weights.append([0.5, 0, 0.5])
        weights.append([0, 0.5, 0.5])
        
        # Deduplicate and normalize
        unique_weights = []
        for w in weights:
            w = np.array(w)
            w = w / w.sum()
            is_dup = False
            for uw in unique_weights:
                if np.allclose(w, uw, atol=1e-3):
                    is_dup = True
                    break
            if not is_dup:
                unique_weights.append(w)
        
        final_weights = unique_weights
        
        # Time limit with safety margin
        time_limit = 295.0
        
        def check_time():
            return time.time() - self.start_time < time_limit
        
        def check_budget():
            return self.eval_count < self.budget
        
        def eval_and_count(x):
            if not check_budget() or not check_time():
                raise Exception("Budget or time exceeded")
            x = np.clip(x, lo, hi)
            res = problem(x)
            self.eval_count += 1
            return res
        
        # 1. Random Sampling - use 5% of budget for initial diversity
        random_budget = max(5, int(self.budget * 0.05))
        for _ in range(random_budget):
            if not check_budget() or not check_time():
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            eval_and_count(x)

        # 2. Weighted Local Search with multiple restarts per weight
        # Allocate remaining budget across weights, with multiple restarts
        remaining_budget = self.budget - self.eval_count
        n_restarts_per_weight = 3
        total_slots = len(final_weights) * n_restarts_per_weight
        budget_per_slot = max(5, remaining_budget // total_slots)
        
        bounds_opt = list(zip(lo, hi))
        best_solutions = []
        
        for w in final_weights:
            for restart in range(n_restarts_per_weight):
                if not check_budget() or not check_time():
                    break
                
                # Start from a random point
                x0 = lo + (hi - lo) * rng.random(self.dim)
                
                # Define the scalar objective for this weight
                def obj(x, w=w):
                    if not check_budget() or not check_time():
                        raise Exception("Budget or time exceeded")
                    f = problem(x)
                    self.eval_count += 1
                    return np.dot(w, f)
                
                try:
                    res = opt.minimize(obj, x0, method='L-BFGS-B', bounds=bounds_opt, 
                                        options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6})
                    if res.x is not None and res.fun is not None:
                        best_solutions.append(np.clip(res.x, lo, hi))
                except Exception:
                    pass

        # 3. Polish pass: refine the best solutions found with tighter tolerances
        if best_solutions and check_time():
            # Keep only the most promising solutions (by first objective as proxy)
            if len(best_solutions) > 6:
                # Sort by sum of objectives (lower is better)
                def solution_score(x):
                    try:
                        f = eval_and_count(x)
                        return np.sum(f)
                    except Exception:
                        return np.inf
                
                scored = [(solution_score(x), x) for x in best_solutions]
                scored.sort(key=lambda t: t[0])
                best_solutions = [x for _, x in scored[:6]]
            
            for x_best in best_solutions:
                if not check_budget() or not check_time():
                    break
                
                # Use a small perturbation neighborhood search
                for obj_idx in range(self.n_obj):
                    if not check_budget() or not check_time():
                        break
                    
                    w_polish = np.zeros(self.n_obj)
                    w_polish[obj_idx] = 1.0
                    
                    def obj_polish(x, w=w_polish):
                        if not check_budget() or not check_time():
                            raise Exception("Budget or time exceeded")
                        f = problem(x)
                        self.eval_count += 1
                        return np.dot(w, f)
                    
                    try:
                        res = opt.minimize(obj_polish, x_best, method='L-BFGS-B', 
                                            bounds=bounds_opt,
                                            options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-8})
                    except Exception:
                        pass

        return None
# EVOLVE-BLOCK-END
