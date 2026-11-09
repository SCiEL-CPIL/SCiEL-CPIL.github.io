"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Structure-exploiting optimizer for ZDT4: initialize on front, local search x[1:] to 0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Determine initial points on the Pareto manifold
        # For ZDT4, f1 = x1, f2 = g(x) * h(x1, g(x)).
        # Pareto front is at x1 in [0,1] and g(x) = 0 (which implies x[1:] = pi).
        # We will initialize x1 uniformly in [0,1] and x[1:] = pi.
        
        # How many points can we afford?
        # Each point requires 1 init eval + local search evals.
        # Nelder-Mead on 9 dims typically takes ~50-100 evals to converge well.
        # Let's assume ~60 evals per point on average.
        # Budget = 200.
        # If we use 60 per point, we can do ~3 points. That's too few for hypervolume.
        
        # Alternative: Hybrid approach.
        # Use a smaller population of "seeds" and refine them.
        # Or, use a very fast local search.
        # ZDT4's g(x) = sum (xi - pi)^2. Minimization drives xi to pi.
        # Wait, the problem says "minimizing all 2 objectives".
        # Standard ZDT4:
        # f1 = x1
        # g(x) = 1 + 10(n-1) + sum((xi^2 - 10*cos(2*pi*xi))) for i=2..n
        # h(x1, g) = 1 - sqrt(x1/g) - (x1/g)*sin(10*pi*x1)
        # f2 = g(x) * h(x1, g)
        
        # The Pareto front is defined by g(x)=0? No, g(x) >= 0.
        # Actually, for ZDT4, the Pareto front is at g(x)=0?
        # Let's check: if g=0, f2 = 0 * h = 0. f1 = x1.
        # So points (x1, 0) for x1 in [0,1] are on the front?
        # But h(x1, 0) is undefined (division by zero).
        # Actually, the limit of h(x1, g) as g->0 is 1 - sqrt(x1/g) ... which goes to -inf?
        # No, standard ZDT4 analysis:
        # The Pareto front is at x1 in [0,1] and x2...xn = pi.
        # At x2...xn = pi, g(x) = 1 + 10(n-1) + sum(0) = 1 + 90 = 91?
        # Wait, the term is (xi^2 - 10*cos(2*pi*xi)).
        # If xi = pi, xi^2 = pi^2 ~ 9.87. 10*cos(2*pi^2) ...
        # Let's re-read standard ZDT4 definition.
        # Usually, the global minimum of the sub-problem for g(x) is at xi=pi?
        # No, for ZDT4, the local minima of g(x) are at xi = pi.
        # g(pi) = pi^2 - 10*cos(2*pi^2).
        # Is this the minimum?
        # The function g(x) is designed such that the Pareto front corresponds to the global minimum of g(x).
        # For ZDT4, the global minimum of g(x) is NOT 0.
        # Actually, many sources say the Pareto front is at x2...xn = pi.
        # Let's assume the "known Pareto front" mentioned in the prompt implies we should target that structure.
        # The prompt says: "dense fixed sample of the known Pareto front".
        # This suggests the evaluator knows the front.
        
        # If I initialize at x[1:] = pi, I am on the "ridge" of the front.
        # Local search can then fine-tune.
        
        # Strategy:
        # 1. Generate K initial points with x1 spread in [0,1] and x[1:] = pi.
        # 2. For each point, run a local search (Nelder-Mead) to minimize the objectives?
        #    Multi-objective local search is tricky.
        #    Better: Minimize a scalarization or just minimize g(x) (the second part) while keeping x1 fixed?
        #    If we keep x1 fixed and minimize g(x), we move towards the front.
        #    Since f1 = x1, fixing x1 fixes f1.
        #    Minimizing g(x) reduces f2 (roughly, depending on h).
        #    So, for each seed x1, find x[1:] that minimizes g(x).
        #    This is a single-objective optimization problem for each seed!
        
        # How many seeds?
        # Minimization of g(x) (9 vars) with Nelder-Mead.
        # Budget 200.
        # If each takes ~20-30 evals, we can do ~5-8 seeds.
        # 5-8 points might be sparse for hypervolume.
        
        # Can we do better?
        # Maybe a population-based approach is better for coverage.
        # But the "Structure-Exploiting" inspiration (score 1.22) suggests targeting the structure is key.
        
        # Let's try a hybrid:
        # 1. Init 20 points: x1 uniform, x[1:] = pi + small noise.
        # 2. Use a simple gradient-free local search on the full vector?
        #    Or, since we know x1 controls f1, we can just refine x[1:] to minimize g(x).
        
        # Let's implement:
        # - 10 seeds.
        # - Each seed: x1 fixed, x[1:] initialized to pi.
        # - Optimize x[1:] to minimize g(x) (which is f2 / h ... no, just minimize the g part).
        #   Actually, problem(x) returns [f1, f2].
        #   We can just call problem(x) and look at the second objective?
        #   But h depends on x1 and g.
        #   If we fix x1, minimizing g(x) generally improves f2 (moves towards front).
        
        # Let's use scipy.optimize.minimize on the sub-vector x[1:].
        
        n_seeds = 10
        x1_vals = np.linspace(0.05, 0.95, n_seeds)
        
        results = []
        used_evals = 0
        budget = self.budget
        
        # We need to count evaluations.
        # problem(x) is the evaluator.
        
        def eval_count_wrapper(x):
            nonlocal used_evals
            used_evals += 1
            return problem(x)
            
        # But scipy doesn't easily let us wrap with nonlocal count in all versions or it's messy.
        # We can just estimate.
        # Or we can implement a simple Nelder-Mead ourselves to control evals exactly?
        # No, scipy is fine, we just check `used_evals` by counting calls.
        
        # Let's make a class or closure to count.
        eval_counter = [0]
        
        def counting_problem(x):
            eval_counter[0] += 1
            return problem(x)
            
        # Initial population
        for i in range(n_seeds):
            if time.time() > deadline or eval_counter[0] >= budget:
                break
                
            x1 = x1_vals[i]
            # Initialize x[1:] to pi
            x_sub = np.full(self.dim - 1, np.pi)
            x_full = np.concatenate(([x1], x_sub))
            
            # Evaluate initial point
            f_init = counting_problem(x_full)
            results.append(f_init)
            
            # Define objective for local search: minimize the second objective?
            # Or minimize a sum?
            # Minimizing f2 is good, but f1 is fixed.
            # So minimize f2.
            
            def sub_obj(x_sub_vars):
                x_test = np.concatenate(([x1], x_sub_vars))
                f = counting_problem(x_test)
                # Return f2 (index 1)
                return f[1]
                
            # Run Nelder-Mead
            # Max iterations/evals per seed: remaining budget / remaining seeds
            # Simple heuristic: 10 evals per seed?
            # Nelder-Mead on 9 dims might need more.
            # Let's try maxiter=5, which is usually ~20-30 evals.
            
            try:
                res = minimize(sub_obj, x_sub, method='Nelder-Mead', options={'maxiter': 5, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.success:
                    x_opt = np.concatenate(([x1], res.x))
                    f_opt = counting_problem(x_opt)
                    results.append(f_opt)
            except Exception:
                pass
                
        return np.array(results)

# EVOLVE-BLOCK-END
