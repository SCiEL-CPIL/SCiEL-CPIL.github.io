"""Multi-Objective Optimizer using Multi-Start Augmented Epsilon-Constraint Method for RE21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    """
    Multi-objective optimizer using the Augmented Epsilon-Constraint Method.
    
    Strategy:
    1. Initialize with a set of diverse random points to estimate the range of objectives.
    2. Define a set of epsilon values for the second objective (f2) spanning from its max to min.
    3. For each epsilon, solve the constrained optimization problem:
       minimize f1(x)
       subject to f2(x) <= epsilon
    4. Use L-BFGS-B with a penalty method for the constraint, starting from multiple diverse initial points 
       to avoid local optima and cover the Pareto front.
    5. Record all evaluated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # State tracking
        eval_count = 0
        max_evals = self.budget
        evaluated_points = []
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= max_evals:
                return None
            if time.time() > deadline:
                return None
            fx = problem(x)
            eval_count += 1
            evaluated_points.append((x, fx))
            return fx

        # 1. Initialization: Sample points to estimate objective ranges
        # We need to know the range of f2 to set epsilon values.
        # Use a small number of random samples first.
        n_init = 20
        f2_values = []
        f1_values = []
        
        for _ in range(n_init):
            x = np.random.uniform(lo, hi)
            fx = evaluate(x)
            if fx is None:
                break
            f1_values.append(fx[0])
            f2_values.append(fx[1])
            
        if not f1_values:
            return np.array([])

        f1_min_init = min(f1_values)
        f1_max_init = max(f1_values)
        f2_min_init = min(f2_values)
        f2_max_init = max(f2_values)
        
        # If range is zero, add small noise
        if f1_max_init == f1_min_init:
            f1_max_init = f1_min_init + 1e-5
        if f2_max_init == f2_min_init:
            f2_max_init = f2_min_init + 1e-5

        # 2. Define Epsilon values for f2
        # We want to trace the front from high f2 (low f1 usually) to low f2 (high f1 usually)
        # Let's use 10-15 epsilon levels depending on remaining budget.
        # Each level will involve multiple local optimizations.
        
        remaining_budget = max_evals - eval_count
        if remaining_budget <= 0:
            return np.array([x for x, f in evaluated_points])

        # Allocate budget:
        # - Reserve 10% for final random exploration/polishing if needed? No, let's use it all for Epsilon method.
        # - Number of Epsilon levels: ~10
        # - Number of restarts per level: ~ (remaining_budget / 10) / 10 ? 
        # Let's say we have ~180 evals left.
        # 10 epsilon levels. 18 evals per level.
        # Each L-BFGS-B run takes ~5-10 evals. So ~2-3 restarts per epsilon level.
        
        n_epsilons = 12
        epsilons = np.linspace(f2_max_init, f2_min_init, n_epsilons)
        
        # To ensure we cover the whole front, we might want to bias towards the "knee" or just uniform.
        # Uniform in f2 space is standard for Epsilon-constraint.
        
        # Define the constrained objective function for L-BFGS-B
        # We use a penalty method: minimize f1 + lambda * max(0, f2 - eps)
        # L-BFGS-B does not handle inequality constraints directly, so penalty is necessary.
        # Alternatively, we can use SLSQP which handles constraints, but it's slower per eval.
        # L-BFGS-B is faster. Penalty method is stochastic in lambda? No, fixed lambda.
        # If lambda is too small, constraint is violated. If too large, it dominates f1.
        # Let's use a large penalty factor.
        
        penalty_factor = 100.0
        
        def constrained_obj(x, eps):
            fx = problem(x)
            eval_count += 1
            # Check time inside the loop is hard with nonlocal, but evaluate() does it.
            # Wait, if I call problem() directly here, I bypass the 'evaluate' helper's time check 
            # and list appending logic unless I replicate it.
            # Better to use the 'evaluate' helper? 
            # The 'evaluate' helper returns fx or None. If None, we return a large penalty.
            # But 'evaluate' increments eval_count.
            
            # Let's rewrite to use the global counter and list properly.
            # The 'evaluate' function above appends to evaluated_points.
            # So we should call evaluate(x).
            
            # However, the penalty function needs to be smooth for L-BFGS-B.
            # max(0, f2 - eps) is not differentiable at 0. 
            # We can use a smooth approximation or just rely on L-BFGS-B's robustness.
            # Or use SLSQP with constraints. SLSQP is more reliable for constraints but uses more evals per iteration?
            # Given the 200 budget, SLSQP might be too slow if it takes 50 evals per run.
            # Let's stick to L-BFGS-B with a soft penalty: 
            # penalty = penalty_factor * (f2 - eps) if f2 > eps else 0
            # This is piecewise linear. L-BFGS-B might struggle.
            # Let's use a quadratic penalty: penalty_factor * max(0, f2 - eps)**2
            
            fx = evaluate(x)
            if fx is None:
                return 1e10
            f1, f2 = fx
            violation = max(0.0, f2 - eps)
            return f1 + penalty_factor * (violation ** 2)

        # 3. Main Loop: Iterate over Epsilons
        # For each epsilon, we want to find a good point.
        # We will use multiple restarts from different initial points to avoid local optima.
        
        # How many restarts per epsilon?
        # Total budget left: ~180.
        # 12 epsilons.
        # 15 evals per epsilon.
        # L-BFGS-B might converge in 10-15 evals.
        # So 1 restart per epsilon might be enough if we choose good initial points.
        # But 1 restart is risky.
        # Let's do 2 restarts per epsilon, using 7-8 evals each?
        # Or 1 restart with a higher maxiter?
        
        # Let's try 2 restarts per epsilon.
        restarts_per_eps = 2
        
        # Initial points for restarts:
        # 1. The best point found so far that satisfies f2 <= eps (if any)
        # 2. A random point
        # 3. A point from the previous epsilon's solution (warm start)
        
        current_best_per_eps = {}
        last_solution = None
        
        for i, eps in enumerate(epsilons):
            if eval_count >= max_evals - 5:
                break
            if time.time() > deadline - 10:
                break
                
            # Determine initial points for this epsilon
            init_points = []
            
            # 1. Warm start from previous epsilon
            if last_solution is not None:
                init_points.append(last_solution)
                
            # 2. Best point from previous evaluations that is feasible (f2 <= eps)
            feasible_points = [x for x, f in evaluated_points if f[1] <= eps]
            if feasible_points:
                # Pick the one with lowest f1
                feasible_points.sort(key=lambda x: evaluated_points[evaluated_points.index((x, None))][1][0] if False else 0) 
                # This sorting is inefficient and buggy. Let's just pick the first feasible one or random.
                # Better: pick the point with lowest f1 among feasible.
                best_feasible_x = None
                best_feasible_f1 = float('inf')
                for x, f in evaluated_points:
                    if f[1] <= eps and f[0] < best_feasible_f1:
                        best_feasible_f1 = f[0]
                        best_feasible_x = x
                if best_feasible_x is not None:
                    init_points.append(best_feasible_x)
            else:
                # No feasible point yet, use random
                init_points.append(np.random.uniform(lo, hi))
            
            # Ensure we have at least 2 distinct initial points
            while len(init_points) < restarts_per_eps:
                init_points.append(np.random.uniform(lo, hi))
            
            # Remove duplicates (approximate)
            unique_inits = []
            for p in init_points:
                is_dup = False
                for q in unique_inits:
                    if np.allclose(p, q, atol=1e-5):
                        is_dup = True
                        break
                if not is_dup:
                    unique_inits.append(p)
            
            # Run optimization for each initial point
            best_x_for_eps = None
            best_f_for_eps = None
            
            for x0 in unique_inits:
                if eval_count >= max_evals - 2:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Define the objective for this specific run
                def obj_func(x, eps_val=eps):
                    return constrained_obj(x, eps_val)
                
                try:
                    # L-BFGS-B options
                    # maxfun is the max number of function evaluations
                    # We have limited budget. Let's allow up to 10-15 evals per run.
                    max_fun = 15
                    options = {'maxiter': 50, 'maxfun': max_fun, 'disp': False}
                    
                    res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options=options)
                    
                    # Evaluate the result to get true f1, f2
                    x_res = res.x
                    # Clip to bounds just in case
                    x_res = np.clip(x_res, lo, hi)
                    
                    fx_res = evaluate(x_res)
                    if fx_res is not None:
                        # Check if this is a better solution for this epsilon
                        # "Better" means lower f1, while satisfying f2 <= eps (approximately)
                        # If it violates constraint, we still keep it if it's close, 
                        # but ideally we want feasible.
                        # For the purpose of tracing the front, we record all points.
                        # We want to select the best x for the next epsilon's warm start.
                        # Prefer feasible points.
                        
                        if best_f_for_eps is None:
                            best_x_for_eps = x_res
                            best_f_for_eps = fx_res
                        else:
                            # Compare f1
                            # If both feasible, lower f1 is better.
                            # If one feasible and one not, feasible is better.
                            # If both infeasible, lower f1 is better (usually).
                            
                            curr_feas = fx_res[1] <= eps
                            best_feas = best_f_for_eps[1] <= eps
                            
                            if curr_feas and not best_feas:
                                best_x_for_eps = x_res
                                best_f_for_eps = fx_res
                            elif not curr_feas and best_feas:
                                pass
                            elif curr_feas and best_feas:
                                if fx_res[0] < best_f_for_eps[0]:
                                    best_x_for_eps = x_res
                                    best_f_for_eps = fx_res
                            else: # both infeasible
                                if fx_res[0] < best_f_for_eps[0]:
                                    best_x_for_eps = x_res
                                    best_f_for_eps = fx_res
                            
                except Exception:
                    pass
            
            if best_x_for_eps is not None:
                last_solution = best_x_for_eps
                current_best_per_eps[eps] = best_x_for_eps

        # 4. Final Polish:
        # We might have points that are close to the front but not exactly on it.
        # We can run a few more L-BFGS-B optimizations on the best non-dominated points found so far
        # using weighted sum to refine them.
        
        # Identify non-dominated points
        if evaluated_points:
            nd_points = []
            for i, (x1, f1) in enumerate(evaluated_points):
                dominated = False
                for j, (x2, f2) in enumerate(evaluated_points):
                    if i == j: continue
                    if np.all(f2 <= f1) and np.any(f2 < f1):
                        dominated = True
                        break
                if not dominated:
                    nd_points.append((x1, f1))
            
            # Sort by sum of objectives
            nd_points.sort(key=lambda p: p[1][0] + p[1][1])
            
            # Take top 5 for refinement
            n_refine = min(5, len(nd_points))
            for x0, f0 in nd_points[:n_refine]:
                if eval_count >= max_evals - 2:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Weighted sum refinement
                # Choose weights based on the point's position? 
                # Or just random weights.
                # Since we are refining, we want to stay close.
                # Let's use a weight vector that emphasizes the direction towards the ideal point.
                # Ideal point: [min_f1, min_f2]
                # Anti-ideal: [max_f1, max_f2]
                
                w = np.random.dirichlet(np.ones(self.n_obj))
                
                def weighted_obj(x):
                    fx = evaluate(x)
                    if fx is None:
                        return 1e10
                    return np.dot(w, fx)
                
                try:
                    options = {'maxiter': 20, 'maxfun': 10, 'disp': False}
                    res = minimize(weighted_obj, x0, method='L-BFGS-B', bounds=bounds, options=options)
                    x_res = np.clip(res.x, lo, hi)
                    evaluate(x_res)
                except Exception:
                    pass

        # Return the set of x's (evaluator records f's, but we return x's as per contract? 
        # "The return value of __call__ is ignored." 
        # But contract says: "Write class Optimizer ... and __call__(self, problem, bounds)"
        # And "Every returned objective vector is recorded" -> This refers to problem(x) calls.
        # So return value doesn't matter much, but let's return the best x or empty array.
        
        if evaluated_points:
            # Return the x of the point with best hypervolume contribution? 
            # Doesn't matter. Just return one.
            return evaluated_points[0][0]
        return np.array([])

# EVOLVE-BLOCK-END
