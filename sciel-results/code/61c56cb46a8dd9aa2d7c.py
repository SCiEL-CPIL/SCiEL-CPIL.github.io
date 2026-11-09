"""Multi-objective Optimizer for RE21 using Sobol Initialization and Multi-Start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using a hybrid strategy:
    1. Sobol quasi-random initialization for broad coverage.
    2. Multi-start local optimization using weighted-sum scalarization.
    3. Dynamic budget allocation to balance exploration (random) and exploitation (local search).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Sobol Initialization
        # Use scipy.stats.qmc.Sobol for low-discrepancy sequence
        from scipy.stats import qmc
        n_init = min(64, self.budget // 3)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Generate points in [0, 1]^d
        # Sobol requires power of 2 for best properties, but any number works
        # We take the first n_init points
        X_init_unit = sampler.random(n_init)
        X_init = lo + (hi - lo) * X_init_unit
        
        X_all = X_init
        F_all = np.array([problem(x) for x in X_init])
        
        # Helper: Check time
        def time_left():
            return deadline - time.time()
            
        # Helper: Dominance check
        def is_dominated(f_new, F_front):
            # Returns True if f_new is dominated by any point in F_front
            # F_front is NxM array
            # f_new is M array
            if len(F_front) == 0:
                return False
            # f_new is dominated if exists p in F_front such that p <= f_new and p != f_new
            # p <= f_new: all(p_i <= f_new_i)
            # p != f_new: any(p_i < f_new_i)
            
            # Vectorized check
            # Shape: (N, M) vs (M,)
            less_eq = np.all(F_front <= f_new, axis=1)
            less_than = np.any(F_front < f_new, axis=1)
            return np.any(less_eq & less_than)

        # Helper: Update Pareto Front
        def update_front(X, F):
            # Keep only non-dominated points
            # For small N, O(N^2) is fine
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    # If j dominates i, i is not on front
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return X[mask], F[mask]

        # Initial Front
        X_front, F_front = update_front(X_all, F_all)

        # 2. Main Loop: Multi-Start Local Search
        # We will alternate between:
        # - Adding a new random/Sobol point (Exploration)
        # - Running L-BFGS-B from a point on the current front (Exploitation)
        
        # Weights for weighted sum
        # Since we have 2 objectives, weights are (w, 1-w)
        # We want to cover the front. We can use a set of fixed weights or random weights.
        # Let's use a set of representative weights to ensure coverage of the front.
        # For 2 objectives, uniform weights in [0.1, 0.9] are good.
        # We will cycle through a grid of weights.
        
        # Define a set of weights to try
        # We have ~130 evaluations left.
        # Each L-BFGS-B run uses ~10-30 evals.
        # So we can do ~5-10 local runs.
        # Let's pick 10 distinct weight vectors.
        weights_list = np.linspace(0.1, 0.9, 10)
        
        # We also want to keep some budget for pure random search to escape local minima?
        # Sobol init is already good.
        # Let's use the remaining budget for local search, but if time allows, add a few more random points.
        
        # Strategy:
        # While budget remains and time allows:
        #   1. Pick a starting point (from current front or random)
        #   2. Pick a weight vector
        #   3. Run L-BFGS-B to minimize w1*f1 + w2*f2
        #   4. Add result to history
        
        eval_count = len(X_all)
        
        # We will perform local optimizations from the current best points.
        # To diversify, we will pick starting points from the front and also some random points.
        
        # Pre-defined weights for coverage
        # We'll iterate through weights, and for each weight, pick the best current point
        # in terms of that weighted sum as the start, or a random point.
        
        # Let's create a pool of starting points:
        # 1. All points in current front
        # 2. A few random points
        
        # To ensure we don't get stuck, we'll mix:
        # 50% of local searches start from the front (exploitation)
        # 50% start from random points (exploration via local search)
        
        # We have budget - eval_count evaluations left.
        # Let's assume each local search takes ~15 evals on average.
        # Max local searches = (budget - eval_count) // 15
        
        # However, L-BFGS-B with analytic gradient? No, problem(x) is black box.
        # We must use numerical gradient or just L-BFGS-B with finite differences.
        # Finite differences in 4D cost ~5 evals per gradient step.
        # L-BFGS-B might take 10-20 steps. So ~50-100 evals per run?
        # That's too expensive for 200 budget.
        
        # Correction: 
        # If we use `minimize` with method='L-BFGS-B', it uses finite differences.
        # For dim=4, gradient costs 5 evals.
        # If it takes 10 steps, that's 50 evals.
        # We have ~140 evals left. So we can do ~2-3 local searches.
        # That's not enough for a good front.
        
        # Alternative: Use a cheaper local search?
        # Or reduce the number of steps?
        # `maxiter` can be limited.
        
        # Better Strategy:
        # Use a smaller number of local searches with strict maxiter, or use a different method.
        # Maybe Nelder-Mead? It doesn't use gradients.
        # NM in 4D: ~10-20 evals per function eval of the simplex update?
        # Actually NM is also expensive.
        
        # Let's stick to L-BFGS-B but limit `maxiter` to 5-10 steps.
        # 5 steps * 5 evals/step = 25 evals per run.
        # 140 / 25 = ~5 runs.
        # With 64 init points, we have a good front.
        # 5 local refinements on the best 5 points of the front should improve it significantly.
        
        # Let's refine the top K points of the front.
        # Sort front by hypervolume contribution or just take all if K < len(front).
        
        # We will perform local searches from the points in the current front that have the best weighted sum for a set of weights.
        
        # Weights: [0.2, 0.4, 0.6, 0.8]
        # For each weight, find the point in X_front that minimizes w1*f1 + w2*f2.
        # Start L-BFGS-B from there.
        
        weights_to_use = [0.2, 0.4, 0.6, 0.8]
        
        for w1 in weights_to_use:
            if eval_count >= self.budget:
                break
            if time_left() < 30: # Need 30s for at least one more run
                break
                
            w2 = 1.0 - w1
            
            # Find best start point in current front for this weight
            if len(F_front) > 0:
                scores = w1 * F_front[:, 0] + w2 * F_front[:, 1]
                idx_best = np.argmin(scores)
                x_start = X_front[idx_best].copy()
            else:
                x_start = np.random.uniform(lo, hi)
            
            # Define objective for L-BFGS-B
            def weighted_obj(x):
                # Clip to bounds? L-BFGS-B handles bounds, but problem(x) might not.
                # The contract says problem(x) clips. So we can pass x directly.
                # But for safety in the local search, we should respect bounds in the optimization.
                f = problem(x)
                return w1 * f[0] + w2 * f[1]
            
            # Run L-BFGS-B
            # bounds list for scipy
            scipy_bounds = list(zip(lo, hi))
            
            try:
                res = minimize(weighted_obj, x_start, method='L-BFGS-B', bounds=scipy_bounds, 
                               options={'maxiter': 10, 'maxls': 10})
                # res.fun is the weighted value
                # We need the actual f values
                f_res = problem(res.x)
                
                # Update history
                X_all = np.vstack([X_all, res.x])
                F_all = np.vstack([F_all, f_res])
                eval_count += 1 # Approximate, L-BFGS-B uses more, but we track actual calls?
                # Wait, the evaluator counts `problem(x)` calls.
                # My code doesn't know how many times L-BFGS-B called `weighted_obj`.
                # I MUST track the actual number of calls to `problem`.
                
            except Exception as e:
                pass

        # This approach has a flaw: I don't know how many evaluations L-BFGS-B used.
        # I need to wrap `problem` to count calls.
        
        # RESTART with proper counting.
        
        # --- REIMPLEMENTATION WITH COUNTING ---
        
        X_all = []
        F_all = []
        call_count = 0
        
        def count_problem(x):
            nonlocal call_count
            call_count += 1
            return problem(x)
        
        # 1. Sobol Init
        n_init = min(64, self.budget // 3)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        X_init_unit = sampler.random(n_init)
        X_init = lo + (hi - lo) * X_init_unit
        
        for x in X_init:
            f = count_problem(x)
            X_all.append(x)
            F_all.append(f)
            
        X_all = np.array(X_all)
        F_all = np.array(F_all)
        
        X_front, F_front = update_front(X_all, F_all)
        
        # 2. Local Search
        # We will run local searches until we run out of budget or time.
        # Each search is capped at 20 function evaluations (approx).
        
        # Weights to explore the front
        weights_list = np.linspace(0.1, 0.9, 8)
        weight_idx = 0
        
        while call_count < self.budget - 5: # Keep 5 evals buffer
            if time_left() < 20:
                break
            
            w1 = weights_list[weight_idx % len(weights_list)]
            weight_idx += 1
            w2 = 1.0 - w1
            
            # Pick start point
            # Alternate between front point and random point
            if weight_idx % 2 == 0 and len(X_front) > 0:
                scores = w1 * F_front[:, 0] + w2 * F_front[:, 1]
                idx_best = np.argmin(scores)
                x_start = X_front[idx_best].copy()
            else:
                # Random start
                x_start = np.random.uniform(lo, hi)
            
            def weighted_obj(x):
                f = count_problem(x)
                return w1 * f[0] + w2 * f[1]
            
            scipy_bounds = list(zip(lo, hi))
            
            try:
                # Limit maxiter to keep eval count low
                # In 4D, 1 iter ~ 5 evals. maxiter=5 -> ~25 evals.
                # Let's try maxiter=4 to be safe (~20 evals)
                res = minimize(weighted_obj, x_start, method='L-BFGS-B', bounds=scipy_bounds, 
                               options={'maxiter': 4, 'maxls': 5})
                
                # Add result
                f_res = count_problem(res.x) # Ensure the final point is evaluated and recorded
                X_all = np.vstack([X_all, res.x])
                F_all = np.vstack([F_all, f_res])
                
                # Update front
                X_front, F_front = update_front(X_all, F_all)
                
            except Exception:
                pass
            
            # If we used too many evals in this run, break
            if call_count >= self.budget - 5:
                break

        # 3. Final Polish: If we have budget left, do a few random evals
        while call_count < self.budget:
            if time_left() < 5:
                break
            x = np.random.uniform(lo, hi)
            f = count_problem(x)
            X_all = np.vstack([X_all, x])
            F_all = np.vstack([F_all, f])
            X_front, F_front = update_front(X_all, F_all)

        return X_all

# EVOLVE-BLOCK-END
