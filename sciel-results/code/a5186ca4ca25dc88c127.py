"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Weighted-sum scalarization with L-BFGS-B for Multi-objective Optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # 1. Global Coverage: Latin Hypercube Sampling
        # Use ~30% of budget for initial global sampling to ensure we don't miss
        # disconnected parts of the front or bad basins.
        n_lhs = max(10, int(self.budget * 0.3))
        
        lhs_points = np.zeros((n_lhs, dim))
        for j in range(dim):
            u = np.random.rand(n_lhs)
            points = (np.arange(n_lhs) + u) / n_lhs
            np.random.shuffle(points)
            lhs_points[:, j] = points
            
        X_lhs = lo + (hi - lo) * lhs_points
        
        F_lhs = np.zeros((n_lhs, n_obj))
        eval_count = 0
        
        for i in range(n_lhs):
            if time.time() > deadline or eval_count >= self.budget:
                break
            try:
                F_lhs[i] = problem(X_lhs[i])
                eval_count += 1
            except Exception:
                F_lhs[i] = np.full(n_obj, 1e6)
        
        # 2. Local Refinement: Weighted Sum Scalarization
        # We define a set of weight vectors to trace the Pareto front.
        # For 3 objectives, we can use a grid of weights.
        # Since the front is likely convex or nearly so for RE37, this works well.
        
        # Generate weight vectors on the simplex
        # We'll use a set of K weight vectors.
        # K should be small enough to allow multiple restarts per weight if needed,
        # or just 1 restart per weight if time is tight.
        # Given budget=200 and n_lhs=60, we have ~140 evals left.
        # Let's try K=10 weight vectors.
        
        # Simple grid for 3 objectives:
        # w1 + w2 + w3 = 1, wi >= 0
        # We can sample points on the triangle.
        
        n_weights = 12
        weights = []
        # Add extreme points
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        # Add mid-points of edges
        weights.append([0.5, 0.5, 0.0])
        weights.append([0.5, 0.0, 0.5])
        weights.append([0.0, 0.5, 0.5])
        # Add center
        weights.append([1/3, 1/3, 1/3])
        # Add some random points on the simplex to cover other areas
        while len(weights) < n_weights:
            # Dirichlet distribution with alpha=1
            w = np.random.dirichlet([1, 1, 1])
            # Check if it's distinct enough from existing? For simplicity, just add.
            weights.append(w)
            
        weights = np.array(weights)
        
        # Select starting points for optimization
        # For each weight vector, we want to start near a point that is good for that direction.
        # A simple heuristic: pick the LHS point that minimizes the weighted sum for that weight.
        
        # Calculate weighted sums for all LHS points
        # F_lhs shape: (n_lhs, n_obj)
        # weights shape: (n_weights, n_obj)
        # Weighted sums: (n_weights, n_lhs)
        
        # Only consider valid LHS points (those that didn't throw exceptions)
        valid_lhs_mask = np.all(F_lhs < 1e5, axis=1)
        X_lhs_valid = X_lhs[valid_lhs_mask]
        F_lhs_valid = F_lhs[valid_lhs_mask]
        
        if len(X_lhs_valid) == 0:
            return None
            
        # If we have very few valid points, just use random starts
        if len(X_lhs_valid) < n_weights:
             start_indices = np.random.choice(len(X_lhs_valid), size=n_weights, replace=True)
        else:
             # For each weight, find the best LHS point
             start_indices = np.zeros(n_weights, dtype=int)
             for k in range(n_weights):
                 ws = F_lhs_valid @ weights[k]
                 start_indices[k] = np.argmin(ws)
        
        # Remaining budget
        remaining_budget = self.budget - eval_count
        # We will run n_weights optimizations.
        # Each optimization might take several function evaluations.
        # L-BFGS-B usually converges in < 50 evals for 4D problems.
        # 12 weights * 50 evals = 600 evals. We only have ~140.
        # So we need to limit the maxiter or use fewer weights.
        
        # Let's reduce n_weights to 5 and allow more iterations, or keep 12 but limit iterations.
        # Actually, 140 evals / 12 weights = ~11 evals per weight.
        # That might be too few for L-BFGS-B to converge well from a random start.
        
        # Alternative: Use fewer, better-spaced weights.
        # Let's try 5 weights: 3 extremes + 2 mid-edge or center.
        # Or stick with 12 but accept that they won't fully converge.
        # Convergence isn't strictly required if we record all intermediate points?
        # No, `problem` is called by `minimize`. We can't easily intercept intermediate calls 
        # unless we wrap the problem function.
        
        # Let's wrap the problem to count and check time.
        
        # To maximize hypervolume, we want points on the front.
        # If we don't converge, we might end up in the interior.
        # However, even interior points can contribute if they are non-dominated.
        
        # Let's try a different strategy for the local phase:
        # Use the best LHS points as seeds for a local random search with smaller steps,
        # but guided by the weighted sums?
        
        # Actually, let's stick to Weighted Sum but limit the number of weights to fit the budget.
        # If we have 140 evals, and we want to run 10 optimizations, we have 14 evals each.
        # L-BFGS-B with maxiter=10 might still move towards the optimum.
        
        n_opt = min(n_weights, max(5, remaining_budget // 15)) # At least 5, at most budget/15
        
        # Select the first n_opt weights
        weights_used = weights[:n_opt]
        
        for k in range(n_opt):
            if time.time() > deadline or eval_count >= self.budget:
                break
                
            w = weights_used[k]
            
            # Define scalar objective
            def scalar_obj(x, w=w):
                try:
                    f = problem(x)
                    # Check if this call is valid
                    if not np.all(np.isfinite(f)):
                        return 1e6
                    return np.dot(f, w)
                except Exception:
                    return 1e6
            
            # Start point
            x0 = X_lhs_valid[start_indices[k % len(start_indices)]].copy()
            
            # Bounds
            bounds_opt = list(zip(lo, hi))
            
            # Run optimization
            # Use L-BFGS-B
            try:
                res = minimize(
                    scalar_obj, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds_opt,
                    options={'maxiter': 20, 'ftol': 1e-8}
                )
            except Exception:
                pass

        return None
# EVOLVE-BLOCK-END
