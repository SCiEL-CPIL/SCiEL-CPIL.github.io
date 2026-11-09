import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 320s deadline, 10s margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        evals_used = 0
        results = []
        
        # Generate weight vectors: simplex grid for 3 objectives
        # Include vertices, edge midpoints, and interior points
        weights = []
        n_grid = 7
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                k = n_grid - i - j
                w = np.array([i / n_grid, j / n_grid, k / n_grid])
                weights.append(w)
        
        # Sort weights by a hash to ensure deterministic order
        weights = sorted(weights, key=lambda w: w[0] * 1000 + w[1] * 100 + w[2])
        
        n_weights = len(weights)
        # Budget allocation: spend most on promising weights
        # Use more evaluations for vertex and edge weights which are critical for hypervolume
        base_budget = 20
        extra_budget = (self.budget - n_weights * base_budget) // n_weights if n_weights > 0 else 0
        
        for idx, w in enumerate(weights):
            # Check deadline
            if time.time() > deadline - 2:  # 2s margin
                break
            
            if evals_used >= self.budget:
                break
            
            # Determine budget for this weight
            # Give more to vertices and edge midpoints
            if np.sum(w > 0) == 1:  # vertex
                budget_w = min(base_budget + extra_budget * 2, self.budget - evals_used)
            elif np.sum(w > 0) == 2:  # edge
                budget_w = min(base_budget + extra_budget, self.budget - evals_used)
            else:  # interior
                budget_w = min(base_budget, self.budget - evals_used)
            
            if budget_w < 5:
                continue
            
            # Multiple restarts with different starting points
            n_restarts = min(3, max(1, budget_w // 10))
            evals_per_restart = budget_w // n_restarts
            
            best_x = None
            best_val = np.inf
            
            for r in range(n_restarts):
                if time.time() > deadline - 2:
                    break
                if evals_used >= self.budget:
                    break
                
                # Generate diverse starting points
                if r == 0:
                    x0 = lo + (hi - lo) * np.random.random(self.dim)
                elif r == 1:
                    # Start near one of the vertices in objective space (heuristic)
                    x0 = lo + (hi - lo) * np.random.random(self.dim)
                else:
                    x0 = (lo + hi) / 2.0 + span * (np.random.random(self.dim) - 0.5) * 0.5
                
                x0 = np.clip(x0, lo, hi)
                
                def cheb_obj(x, _w=w):
                    nonlocal evals_used
                    if time.time() > deadline - 1:
                        raise TimeoutError()
                    if evals_used >= self.budget:
                        raise ValueError("Budget exceeded")
                    
                    f = problem(x)
                    evals_used += 1
                    
                    # Track running ideal point for this weight
                    # Use a simple normalization based on weight
                    # Chebyshev: minimize max(w_i * (f_i - ideal_i))
                    # Since we don't know the true ideal, use a heuristic:
                    # Scale objectives by their typical range implied by weights
                    # For WFG7, objectives are in [0, ~10]
                    # Use weight-based scaling
                    d = _w * f
                    return np.max(d)
                
                try:
                    res = minimize(cheb_obj, x0, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6,
                                            'maxls': 10})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_x = res.x
                except (TimeoutError, ValueError):
                    pass
                except Exception:
                    pass
            
            # Also try a few random samples within the budget
            remaining = min(5, self.budget - evals_used)
            for _ in range(remaining):
                if time.time() > deadline - 1 or evals_used >= self.budget:
                    break
                x_rand = lo + (hi - lo) * np.random.random(self.dim)
                try:
                    f = problem(x_rand)
                    evals_used += 1
                    results.append(x_rand)
                except Exception:
                    pass
            
            if best_x is not None:
                results.append(best_x)
        
        return results
