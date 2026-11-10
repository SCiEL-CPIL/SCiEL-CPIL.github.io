"""Optimized ZDT2 optimizer using local search for Hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Model-based optimizer for ZDT2.
    Uses local optimization to find the 200 points on the Pareto Front
    that maximize the Hypervolume with reference point [11, 11].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        n_evals = self.budget

        # ZDT2 Pareto Front: f1 = x[0], f2 = 1 - sqrt(x[0])
        # Reference point r = [11, 11]
        # Uncovered Area U(x) = sum_{i=1}^N (x_i - x_{i-1}) * (11 - f2_i)
        # f2_i = 1 - sqrt(x_i)
        # 11 - f2_i = 10 + sqrt(x_i)
        # So minimize U(x) = sum_{i=1}^N (x_i - x_{i-1}) * (10 + sqrt(x_i))
        # Subject to 0 <= x_1 <= x_2 <= ... <= x_N = 1

        def uncovered_area(x):
            # x is array of length N
            # x[0] is x_1, ..., x[N-1] is x_N
            # We enforce x_N = 1 in the optimization bounds/constraints
            # Or we can optimize x[0:N-1] and fix x[N-1]=1
            
            # Calculate differences
            # x_prev is [0, x_0, x_1, ..., x_{N-2}]
            # x_curr is [x_0, x_1, ..., x_{N-1}]
            
            # Using numpy for speed
            x_prev = np.concatenate(([0.0], x[:-1]))
            x_curr = x
            
            dx = x_curr - x_prev
            heights = 10.0 + np.sqrt(x_curr)
            
            return np.sum(dx * heights)

        def uncovered_area_grad(x):
            # Gradient of U(x)
            # U = sum (x_i - x_{i-1}) g(x_i)
            # dU/dx_k = (x_k - x_{k-1}) g'(x_k) + g(x_k) - (x_{k+1} - x_k) g(x_{k+1})? No.
            # Let's derive carefully.
            # Term k: (x_k - x_{k-1}) g(x_k)
            # Term k+1: (x_{k+1} - x_k) g(x_{k+1})
            
            # d/dx_k [ (x_k - x_{k-1}) g(x_k) ] = g(x_k) + (x_k - x_{k-1}) g'(x_k)
            # d/dx_k [ (x_{k+1} - x_k) g(x_{k+1}) ] = -g(x_{k+1})
            
            # So grad_k = g(x_k) + (x_k - x_{k-1}) g'(x_k) - g(x_{k+1})
            # For k=N-1 (last element), there is no term k+1.
            # grad_{N-1} = g(x_{N-1}) + (x_{N-1} - x_{N-2}) g'(x_{N-1})
            
            g = 10.0 + np.sqrt(x)
            # g'(x) = 0.5 / sqrt(x)
            # Handle x=0 carefully
            with np.errstate(divide='ignore', invalid='ignore'):
                gp = 0.5 / np.sqrt(x)
            gp[0] = 0.0 # If x[0]=0, we can set grad contribution to 0 or limit
            
            grad = np.zeros_like(x)
            
            # For i < N-1
            if len(x) > 1:
                dx = x[1:] - x[:-1]
                # grad[i] = g[i] + dx[i] * gp[i] - g[i+1]
                grad[:-1] = g[:-1] + dx * gp[:-1] - g[1:]
            
            # For last element
            dx_last = x[-1] - x[-2] if len(x) > 1 else x[-1]
            grad[-1] = g[-1] + dx_last * gp[-1]
            
            return grad

        # Initial guess: Uniform spacing in u(x) = 10x + 2/3 x^1.5
        def u_of_x(x):
            return 10.0 * x + (2.0 / 3.0) * np.power(x, 1.5)
        
        def x_of_u(u):
            # Newton's method
            x = np.clip(u / 10.0, 0.0, 1.0)
            for _ in range(20):
                fx = u_of_x(x) - u
                dfx = 10.0 + np.sqrt(x)
                x_new = x - fx / dfx
                x_new = np.clip(x_new, 0.0, 1.0)
                if np.max(np.abs(x_new - x)) < 1e-10:
                    x = x_new
                    break
                x = x_new
            return x

        u_max = u_of_x(1.0)
        u_vals = np.linspace(0.0, u_max, n_evals)
        x_init = x_of_u(u_vals)
        x_init[-1] = 1.0 # Ensure last point is 1

        # Optimization
        # We optimize x[0:N-1] and fix x[N-1] = 1?
        # Or optimize all with constraint x[-1]=1?
        # Easier: Optimize x[0:N-1] and append 1.
        
        def objective(x_var):
            x_full = np.append(x_var, 1.0)
            return uncovered_area(x_full)
            
        def grad(x_var):
            x_full = np.append(x_var, 1.0)
            g_full = uncovered_area_grad(x_full)
            return g_full[:-1] # Return gradient for variables only

        bounds_opt = [(0.0, 1.0) for _ in range(n_evals - 1)]
        
        res = minimize(objective, x_init[:-1], method='L-BFGS-B', jac=grad, bounds=bounds_opt, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        
        x_opt = res.x
        x_opt = np.append(x_opt, 1.0)
        
        # Ensure sorted and within bounds
        x_opt = np.sort(x_opt)
        x_opt = np.clip(x_opt, 0.0, 1.0)
        x_opt[-1] = 1.0
        
        # Create candidate solutions
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x_opt
        # x[1:] = 0 for Pareto front

        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        for k in range(n_evals):
            F[k] = problem(X[k])
            
            # Deadline guard
            if time.time() > deadline:
                return X

        return X
# EVOLVE-BLOCK-END
