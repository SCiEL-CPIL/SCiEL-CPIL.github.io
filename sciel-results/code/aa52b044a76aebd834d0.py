"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, line_search

class Optimizer:
    """Multi-Start Powell Optimizer with Newton Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 325.0  # 300s limit, 25s margin
        
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(x)

        best_x = np.zeros(self.dim)
        best_fx = float('inf')
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-start Powell
        # Generate diverse starting points
        starts = [np.zeros(self.dim)]
        # Axis-aligned points to help detect the valley orientation
        for axis in range(self.dim):
            for sign in [1, -1]:
                pt = np.zeros(self.dim)
                pt[axis] = sign * 4.0
                starts.append(pt)
        # Random points
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, self.dim))

        for x0 in starts:
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
                
            try:
                # Use Powell with moderate tolerance to find a good basin
                res = minimize(
                    eval_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 100,
                        'ftol': 1e-6,
                        'xatol': 1e-4
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                break
            except Exception:
                continue

        # Phase 2: Newton Polish
        # If we have budget and time left, try to refine using a local quadratic model
        if self.calls < self.budget - 20 and time.time() - self.start_time < self.time_limit - 10:
            x_curr = best_x
            f_curr = best_fx
            
            # Estimate Hessian and Gradient at best_x
            # Use central differences for better accuracy
            h = 1e-5
            grad = np.zeros(self.dim)
            hess = np.zeros((self.dim, self.dim))
            
            # Gradient
            for i in range(self.dim):
                xp = x_curr.copy()
                xm = x_curr.copy()
                xp[i] += h
                xm[i] -= h
                fp = eval_func(xp)
                fm = eval_func(xm)
                grad[i] = (fp - fm) / (2 * h)
                
                # Hessian (only need to fill upper triangle and mirror)
                # We can stop if budget is tight, but 2*N+1 = 11 calls is small
                for j in range(i, self.dim):
                    xpp = x_curr.copy()
                    xpm = x_curr.copy()
                    xmp = x_curr.copy()
                    xmm = x_curr.copy()
                    
                    xpp[i] += h
                    xpp[j] += h
                    xpm[i] -= h
                    xpm[j] += h
                    xmp[i] += h
                    xmp[j] -= h
                    xmm[i] -= h
                    xmm[j] -= h
                    
                    fpp = eval_func(xpp)
                    fpm = eval_func(xpm)
                    fmp = eval_func(xmp)
                    fmm = eval_func(xmm)
                    
                    hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * h * h)
                    if i != j:
                        hess[j, i] = hess[i, j]

            # Check if Hessian is positive definite (or at least invertible)
            # Add small regularization if needed
            try:
                # Newton step: x_new = x - H^-1 * grad
                # Solve H * delta = grad
                delta = np.linalg.solve(hess + 1e-10 * np.eye(self.dim), grad)
                
                # Line search to ensure we decrease the function
                # scipy's line_search works with a function and a direction
                # We want to minimize f(x + alpha * delta)
                # Note: delta is the Newton step, so we move in direction -delta
                # Actually, line_search expects the direction of the step.
                # Let's define the direction as -delta
                dir_vec = -delta
                
                # Initial step size 1.0
                alpha, fc, f_new = line_search(
                    lambda a: eval_func(x_curr + a * dir_vec), 
                    0.0, 
                    1.0, 
                    f_curr, 
                    np.dot(grad, dir_vec), 
                    c1=1e-4, 
                    c2=0.9
                )
                
                if alpha > 0 and f_new < best_fx:
                    best_x = x_curr + alpha * dir_vec
                    best_fx = f_new
            except (StopIteration, np.linalg.LinAlgError):
                pass
            except Exception:
                pass

        # Phase 3: Final Nelder-Mead Polish
        # Nelder-Mead is robust and fast for local convergence
        if self.calls < self.budget and time.time() - self.start_time < self.time_limit - 5:
            try:
                res = minimize(
                    eval_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Ensure we return something valid
        if best_x is None or not np.all(np.isfinite(best_x)):
             best_x = np.zeros(self.dim)
             if self.calls < self.budget:
                 best_fx = eval_func(best_x)
             else:
                 best_fx = 0.0
                 
        return best_x, best_fx
# EVOLVE-BLOCK-END
