import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 320.0  # 320s margin for 330s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Exploration with Latin Hypercube Sampling
        # Allocate ~20% of budget for exploration/surrogate fitting
        n_explore = max(20, budget // 5) 
        if n_explore > budget - 10:
            n_explore = budget - 10
            
        # Generate LHS points
        # LHS ensures one point per bin in each dimension
        bins = np.linspace(-5, 5, n_explore)
        # Randomize order in each dimension to break correlation
        lhs_points = np.zeros((n_explore, dim))
        for d in range(dim):
            perm = np.random.permutation(n_explore)
            # Center of bins
            centers = (bins[:-1] + bins[1:]) / 2.0
            # Add random offset within bin
            offsets = np.random.uniform(0, bins[1] - bins[0], n_explore)
            lhs_points[:, d] = centers[perm] + offsets[perm]
            # Clip to bounds just in case
            lhs_points[:, d] = np.clip(lhs_points[:, d], -5, 5)

        # Evaluate exploration points
        X_explore = lhs_points
        F_explore = np.array([func(x) for x in X_explore])
        
        best_x = X_explore[np.argmin(F_explore)]
        best_f = np.min(F_explore)
        
        calls_made = n_explore
        
        # Check deadline
        if time.time() > self.deadline - 5.0:
            return best_x, best_f

        # Phase 2: Fit Quadratic Surrogate
        # Design matrix for quadratic model: [1, x_i, x_i^2, x_i*x_j]
        def design_matrix(X):
            n, d = X.shape
            # 1 constant term
            # d linear terms
            # d quadratic terms
            # d*(d-1)/2 interaction terms
            n_features = 1 + d + d + d * (d - 1) // 2
            M = np.zeros((n, n_features))
            M[:, 0] = 1.0
            M[:, 1:1+d] = X
            # Quadratic terms
            for i in range(d):
                M[:, 1 + d + i] = X[:, i] ** 2
            # Interaction terms
            col_idx = 1 + d + d
            for i in range(d):
                for j in range(i + 1, d):
                    M[:, col_idx] = X[:, i] * X[:, j]
                    col_idx += 1
            return M

        M_explore = design_matrix(X_explore)
        
        # Fit least squares
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(M_explore, F_explore, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = None

        # Phase 3: Find minimum of surrogate
        x_start = best_x
        if coeffs is not None:
            # Define surrogate function
            def surrogate(x):
                x = np.clip(x, -5, 5)
                M_x = design_matrix(x.reshape(1, -1))
                return float(np.dot(M_x, coeffs))
            
            # Use scipy minimize to find min of surrogate
            # We can try multiple starts from the best exploration points
            from scipy.optimize import minimize
            
            best_surrogate_x = best_x
            best_surrogate_f = surrogate(best_x)
            
            # Try optimizing from the top 5 best exploration points
            top_indices = np.argsort(F_explore)[:5]
            for idx in top_indices:
                res = minimize(surrogate, X_explore[idx], method='L-BFGS-B', bounds=[(-5, 5)] * dim, options={'maxiter': 50})
                if res.fun < best_surrogate_f:
                    best_surrogate_f = res.fun
                    best_surrogate_x = res.x
            
            x_start = np.clip(best_surrogate_x, -5, 5)
        
        # Phase 4: Local Search (1+1)-ES
        x = x_start.copy()
        fx = func(x)
        calls_made += 1
        
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        sigma = 0.5 # Initial step size
        no_improve = 0
        max_no_improve = 20
        
        # Remaining budget
        remaining = budget - calls_made
        
        for i in range(remaining):
            # Deadline check
            if time.time() > self.deadline - 1.0:
                break
                
            # Generate candidate
            y = x + sigma * np.random.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1 # Increase step size
                no_improve = 0
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9 # Decrease step size
                no_improve += 1
                
                # Stagnation handling: if stuck, try a larger leap or restart
                if no_improve >= max_no_improve:
                    # Restart from a random point near the best found so far, or a random LHS point
                    # Let's try a large jump from current x
                    jump_sigma = 2.0
                    y_jump = x + jump_sigma * np.random.standard_normal(dim)
                    y_jump = np.clip(y_jump, -5, 5)
                    fy_jump = func(y_jump)
                    calls_made += 1
                    
                    if fy_jump < fx:
                        x = y_jump
                        fx = fy_jump
                        sigma = 1.0
                        no_improve = 0
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # If jump failed, reset sigma and try again
                        sigma = 0.5
                        no_improve = 0
                        # If we have used all budget, break
                        if calls_made >= budget:
                            break

        return best_x, best_f
