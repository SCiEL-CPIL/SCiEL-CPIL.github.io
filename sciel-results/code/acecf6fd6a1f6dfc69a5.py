import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                pt = pt.reshape(dim)
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Sphere-Specific Model Construction
        # Assume f(x) = a * ||x||^2 + q^T x + c
        # Points: 0, +/- d * e_i
        d = 2.0
        points = [np.zeros(dim)]
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = d
            p2 = np.zeros(dim); p2[i] = -d
            points.append(p1)
            points.append(p2)
            
        values = np.array([evaluate(p) for p in points])
        
        c_val = values[0]
        a_vals = []
        q_vec = np.zeros(dim)
        
        for i in range(dim):
            val_pos = values[1 + 2*i]
            val_neg = values[2 + 2*i]
            
            # f(d*e_i) = a*d^2 + q_i*d + c
            # f(-d*e_i) = a*d^2 - q_i*d + c
            # Diff: 2*q_i*d = val_pos - val_neg
            # Sum: 2*a*d^2 + 2*c = val_pos + val_neg
            
            q_i = (val_pos - val_neg) / (2 * d)
            a_i = (val_pos + val_neg - 2*c_val) / (2 * d * d)
            
            a_vals.append(a_i)
            q_vec[i] = q_i
            
        a = np.mean(a_vals)
        
        if abs(a) < 1e-12:
            a = 1e-12 * np.sign(a) if a != 0 else 1e-12
            
        # Minimum at x* = -q / (2*a)
        x_init = -q_vec / (2 * a)
        x_init = np.clip(x_init, -5, 5)
        
        # Evaluate the estimated optimum
        f_init = evaluate(x_init)
        
        # Phase 2: CMA-ES Polish
        # Initialize CMA-ES parameters
        x = best_x.copy()
        sigma = 1.0 # Initial step size
        
        # For Sphere, the Hessian is isotropic. 
        # If a is significantly different from 1, scale sigma?
        # f(x) ~ a * ||x - x*||^2. 
        # The "width" of the valley is related to 1/sqrt(a).
        # Let's start with a sigma that is a fraction of the distance from 0 to x_init, or just a fixed small value.
        # Given we are very close, a small sigma is safer to avoid overshooting.
        # However, CMA-ES adapts sigma.
        
        # Population size for CMA-ES
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(np.arange(1, mu+1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # CMA-ES state
        C = np.eye(dim)
        Cs = np.zeros(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # Damping for step size control (1/5-th rule equivalent in CMA-ES is handled by sigma update)
        # Standard CMA-ES sigma update:
        # sigma = sigma * exp(c_sigma * (||ps|| / sigma_norm - 1) / sqrt(mu_eff))
        
        sigma_norm = np.sqrt(2 * (dim + 1) ** 0.8 - dim)
        
        # Initial sigma: 
        # If we are at the minimum, we want to explore locally.
        # The model error might be small. 
        # Let's estimate the residual from the model fit to set an initial sigma.
        # residual = sum((f(x) - model(x))^2)
        # For now, use a heuristic: if x_init is close to 0, start small.
        # Actually, 1.0 is a reasonable start for a box of size 10, but we are already at the min.
        # Let's start with 0.1 to be safe, or 1.0 and let it shrink.
        # CMA-ES usually starts with 1.0.
        sigma = 1.0
        
        # Run CMA-ES
        while calls_made < budget and time.time() < deadline:
            # Generate population
            Y = []
            for k in range(lambda_):
                B = np.linalg.cholesky(C)
                z = np.random.randn(dim)
                y = B @ z
                Y.append(y)
            Y = np.array(Y)
            
            # Sample
            X = np.array([x + sigma * y for y in Y])
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([evaluate(x_i) for x_i in X])
            
            # Sort
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update mean
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update Covariance Matrix
            Y_inv_sigma = (X_sorted[:mu] - x) / sigma
            delta_x = x_new - x
            x = x_new
            
            # Rank-one update
            C = (1 - c_c) * C + c_c * (np.outer(delta_x/sigma, delta_x/sigma) + (1 - (1-c_c)*c_sigma) * np.outer(ps/sigma, ps/sigma))
            # Note: The standard update is:
            # C = (1 - c_c) * C + c_c * ( (1 - (1-c_c)*c_sigma) * y_plus^T y_plus + sum w_i y_i^T y_i )
            # where y_plus = ps / sigma_norm, y_i = (x_i - x_old) / sigma
            # Let's use the standard formula more carefully.
            
            # Recalculate C update properly
            C = (1 - c_c) * C 
            # y_plus
            y_plus = ps / sigma_norm
            C += c_c * ( (1 - (1-c_c)*c_sigma) * np.outer(y_plus, y_plus) + np.sum(weights[:, None] * (Y_inv_sigma.T * Y_inv_sigma), axis=0) )
            
            # Ensure symmetry and positive definite
            C = (C + C.T) / 2
            
            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff * c_c * (2 - c_c)) * delta_x / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(mu_eff * c_sigma * (2 - c_sigma)) * delta_x / sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma / sigma_norm * (np.linalg.norm(ps) / sigma_norm - 1))
            
            # Convergence check
            if sigma < 1e-12:
                break
                
            # If we made no progress in a while, we might be stuck, but CMA-ES handles this well.
            
        return best_x, best_f
