"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        # CMA-ES Parameters
        # Standard population size for dim=5
        # lambda = 4 + floor(3 * log(mu)) is common, but let's use fixed standard values
        # For dim 5, lambda=9 is a good standard choice
        lam = 4 + int(3 * np.log(1 + dim))  # 4 + 3*1.609 = 8.8 -> 8? No, 4+3*log(6) ~ 8.8. Let's use 9.
        lam = 9
        mu = lam // 2  # 4
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        c1 = (2 / ((dim + 1.3)**2 + mueff)) / 10
        cmu = min(1 - c1, 2 * ((mueff - 2 + 2 / dim) / ((dim + 2)**2 + mueff))) / 10
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        csigma = 1 / ds
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Initialization
        # Start at origin, which is a good heuristic for BBOB if shifts aren't huge, 
        # but random is safer. Let's use a random start within [-2, 2] to be close to center.
        x0 = np.zeros(dim)
        sigma0 = 2.0
        
        # State
        xmean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # History
        best_x = x0.copy()
        best_f = np.inf
        
        # Precompute B and D for sqrt(C)
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-10))
        
        calls_used = 0
        
        # Evaluation helper
        def eval_point(x):
            nonlocal best_x, best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Initial evaluation
        f0 = eval_point(x0)
        calls_used += 1
        
        # Main Loop
        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            pop_x = np.zeros((lam, dim))
            pop_f = np.zeros(lam)
            
            for i in range(lam):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                z = np.random.standard_normal(dim)
                # y = B * D * z
                y = B @ (D * z)
                xi = xmean + sigma * y
                # Clip to box
                xi = np.clip(xi, -5, 5)
                pop_x[i] = xi
                pop_f[i] = eval_point(xi)
                calls_used += 1
            
            # If budget ran out inside population generation, break
            if calls_used >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(pop_f)
            pop_x = pop_x[idx]
            pop_f = pop_f[idx]
            
            # Update mean
            xmean_old = xmean.copy()
            xmean = np.sum(weights[:, np.newaxis] * pop_x[:mu], axis=0)
            
            # Update step size control variables
            # pc = (1-cs)*pc + cs * hs * 1/sigma * B @ (D * (xmean_old - xmean))
            # Actually, the standard formula is:
            # pc = (1-cs)*pc + hs * cs / sigma * (xmean_new - xmean_old)
            # where hs = 1 if ||pc/sigma|| / chi_n < (1.4 + 2/(dim+1)) * chi_n
            # Note: In standard CMA-ES, the update is in the x-space or y-space?
            # The standard implementation updates pc in the y-space (standard normal space) or x-space?
            # Let's stick to the standard x-space update for pc, which is:
            # pc = (1-cs)*pc + cs * (xmean_new - xmean_old) / sigma
            # Wait, the standard CMA-ES (Hansen) updates pc in the "search space" scaled by 1/sigma.
            # Let's use the formulation:
            # y_mean = (xmean_new - xmean_old) / sigma
            # pc = (1-cs)*pc + cs * hs * y_mean
            y_mean = (xmean - xmean_old) / sigma
            
            # Check hs condition
            # ||pc|| / sigma? No, pc is already scaled.
            # The condition is: ||pc|| / chi_n < (1.4 + 2/(dim+1)) * chi_n
            # Actually, the standard check is on the norm of pc relative to chi_n.
            # Let's compute norm of y_mean to check for hs
            # The condition is: ||pc|| < (1.4 + 2/(dim+1)) * chi_n
            # Wait, pc is accumulated.
            # Let's use the robust check:
            # if ||pc|| / chi_n < (1.4 + 2/(dim+1)): hs = 1 else hs = 0
            # But pc is updated using hs. So we need to check the previous pc?
            # Standard algorithm:
            # 1. Compute y_mean
            # 2. Compute pc_norm = ||pc||
            # 3. hs = 1 if pc_norm / chi_n < (1.4 + 2/(dim+1)) * chi_n else 0
            #    Actually, the condition is usually: ||pc|| < (1.4 + 2/(dim+1)) * chi_n
            #    Wait, the formula in Hansen's paper is:
            #    hs = 1 if ||pc|| / chi_n < (1.4 + 2/(dim+1)) * chi_n
            #    This simplifies to ||pc|| < (1.4 + 2/(dim+1)) * chi_n^2 ? No.
            #    Let's look at the standard CMA-ES pseudocode:
            #    pc = (1-cs)*pc + hs * cs * (xmean_new - xmean_old)/sigma
            #    hs = 1 if ||pc||/chi_n < (1.4 + 2/(dim+1))*chi_n else 0
            #    This is slightly circular. The standard implementation checks the norm of the *new* pc?
            #    No, it checks the norm of the *old* pc to decide hs for the *new* pc update?
            #    Actually, most implementations compute the norm of the *updated* pc?
            #    Let's use a simpler heuristic or the standard one:
            #    The condition is typically:
            #    if np.linalg.norm(pc) / chi_n < 1.4 + 2/(dim+1):
            #        hs = 1
            #    else:
            #        hs = 0
            #    This is applied to the *current* pc before updating?
            #    Let's assume the standard check on the *current* pc.
            
            pc_norm = np.linalg.norm(pc)
            if pc_norm / chi_n < (1.4 + 2 / (dim + 1)) * chi_n:
                hs = 1.0
            else:
                hs = 0.0
                
            pc = (1 - cs) * pc + hs * cs * y_mean
            
            # Update ps
            # ps = (1-csigma)*ps + csigma * hs * 1/sigma * B @ (D * (xmean_new - xmean_old))
            # ps = (1-csigma)*ps + csigma * hs * y_mean
            ps = (1 - csigma) * ps + csigma * hs * y_mean
            
            # Update C
            # C = (1 + c1 * (2 - c1) - c1 - cmu * sum(w_i^2)) * C
            #    + c1 * (pc * pc^T + (1-hs)*c1*hs*C)
            #    + cmu * sum(w_i * (y_i - y_mean) * (y_i - y_mean)^T)
            
            # Calculate y_i - y_mean for the top mu individuals
            # y_i = (x_i - xmean_old) / sigma
            # But we generated x_i using xmean_old. So y_i = B @ (D * z_i)
            # We didn't store z_i. We can reconstruct y_i = (pop_x[i] - xmean_old) / sigma
            # However, clipping might distort this. For CMA-ES on bounded domains, 
            # it's better to work in the unbounded space if possible, but here we clip.
            # Using (x_i - xmean_old)/sigma is an approximation.
            
            delta_y = (pop_x[:mu] - xmean_old) / sigma
            delta_y_minus_mean = delta_y - y_mean
            
            # Sum of outer products
            sum_outer = np.sum(weights[:, np.newaxis, np.newaxis] * 
                               delta_y_minus_mean[:, :, np.newaxis] * 
                               delta_y_minus_mean[:, np.newaxis, :], 
                               axis=0)
            
            # Update C
            c_c = (1 + c1 * (2 - c1) - c1 - cmu * np.sum(weights**2))
            C = c_c * C + c1 * (np.outer(pc, pc) + (1 - hs) * c1 * hs * C) + cmu * sum_outer
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            
            # Update sigma
            # sigma = sigma * exp(cs/dlambda * (||ps||/chi_n - 1))
            ps_norm = np.linalg.norm(ps)
            sigma = sigma * np.exp(cs / ds * (ps_norm / chi_n - 1))
            
            # Cap sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update B and D
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            
            # Check for convergence
            if sigma < 1e-8:
                break
                
        # Return best found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
