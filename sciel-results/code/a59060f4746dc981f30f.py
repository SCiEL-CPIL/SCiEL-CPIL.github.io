"""CMA-ES Optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin before 330s kill
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(d))
        # Step size
        sigma = 2.0  # Initial step size, slightly larger than standard 1.0 to cover [-5,5] quickly
        
        # Start from center or a random point? Center is usually safe for BBOB.
        # Let's start at 0.
        m = np.zeros(d)
        
        # Covariance matrix
        C = np.eye(d)
        
        # Eigen decomposition cache
        B = np.eye(d)
        D = np.ones(d)
        
        # Adaptation rates
        cs = (np.sqrt(lam + 2) - 1) / (lam + np.sqrt(2) + 3)
        ds = 1 + 2 * max(0, (cs - 1) / (lam - 1)) + cs
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        calls_used = 0
        best_x = m.copy()
        best_f = float('inf')
        
        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return None
            if time.time() > deadline - 2:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Initial evaluation
        f0 = try_eval(m)
        if f0 is None:
            return best_x, best_f
        best_f = f0

        # CMA-ES Loop
        while calls_used < budget and time.time() < deadline - 10:
            # 1. Update eigen decomposition of C if necessary (simplified: do it every iteration for robustness in small d)
            # In standard CMA-ES, this is updated less frequently, but for d=5, it's cheap.
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-16))
            
            # 2. Sample offspring
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if calls_used >= budget or time.time() > deadline - 2:
                    break
                    
                # Standard normal sample
                zi = np.random.standard_normal(d)
                # Transform to search space
                yi = B @ (D * zi)
                xi = m + sigma * yi
                
                fi = try_eval(xi)
                if fi is None:
                    break
                    
                ys.append(yi)
                xs.append(xi)
                fs.append(fi)
            
            if not ys:
                break
                
            # Convert to arrays
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # 3. Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Use only top mu for mean update
            ys_used = ys_sorted[:mu]
            xs_used = xs_sorted[:mu]
            fs_used = fs_sorted[:mu]
            
            # 4. Update mean
            m_old = m.copy()
            m = np.sum(weights[:, np.newaxis] * xs_used, axis=0)
            
            # 5. Update path and step size
            # pc = (1 - cs) * pc + cs * sqrt(mueff) / sigma * (m - m_old)
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) / sigma * (m - m_old)
            
            # ps = (1 - ds) * ps + ds * sqrt(mueff) / sigma * B^T @ (m - m_old)
            # Note: B^T @ (m - m_old) is the step in the principal axes coordinates
            ps = (1 - ds) * ps + ds * np.sqrt(mueff) / sigma * (B.T @ (m - m_old))
            
            # Hessian norm check for sigma adaptation
            norm_ps_sq = np.dot(ps, ps)
            sigma = sigma * np.exp(cs / ds * (norm_ps_sq / d - 1) / ds)
            
            # 6. Update Covariance Matrix
            # C = (1 - mu_1 - mu_mu) * C + mu_1 * pc pc^T + mu_mu * sum(weights_i * (y_i - y_bar)(y_i - y_bar)^T)
            
            # y_bar is the weighted mean of the successful steps
            y_bar = np.sum(weights[:, np.newaxis] * ys_used, axis=0)
            
            # Rank-1 update
            c1_term = c1 * np.outer(pc, pc)
            
            # Rank-mu update
            diff = ys_used - y_bar
            rank_mu_term = cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->ijk', weights, diff, diff), axis=0)
            
            C = (1 - c1 - cmu) * C + c1_term + rank_mu_term
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Check convergence
            if sigma < 1e-15:
                break
                
            # If we are stuck or not improving, break early to save time for polishing
            # Simple stagnation check: if best_f hasn't improved in last 'lam' iterations, break
            # (Implemented implicitly by the loop condition and budget)
            
            # Optional: If progress is very slow, break
            # For now, let it run until budget or time limit.
            
        # Phase 2: Polish
        # Use remaining budget for a simple local search around the best point found by CMA-ES.
        # A simple Nelder-Mead or coordinate descent can be effective here.
        # Given the time, a few steps of a simple gradient-free local search.
        
        if calls_used < budget and time.time() < deadline - 5 and best_f != float('inf'):
            x_curr = best_x.copy()
            f_curr = best_f
            step_size = 0.1
            
            for _ in range(20):  # Max 20 polish steps
                if calls_used >= budget or time.time() > deadline - 2:
                    break
                
                improved = False
                # Try moving in each dimension
                for i in range(d):
                    if calls_used >= budget or time.time() > deadline - 2:
                        break
                    # Try +
                    x_test = x_curr.copy()
                    x_test[i] += step_size
                    f_test = try_eval(x_test)
                    if f_test is not None and f_test < f_curr:
                        x_curr = x_test
                        f_curr = f_test
                        improved = True
                    # Try -
                    x_test = x_curr.copy()
                    x_test[i] -= step_size
                    f_test = try_eval(x_test)
                    if f_test is not None and f_test < f_curr:
                        x_curr = x_test
                        f_curr = f_test
                        improved = True
                
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-5:
                        break
                else:
                    # Keep step size or increase slightly if we are making good progress?
                    # Usually keep or decrease slowly. Let's keep it.
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
