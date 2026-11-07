import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        
        # Weights
        weights = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        weights /= weights.sum()
        mu = int(lam / 2)
        w = weights[:mu]
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) * (d + 1)) / (d + mueff + 5) - 1) + c_sigma
        c_c = (4 - mueff + c_sigma) / (d + mueff + 9)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff))
        
        # Initialization
        rng = np.random.default_rng(42)
        
        # Start from center or random? 
        # The problem states minimizer is in [-4,4]^5. 
        # Starting at 0 is a good guess for many BBOB functions if not shifted, 
        # but they are shifted. 
        # Let's start at a random point in [-4, 4] to be safe, or 0?
        # Uniform random in [-5, 5] is the seed's approach.
        # Let's try starting at 0.0, as it's the center of the box. 
        # If the shift is hidden, 0 might be far. 
        # However, CMA-ES is good at moving.
        # Let's use a random start to avoid bias, but constrained to [-4, 4] to be closer to likely opt.
        x_mean = rng.uniform(-4, 4, d)
        
        # Initial sigma
        # Box size is 10. 
        # Typical initial sigma is 1/3 to 1/2 of the box width? 
        # Or based on the distance to the center?
        # Let's use 2.0 as in the seed, or maybe 3.0 for faster initial exploration.
        sigma = 2.0
        
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Initial evaluation
        x0 = x_mean
        f0 = func(x0)
        best_x = x0
        best_f = f0
        calls = 1
        
        # Store points and fitness for the first generation? 
        # CMA-ES generates a population, evaluates, then updates.
        # We need to manage the budget carefully.
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Determine how many more evals we can do
            remaining = budget - calls
            if remaining < lam:
                # If not enough for a full generation, do single steps (1+1) or just break?
                # Doing partial generations is complex. 
                # If remaining is small, switch to (1+1)-ES with adaptive sigma?
                # Or just break and return best.
                # Let's try to finish a full generation if possible, otherwise stop.
                # Actually, if remaining < lam, we can do (remaining) single steps.
                # But CMA-ES update requires a full population.
                # Let's just break if we can't do a full population.
                # Or, we can reduce lam? No, that changes the algorithm.
                # Let's break. The AOCC penalizes lack of progress, but 
                # doing random single steps might not help much.
                # However, we can do a simple (1+1)-ES for the remaining budget.
                break
            
            # Generate population
            ys = []
            for i in range(lam):
                # Sample from N(0, C)
                # z ~ N(0, I)
                z = rng.standard_normal(d)
                # y = B * (D * z)
                y = B @ (D * z)
                ys.append(y)
            
            # Evaluate
            xs = []
            fs = []
            for y in ys:
                if calls >= budget:
                    break
                x_new = x_mean + sigma * y
                # Clip to box to ensure valid input (though func clips, 
                # clipping here avoids potential issues if func doesn't handle 
                # out of bounds gracefully, though contract says it clips).
                # Contract: "func(x) clips ... to the box". So we can pass x_new directly.
                f_new = func(x_new)
                xs.append(x_new)
                fs.append(f_new)
                calls += 1
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new
            
            if len(fs) < lam:
                # We ran out of budget in the middle of a generation.
                # We can't do a proper CMA-ES update.
                break

            # Sort by fitness
            idx = np.argsort(fs)
            xs = np.array(xs)[idx]
            fs = np.array(fs)[idx]
            
            # Update mean
            # x_mean = sum(w_i * x_i) for i in 1..mu
            x_mean_new = np.sum(w[:, None] * xs[:mu], axis=0)
            
            # Update evolution paths
            # y_sigma = (x_mean_new - x_mean) / sigma
            y_sigma = (x_mean_new - x_mean) / sigma
            
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < (1.4 + 2/(d+1)) * E||N(0,I)||
            # Approximation for h_sigma
            # chi_d = sqrt(d) * (1 - 1/(4d) + 1/(21d^2))
            chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            # We don't track 'gen' explicitly in the loop count easily for the power term, 
            # but we can use the standard heuristic:
            # h_sigma = 1 if ||p_sigma|| < (1.4 + 2/(d+1)) * chi_d
            # This is an approximation.
            if np.linalg.norm(p_sigma) < (1.4 + 2/(d+1)) * chi_d:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            
            # Update Covariance Matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_sigma^2) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * y_i * y_i.T)
            # Note: y_i here are the sorted y vectors corresponding to the top mu individuals.
            # We need to reconstruct y_i from x_i: y_i = (x_i - x_mean_old) / sigma
            # Note: x_mean_old is the mean BEFORE the update.
            
            # Calculate y vectors for the top mu
            y_sorted = (xs[:mu] - x_mean) / sigma
            
            delta_h = (1 - h_sigma**2) * c_c * (2 - c_c)
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(p_c, p_c) + delta_h * C)
            
            # Add the rank-mu update
            # sum(w_i * y_i * y_i.T)
            # y_sorted is (mu, d). We need weighted outer products.
            # Can be done as: y_sorted.T @ (w * y_sorted)
            # w is (mu,). y_sorted is (mu, d).
            # (w * y_sorted) is (mu, d).
            # y_sorted.T @ (w * y_sorted) is (d, d).
            C_new += c_mu * (y_sorted.T @ (w[:, None] * y_sorted))
            
            # Ensure C is positive definite (symmetrize)
            C_new = (C_new + C_new.T) / 2
            
            # Update Cholesky/Eigen decomposition
            # We need B and D such that C = B @ diag(D^2) @ B.T
            # Since C is symmetric positive definite, we can use eigendecomposition.
            # For d=5, this is fast.
            eigenvalues, B_new = np.linalg.eigh(C_new)
            
            # Clip small negative eigenvalues to 0 (numerical stability)
            eigenvalues = np.maximum(eigenvalues, 1e-16)
            D_new = np.sqrt(eigenvalues)
            
            # Update state
            x_mean = x_mean_new
            C = C_new
            B = B_new
            D = D_new
            
            # Update sigma
            # sigma = sigma * exp(c_sigma * (||p_sigma|| / chi_d - 1))
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
            
            # Optional: Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # If sigma becomes very small, we might have converged.
            # We can break early if sigma < 1e-6 and best_f hasn't changed much?
            # But budget is only 500, so let's run to completion or time out.
            
        return best_x, best_f
