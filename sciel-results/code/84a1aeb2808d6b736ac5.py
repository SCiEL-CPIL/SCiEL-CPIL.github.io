import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 
    with restarts for Schaffers F7 on [-5, 5]^5.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s/330s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_count = 0

        def time_ok():
            return time.time() < deadline

        def check_budget():
            nonlocal calls_count
            return calls_count < budget

        def eval_func(x):
            nonlocal calls_count, best_f, best_x
            if not time_ok() or not check_budget():
                raise RuntimeError("Budget or time exceeded")
            calls_count += 1
            # Clip to box
            x_clipped = np.clip(x, -5.0, 5.0)
            fy = func(x_clipped)
            if fy < best_f:
                best_f = fy
                best_x = x_clipped.copy()
            return fy

        # CMA-ES Parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of best individuals used for update
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        # Restart strategy:
        # We will run CMA-ES for a certain number of iterations or until stagnation.
        # If stagnation is detected, we restart from the best point found so far.
        
        max_restarts = 5
        # Allocate budget roughly equally, but allow more if early progress is good?
        # Fixed allocation is safer for time.
        calls_per_restart = budget // max_restarts
        
        # Initial random sampling to find a decent start
        # Use 10 calls for random search
        n_init = 10
        init_points = []
        for _ in range(n_init):
            if not time_ok() or not check_budget():
                break
            x_rand = rng.uniform(-5.0, 5.0, dim)
            fy = eval_func(x_rand)
            init_points.append((fy, x_rand))
            
        if best_x is None:
            # Fallback if no evals were made (shouldn't happen)
            return np.zeros(dim), 0.0

        # Sort init points
        init_points.sort(key=lambda x: x[0])
        
        for restart in range(max_restarts):
            if not time_ok() or not check_budget():
                break
                
            # Initialize CMA-ES state
            # Start from best point found so far, or center if first restart and no good point
            if restart == 0:
                x_mean = init_points[0][1].copy()
                # Estimate initial sigma from spread of good points
                good_points = np.array([p[1] for p in init_points[:5]])
                sigma = np.std(good_points, axis=0).mean()
                if sigma < 0.5:
                    sigma = 1.0
                if sigma > 2.0:
                    sigma = 2.0
            else:
                x_mean = best_x.copy()
                # Increase sigma for restart to escape local minima
                sigma = 1.5 + rng.uniform(0, 1.0)

            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Track progress for stagnation detection
            f_prev_best = best_f
            no_improve_count = 0
            max_no_improve = 50 # Generations without improvement before restart
            
            # Run CMA-ES for this restart
            while time_ok() and check_budget():
                # Check if we've used too many calls for this restart
                # We don't have a per-restart counter in the main loop easily, 
                # so we check global budget and stagnation.
                
                # Generate population
                y_list = []
                f_list = []
                
                for i in range(lam):
                    if not time_ok() or not check_budget():
                        break
                    z = rng.standard_normal(dim)
                    # Transform using B and D
                    # y = x_mean + sigma * B @ (D * z)
                    y = x_mean + sigma * (B @ (D * z))
                    fy = eval_func(y)
                    y_list.append(y)
                    f_list.append(fy)
                
                if len(y_list) < mu:
                    break # Not enough points generated, likely budget/time out
                
                y_array = np.array(y_list)
                f_array = np.array(f_list)
                
                # Sort by fitness
                order = np.argsort(f_array)
                y_sorted = y_array[order]
                f_sorted = f_array[order]
                
                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = y_sorted[0].copy()
                    
                # Stagnation check
                if best_f < f_prev_best - 1e-6:
                    f_prev_best = best_f
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > max_no_improve:
                    break # Stagnation, restart
                
                # Weighted mean of mu best
                m = np.sum(weights[:mu, None] * y_sorted[:mu], axis=0)
                
                # Update evolution paths
                # ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * (m - x_mean))) / sigma
                
                # pc
                # Check if ps is too large
                ps_norm = np.linalg.norm(ps)
                # Threshold for ps
                # Approximate threshold: (1.4 + 2/(dim+1)) * chi_n
                # Note: The standard formula involves the cumulative sum of (1-c_sigma)^t
                # For simplicity and robustness, we use a static threshold approximation
                # or the standard one. Let's use the standard one.
                # h_sigma = 1 if ps_norm / sqrt(1 - (1-c_sigma)**(2 * (calls_count - dim))) < (1.4 + 2/(dim+1)) * chi_n else 0
                # But calls_count - dim is not the generation number.
                # Let's use a simpler heuristic: if ps_norm < 1.4 * chi_n, h_sigma = 1
                # This is a common simplification.
                h_sigma = 1 if ps_norm < 1.4 * chi_n else 0
                
                if h_sigma == 1:
                    pc = (1 - c_c) * pc + c_c * (m - x_mean) / sigma
                else:
                    pc = (1 - c_c) * pc
                
                # Update Covariance Matrix C
                # Rank-1 update
                delta_h = 1 - h_sigma
                c1 = (2 / ((dim + 1.3) ** 2 + mu_eff))
                cmu = min(1 - c1, (2 * (mu_eff - 2) / ((dim + 2) ** 2 + mu_eff)) * (weights[0] + weights[1] if mu > 1 else 0))
                # Standard cmu calculation:
                # cmu = min(1 - c1, (2 * (mu_eff - 2) / ((dim + 2) ** 2 + mu_eff)) * (sum(weights[:mu]**2) / sum(weights[:mu])**2 - 1))
                # Actually, simpler:
                # c1 = 2 / ((dim + 1.3)^2 + mu_eff)
                # cmu = min(1 - c1, (2 * (mu_eff - 2) / ((dim + 2)^2 + mu_eff)) * (sum(w_i^2 for i in 1..mu) / (sum(w_i for i in 1..mu))^2 - 1))
                # Let's stick to the standard formula:
                c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
                # For cmu, a common approximation is:
                # cmu = min(1 - c1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + mu_eff) * (sum(weights[:mu]**2) / sum(weights[:mu])**2 - 1))
                # But weights are normalized.
                # Let's use:
                sum_w_sq = np.sum(weights[:mu] ** 2)
                sum_w = np.sum(weights[:mu])
                cmu = min(1 - c1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + mu_eff) * (sum_w_sq / (sum_w ** 2) - 1))
                
                # Update C
                # C = (1 - c1 - cmu) * C + c1 * outer(pc, pc) + cmu * sum(w_i * outer(y_i - x_mean, y_i - x_mean))
                # Wait, the standard update is:
                # C = (1 - c1 - cmu) * C + c1 * outer(pc, pc) + cmu * sum_{i=1..mu} w_i * outer(y_i - x_mean, y_i - x_mean)
                # Note: y_i - x_mean is in the transformed space? No, in the original space.
                # The formula is:
                # C_{t+1} = (1 - c1 - cmu) C_t + c1 pc_t pc_t^T + cmu sum_{i=1..mu} w_i (y_{i:t} - m_t)(y_{i:t} - m_t)^T
                # Wait, y_{i:t} - m_t is the difference from the weighted mean.
                
                # Calculate rank-mu part
                rank_mu_part = np.zeros((dim, dim))
                for i in range(mu):
                    diff = y_sorted[i] - m
                    rank_mu_part += weights[i] * np.outer(diff, diff)
                    
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * rank_mu_part
                
                # Ensure C is symmetric
                C = (C + C.T) / 2
                
                # Update B and D (eigen-decomposition)
                # We need B and D such that C = B @ diag(D^2) @ B.T
                # np.linalg.eigh returns eigenvalues in ascending order.
                evals, evecs = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                evals = np.maximum(evals, 1e-12)
                D = np.sqrt(evals)
                B = evecs
                
                # Update sigma
                # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / chi_n - 1))
                sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
                
                # Update mean
                x_mean = m
                
                # Optional: Clip x_mean to box? 
                # CMA-ES usually works in unbounded space, but we clip evaluations.
                # If x_mean goes out of bounds, the search might be biased.
                # We can clip x_mean to [-5, 5] to keep the center of mass in the box.
                x_mean = np.clip(x_mean, -5.0, 5.0)
                
                # If sigma is very small, we are converged
                if sigma < 1e-6:
                    break

        # Final polish with a few random restarts from best_x if time permits
        # This helps if CMA-ES got stuck in a very narrow local minimum.
        if time_ok() and check_budget():
            for _ in range(5):
                if not time_ok() or not check_budget():
                    break
                x_perturb = best_x + rng.normal(0, 0.5, dim)
                eval_func(x_perturb)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        return best_x, best_f
