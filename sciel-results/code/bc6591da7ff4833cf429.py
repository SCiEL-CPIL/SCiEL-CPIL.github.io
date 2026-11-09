"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with random global exploration and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def check_time():
            return calls_used < budget and time.time() < deadline - 5.0

        def check_time_early():
            return calls_used < budget and time.time() < deadline - 20.0

        rng = np.random.default_rng(12345)

        # Phase 1: Global exploration with random sampling
        # Use 200 random samples to get good initial candidates
        n_global = 200
        global_candidates = []
        for _ in range(n_global):
            if not check_time_early():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            global_candidates.append((x, fx))
            try_update(x, fx)

        # Sort by fitness and take top candidates for CMA-ES
        global_candidates.sort(key=lambda t: t[1])
        top_k = min(3, len(global_candidates))
        start_points = [x for x, f in global_candidates[:top_k]]

        # Phase 2: CMA-ES runs from best candidates
        # Each CMA-ES run gets ~100 evaluations
        cma_budget_per_run = max(50, (budget - calls_used) // max(top_k, 1))
        
        for start_idx, start_x in enumerate(start_points):
            if not check_time():
                break
            
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            sigma = 1.0  # initial step size
            mean = start_x.copy()
            C = np.eye(dim)  # covariance matrix
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            
            # Evolution paths
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cs = (pop_size + 2) / (dim + 4)
            cm = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 3)) - 1) / (dim + 1)
            damps = (dim + 1) / 2 + cs * cm
            dampc = (5 + dim) / (dim + 2) + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 3)) - 1)
            
            # Track best in this run
            run_best_x = mean.copy()
            run_best_f = np.inf
            
            # Evaluate starting point
            f_start = func(mean)
            calls_used += 1
            try_update(mean, f_start)
            run_best_f = f_start
            run_best_x = mean.copy()
            
            # CMA-ES main loop
            max_cma_iters = cma_budget_per_run // pop_size + 5
            for iteration in range(max_cma_iters):
                if not check_time():
                    break
                
                # Generate population
                archive = []
                for i in range(pop_size):
                    # Sample from N(mean, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    # Use Cholesky decomposition for sqrt(C)
                    try:
                        C_chol = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C_chol = np.eye(dim)
                    x_i = mean + sigma * (C_chol @ z)
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    calls_used += 1
                    archive.append((x_i, f_i))
                    try_update(x_i, f_i)
                    if f_i < run_best_f:
                        run_best_f = f_i
                        run_best_x = x_i.copy()
                
                # Sort by fitness
                archive.sort(key=lambda t: t[1])
                lam = pop_size // 2  # number of parents
                
                # Update mean
                weights = np.ones(lam) / lam  # uniform weights for simplicity
                new_mean = np.zeros(dim)
                for i in range(lam):
                    new_mean += weights[i] * archive[i][0]
                
                # Update sigma
                sigma_old = sigma
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(new_mean - mean) / (sigma * np.sqrt(2 * lam)) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update covariance matrix
                y = (new_mean - mean) / (sigma * np.sqrt(2 * lam))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** 2) else 0
                pc = (1 - cm) * pc + hsig * cm * y
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((dim, dim))
                for i in range(lam):
                    z_i = (archive[i][0] - mean) / (sigma * np.sqrt(2 * lam))
                    rank_mu += weights[i] * np.outer(z_i, z_i)
                
                C = (1 + cm * (hsig * (1 - dampc) - 1) + mu_eff * cm) * C + rank_one + rank_mu
                C = (C + C.T) / 2  # symmetrize
                
                # Update mean
                mean = new_mean
                
                # Check convergence
                if sigma < 1e-8 or iteration > 200:
                    break
            
            # After CMA-ES, do local refinement around the best point found in this run
            if check_time() and not np.isnan(run_best_x).any():
                x = run_best_x.copy()
                fx = run_best_f
                sigma_local = 0.1
                
                no_improve = 0
                max_no_improve = 15
                
                while check_time() and no_improve < max_no_improve:
                    y = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        try_update(x, fx)
                        sigma_local *= 1.05
                        no_improve = 0
                    else:
                        sigma_local *= 0.9
                        no_improve += 1

        # Phase 3: Final coordinate-wise refinement if time remains
        if check_time() and not np.isnan(best_x).any():
            x = best_x.copy()
            fx = best_f
            step = 0.1
            for coord in range(dim):
                if not check_time():
                    break
                for _ in range(5):
                    if not check_time():
                        break
                    y = x.copy()
                    y[coord] += step
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        try_update(x, fx)
                    else:
                        y = x.copy()
                        y[coord] -= step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < fx:
                            x, fx = y, fy
                            try_update(x, fx)
                        else:
                            break
                step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
