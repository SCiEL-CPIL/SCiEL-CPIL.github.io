"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES with restarts for Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # Use a deterministic seed for reproducibility
        rng = np.random.default_rng(42)

        # Global best
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # CMA-ES parameters
        # lambda = population size
        lam = max(4, int(4 + 2 * np.log(dim)))  # For dim=5, lam=10
        mu = lam // 2  # Number of parents

        # Weights for ranking
        w_raw = np.log(np.arange(1, mu + 1))
        w_raw /= w_raw.sum()
        mu_w = (w_raw ** 2).sum()
        mu_eff = 1.0 / mu_w

        # CMA-ES hyperparameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 1) / (mu_eff + 2) ** 2 - 1))
        chi_n = np.sqrt(dim * (1 - 1 / (4 * dim) - 1 / (21 * dim ** 2)))

        def cmaes_run(x0, sigma_init, max_calls, rng, deadline, global_best_x, global_best_f):
            """
            Run CMA-ES from initial point x0 with step size sigma_init.
            Returns (best_x, best_f, calls_used)
            """
            nonlocal best_x, best_f, total_calls
            
            n = dim
            # State
            mean = x0.astype(float).copy()
            sigma = sigma_init
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)

            def update_C_sqrt():
                nonlocal B, D
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs
                C_sqrt = (B * D) @ B.T
                C_sqrt_inv = (B / D) @ B.T
                return C_sqrt, C_sqrt_inv

            C_sqrt, C_sqrt_inv = update_C_sqrt()

            calls_used = 0
            local_best_x = mean.copy()
            local_best_f = np.inf
            no_improve = 0
            stagnation_limit = 20  # Restart if no improvement for 20 generations

            # Initial evaluation of x0 if not already done
            if calls_used == 0:
                fx0 = func(mean)
                calls_used += 1
                total_calls += 1
                local_best_f = fx0
                local_best_x = mean.copy()
                if fx0 < global_best_f:
                    global_best_f = fx0
                    global_best_x = mean.copy()
                    best_f = fx0
                    best_x = mean.copy()

            while calls_used < max_calls:
                if time.time() > deadline:
                    break

                # Generate samples
                Z = rng.standard_normal((lam, n))
                X = mean[None, :] + sigma * (Z @ C_sqrt)
                X = np.clip(X, -5, 5)

                # Evaluate
                F = np.empty(lam)
                for i in range(lam):
                    if calls_used >= max_calls:
                        break
                    F[i] = func(X[i])
                    calls_used += 1
                    total_calls += 1

                if calls_used < lam:
                    F[calls_used:] = np.inf

                # Update best
                idx_min = int(np.argmin(F))
                if F[idx_min] < local_best_f:
                    local_best_f = F[idx_min]
                    local_best_x = X[idx_min].copy()
                    no_improve = 0
                    if local_best_f < global_best_f:
                        global_best_f = local_best_f
                        global_best_x = local_best_x.copy()
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                else:
                    no_improve += 1

                # Sort
                order = np.argsort(F)
                X_sorted = X[order]

                # Save old mean
                mean_old = mean.copy()

                # Update mean
                mean = mean_old + sigma * np.sum(w_raw[:, None] * X_sorted[:mu], axis=0)

                # Update ps
                ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_w) / sigma * (C_sqrt_inv @ (mean - mean_old))

                # Update pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used // lam + 1)))) < (1 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * c_c * (mean - mean_old) / sigma

                # Update C
                delta_sig = hsig * (1 - c_c) + c_c
                # Correct C update formula
                C = (1 - c_1 - c_mu) * C + c_1 * (n + delta_sig - 1) * C + c_mu * (np.outer(mean - mean_old, mean - mean_old) / sigma ** 2)
                
                # Symmetrize C to prevent drift
                C = (C + C.T) / 2

                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update C_sqrt
                C_sqrt, C_sqrt_inv = update_C_sqrt()

                # Check for stagnation
                if no_improve >= stagnation_limit:
                    break

            return local_best_x, local_best_f, calls_used

        # Phase 1: Initial diverse sampling to find a good starting point
        # Evaluate origin and a few random points
        init_points = [np.zeros(dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, dim))
        
        init_f = np.inf
        init_x = np.zeros(dim)
        
        for p in init_points:
            if total_calls >= budget:
                break
            fp = func(p)
            total_calls += 1
            if fp < init_f:
                init_f = fp
                init_x = p.copy()
        
        if init_f < best_f:
            best_f = init_f
            best_x = init_x.copy()

        # Phase 2: CMA-ES with restarts
        # We will run CMA-ES in chunks. If it stagnates, we restart from the best point found so far.
        # Budget allocation: 
        # - First run: 100 calls
        # - Subsequent runs: 50 calls each, restarting from best point with larger sigma
        # This allows for multiple attempts to escape local minima.
        
        current_best_x = best_x.copy()
        current_best_f = best_f
        
        # First run
        if total_calls < budget:
            max_calls_chunk = min(100, budget - total_calls)
            x_res, f_res, calls_used = cmaes_run(current_best_x, 2.0, max_calls_chunk, rng, deadline, best_x, best_f)
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()
            current_best_x = best_x.copy()
            current_best_f = best_f

        # Subsequent runs (restarts)
        restart_count = 0
        max_restarts = 5
        
        while total_calls < budget and restart_count < max_restarts:
            if time.time() > deadline:
                break
            
            # Increase sigma for restart to escape local minima
            sigma_restart = 2.0 * (1.0 + 0.5 * restart_count)
            sigma_restart = min(sigma_restart, 4.0)
            
            max_calls_chunk = min(50, budget - total_calls)
            
            # Restart from the current best point
            x_res, f_res, calls_used = cmaes_run(current_best_x, sigma_restart, max_calls_chunk, rng, deadline, best_x, best_f)
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()
                current_best_x = best_x.copy()
                current_best_f = best_f
                restart_count = 0 # Reset restart counter if improvement
            else:
                restart_count += 1
                
            # If we used very few calls, it might have stagnated immediately, try a different start?
            # For now, just continue to next restart or finish.

        return best_x, best_f
# EVOLVE-BLOCK-END
