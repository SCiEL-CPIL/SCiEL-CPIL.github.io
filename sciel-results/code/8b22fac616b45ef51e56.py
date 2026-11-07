# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Adaptive CMA-ES with Dynamic Stagnation-Triggered Restart for BBOB f7.
    
    Strategy:
    1. Run CMA-ES continuously.
    2. Monitor for stagnation (no improvement in best f over N generations).
    3. If stagnated, immediately restart CMA-ES from a new location.
       - 50% chance: New random point in [-5, 5]^d.
       - 50% chance: Global best point + large Gaussian noise.
    4. This adapts the exploration/exploitation balance dynamically based on 
       actual progress, rather than a fixed schedule.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lam = 6  # Population size
        mu = lam // 2  # Number of selected individuals
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) + cs)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Stagnation detection
        stagnation_generations = 0
        stagnation_threshold = 20  # Generations without improvement before restart
        best_f_in_run = np.inf
        best_x_in_run = None

        def _update_eigen(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            return B, D

        def run_cmaes(x0, sigma0, rng_local):
            nonlocal self, wrapped, dim, budget, deadline, rng, stagnation_generations, stagnation_threshold, best_f_in_run, best_x_in_run
            
            mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            
            B, D = _update_eigen(C)
            C_sqrt = B @ np.diag(D)
            C_sqrt_inv = B @ np.diag(1.0 / D)

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            local_best_f = np.inf
            local_best_x = mean.copy()
            no_improve_count = 0
            
            # Reset stagnation counter for this run
            stagnation_generations = 0
            best_f_in_run = np.inf
            best_x_in_run = mean.copy()

            while self._call_count < budget:
                if time.time() > deadline:
                    return False # Time out

                # Generate population
                population = []
                for i in range(lam):
                    if self._call_count >= budget:
                        break
                    z = rng_local.standard_normal(dim)
                    y = C_sqrt @ z
                    x_new = mean + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)
                    population.append(x_new)

                if not population:
                    return False

                # Evaluate
                fitness = []
                for x_new in population:
                    if self._call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    fx = wrapped(x_new)
                    fitness.append((fx, x_new))
                    
                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x_new.copy()

                if len(fitness) < mu:
                    return False

                fitness.sort(key=lambda t: t[0])
                selected = [f[1] for f in fitness[:mu]]

                # Check for improvement in this run
                if fitness[0][0] < best_f_in_run - 1e-10:
                    best_f_in_run = fitness[0][0]
                    best_x_in_run = selected[0].copy()
                    no_improve_count = 0
                    stagnation_generations = 0
                else:
                    no_improve_count += 1
                    stagnation_generations += 1

                # Stagnation check: If no improvement for threshold generations, restart
                if stagnation_generations >= stagnation_threshold:
                    return True # Signal to restart

                # Update mean
                mean_old = mean.copy()
                mean = np.zeros(dim)
                for i in range(mu):
                    mean += weights[i] * selected[i]

                # Update paths
                y_mean = C_sqrt_inv @ (mean - mean_old)
                p_c = (1 - cs) * p_c + cs * np.sqrt(mu_eff) * y_mean
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_mean

                # Update sigma
                h_sigma = (
                    np.linalg.norm(p_sigma) / (1 - (1 - cs) ** (no_improve_count * cs + 1))
                    < (1.4 + 2 / (dim + 1)) * chiN
                )
                if h_sigma:
                    sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chiN - 1))

                # Update C
                delta_h = (1 - h_sigma) * cs * (2 - cs)
                C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + delta_h * C)
                for i in range(mu):
                    yi = C_sqrt_inv @ (selected[i] - mean_old)
                    C += cmu * weights[i] * np.outer(yi, yi)

                C = 0.5 * (C + C.T)
                B, D = _update_eigen(C)
                C_sqrt = B @ np.diag(D)
                C_sqrt_inv = B @ np.diag(1.0 / D)

                # Convergence check
                if sigma < 1e-12:
                    return True # Signal to restart (converged)

            return False # Budget exhausted

        # Initial setup
        # Start with a point near center, but slightly offset to break symmetry if needed
        # Or just center.
        x0 = np.zeros(dim)
        sigma0 = 2.0 # Larger initial step to explore the box quickly
        
        # Main loop: Run CMA-ES until budget exhausted or stuck
        # We allow multiple restarts.
        # If a run finishes due to stagnation/convergence, we restart.
        # If it finishes due to budget, we stop.
        
        # To ensure we don't get stuck in a loop of immediate restarts, 
        # we vary the starting point.
        
        restart_count = 0
        max_restarts = 10 # Safety limit, though budget will likely run out first
        
        while self._call_count < budget and restart_count < max_restarts:
            if time.time() > deadline:
                break
            
            local_rng = np.random.default_rng(rng.integers(0, 2**31))
            
            # Determine starting point for this run
            if restart_count == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                # Strategy: 
                # 50% chance: Random point
                # 50% chance: Global best + noise
                if self._best_x is not None and rng.random() < 0.5:
                    # Perturb global best
                    noise_scale = 1.0 + 0.5 * rng.random()
                    x0 = self._best_x + rng.normal(0, noise_scale, dim)
                else:
                    # Random point
                    x0 = rng.uniform(-4.5, 4.5, dim)
                
                sigma0 = 1.5 + 0.5 * rng.random()
            
            should_restart = run_cmaes(x0, sigma0, local_rng)
            
            # If run_cmaes returned True, it means we should restart (stagnation/convergence)
            # If False, it means budget/time exhausted.
            if not should_restart:
                break
            
            restart_count += 1
            # Small pause to avoid infinite loop if something goes wrong? 
            # The call count increases, so we will eventually exit the budget check.

        # Final polish if we have remaining budget and found a good point
        # The CMA-ES usually refines well, but a tiny local search can help if 
        # we exited due to a hard stagnation limit while still having budget.
        if self._best_x is not None and self._call_count < budget:
             # Simple random walk around best
             remaining = budget - self._call_count
             # Use a small fraction for polish
             polish_budget = min(remaining, 10)
             for _ in range(polish_budget):
                 if self._call_count >= budget:
                     break
                 if time.time() > deadline:
                     break
                 x_try = self._best_x + rng.normal(0, 0.1, dim)
                 x_try = np.clip(x_try, -5, 5)
                 fx = wrapped(x_try)

        if self._best_x is None:
            self._best_x = np.zeros(dim)
            self._best_f = func(self._best_x)

        return self._best_x, self._best_f


# EVOLVE-BLOCK-END
