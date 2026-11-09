"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Optimizer using CMA-ES with PCA-based covariance initialization for BBOB f18 (Schaffers F7 ill-conditioned).
    
    Key improvements:
    1. PCA-based covariance initialization from early samples to align with funnel structure
    2. Tuned sigma adaptation for more stable convergence
    3. Adaptive restart strategy based on progress
    4. Efficient sampling and evaluation
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        # CMA-ES Parameters
        lambda_pop = 4 + int(3 * math.log(dim))
        if lambda_pop < 4:
            lambda_pop = 4
            
        # Weights
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Update rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2)**2 + mueff))
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, math.sqrt((mu - 1) / (dim + 1)) - 1) + cs
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0
        
        # Initial exploration: sample points to estimate funnel direction
        # Use a small set of random points to do PCA and initialize covariance
        n_init_samples = min(lambda_pop * 2, 20)
        init_points = np.random.randn(n_init_samples, dim) * 3.0
        init_points = np.clip(init_points, -5, 5)
        
        init_fvals = []
        for x in init_points:
            if calls >= total_budget:
                break
            if time.time() > deadline - 10.0:
                init_fvals.append(float('inf'))
                continue
            try:
                f = func(x)
                calls += 1
                init_fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            except:
                init_fvals.append(float('inf'))
        
        if len(init_fvals) < n_init_samples:
            init_fvals.extend([float('inf')] * (n_init_samples - len(init_fvals)))
        
        # Filter out inf values for PCA
        valid_mask = np.isfinite(init_fvals)
        if np.sum(valid_mask) >= dim:
            valid_points = init_points[valid_mask]
            # Weight by inverse fitness (lower fitness = higher weight)
            valid_fvals = init_fvals[valid_mask]
            min_f = np.min(valid_fvals)
            max_f = np.max(valid_fvals)
            if max_f > min_f:
                w = 1.0 / (valid_fvals - min_f + 1.0)
            else:
                w = np.ones_like(valid_fvals)
            w /= w.sum()
            
            # Weighted mean
            wmean = np.sum(w[:, np.newaxis] * valid_points, axis=0)
            # Weighted covariance
            centered = valid_points - wmean
            C_init = np.sum(w[:, np.newaxis, np.newaxis] * np.outer(centered, centered), axis=0)
            # Add small regularization
            C_init += 1e-6 * np.eye(dim)
            
            # Ensure positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C_init)
                eigvals = np.maximum(eigvals, 1e-8)
                C_init = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C_init = np.eye(dim)
        else:
            C_init = np.eye(dim)
        
        # Determine initial sigma based on spread of good points
        if np.sum(valid_mask) >= 2:
            good_points = init_points[valid_mask]
            good_fvals = init_fvals[valid_mask]
            # Take the best half
            n_good = max(2, len(good_fvals) // 2)
            idx = np.argsort(good_fvals)[:n_good]
            good_subset = good_points[idx]
            spread = np.std(good_subset, axis=0)
            sigma0 = max(0.5, np.mean(spread) * 1.5)
        else:
            sigma0 = 2.0
        
        # Main CMA-ES loop with restarts
        while calls < total_budget:
            if time.time() > deadline - 5.0:
                break
                
            # Initialize CMA-ES state
            mean = best_x.copy()
            C = C_init.copy()
            sigma = sigma0
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            try:
                chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except:
                C = np.eye(dim)
                chol = np.eye(dim)
                sigma = 1.0
            
            # Track progress for restart decision
            best_f_in_run = best_f
            calls_in_run = 0
            prev_best_f = best_f
            no_improvement_calls = 0
            max_no_improvement = 80  # Restart if no improvement for 80 calls
            
            # Run CMA-ES until budget exhausted or restart triggered
            while calls < total_budget and calls_in_run < 250:
                if time.time() > deadline - 2.0:
                    break
                    
                # Sample population
                z = np.random.randn(lambda_pop, dim)
                pop = mean + sigma * (z @ chol.T)
                
                # Evaluate population
                fvals = []
                for x in pop:
                    if calls >= total_budget:
                        break
                    if time.time() > deadline - 1.0:
                        fvals.append(float('inf'))
                        continue
                    try:
                        f = func(x)
                        calls += 1
                        calls_in_run += 1
                        fvals.append(f)
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        if f < best_f_in_run:
                            best_f_in_run = f
                            no_improvement_calls = 0
                        else:
                            no_improvement_calls += 1
                    except:
                        fvals.append(float('inf'))
                
                if len(fvals) < lambda_pop:
                    fvals.extend([float('inf')] * (lambda_pop - len(fvals)))
                
                # Sort population
                order = np.argsort(fvals)
                sorted_pop = pop[order]
                sorted_fvals = fvals[order]
                
                # Check if all are inf
                if not np.isfinite(sorted_fvals[0]):
                    break
                
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # Check if we should stop
                if np.linalg.norm(chol) < 1e-15:
                    break
                
                # Update evolution paths
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(chol) * (chol.T @ (new_mean - mean) / sigma)
                
                hsig = 1 if np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**((calls + 1) / (cs * (dim + 1)))) < (1.4 + 2 / (dim + 1)) * math.sqrt((dim + 1) / (dim - 1)) else 0
                
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma
                
                # Update C
                C = (1 + c1 * hsig * (cc - cc + 1) + cmu - c1) * C \
                    + c1 * (np.outer(pc, pc) - C) \
                    + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(sorted_pop[:mu] - mean, sorted_pop[:mu] - mean), axis=0) / (sigma**2)
                
                # Ensure C is positive definite
                try:
                    chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except:
                    # If Cholesky fails, reset C to identity and sigma to 1
                    C = np.eye(dim)
                    chol = np.eye(dim)
                    sigma = 1.0
                    
                # Update sigma with tuned rates
                sigma = sigma * math.exp(cs / ds * (np.linalg.norm(ps) / math.sqrt(mueff) - 1))
                
                # Update mean
                mean = new_mean
                
                # Clip mean to box to keep search within bounds
                mean = np.clip(mean, -5, 5)
                
                # Check for restart condition
                if no_improvement_calls > max_no_improvement and calls_in_run > 50:
                    break
            
            # If we still have budget, restart from a new random point near the best found so far
            if calls < total_budget:
                # Perturb the best point slightly to escape local minima
                perturbation_scale = 0.3 + 0.3 * (1 - calls / total_budget)
                best_x_perturbed = best_x + perturbation_scale * np.random.randn(dim)
                best_x_perturbed = np.clip(best_x_perturbed, -5, 5)
                # Try evaluating the perturbed point
                try:
                    f = func(best_x_perturbed)
                    calls += 1
                    if f < best_f:
                        best_f = f
                        best_x = best_x_perturbed.copy()
                except:
                    pass
                
                # Update C_init based on new best
                if calls < total_budget:
                    # Small local search around best point
                    n_local = min(5, total_budget - calls)
                    for _ in range(n_local):
                        if calls >= total_budget:
                            break
                        if time.time() > deadline - 2.0:
                            break
                        local_x = best_x + 0.1 * np.random.randn(dim)
                        local_x = np.clip(local_x, -5, 5)
                        try:
                            f = func(local_x)
                            calls += 1
                            if f < best_f:
                                best_f = f
                                best_x = local_x.copy()
                        except:
                            pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
