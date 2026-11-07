"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for Rosenbrock-style f8 with adaptive boundary handling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s limit
        
        dim = self.dim
        budget = self.budget
        call_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        # Wrapped function to track calls and best
        def wrapped(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time exceeded")
            
            call_count += 1
            # Ensure x is a numpy array and clip to bounds
            x_arr = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_arr)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Lambda: population size
        lam = 4 + int(3 * np.log(dim))
        # Mu: number of best individuals for recombination
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        cc = (4 + np.log(dim)) / (dim + 4 + 2 * np.log(dim))
        cs = (dim + 1.7 - mu_eff) / (dim + 2.7 + mu_eff)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, large enough to explore [-5, 5]
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute chi_lambda (expected norm of standard normal)
        chi_lambda = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
        
        # Main CMA-ES Loop
        # We will run until budget is exhausted or convergence is detected
        # To ensure early progress, we don't stop too early unless sigma is tiny
        
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample offspring
            offspring = []
            for _ in range(lam):
                # Check budget before sampling/evaluating
                if call_count + lam > budget:
                    # If we can't fit a full generation, just break
                    # Or sample fewer? CMA-ES needs full generation for stats.
                    # We'll just break if we can't do a full generation
                    # to maintain statistical integrity, or we can do a partial
                    # step but that's complex. Given 500 budget and lam~10,
                    # we will have ~50 generations.
                    break
                
                # Generate normal random vector
                z = rng.standard_normal(dim)
                # Transform to x
                x_new = x_mean + sigma * C_sqrt @ z
                # Note: We do NOT clip here for the CMA-ES update logic?
                # Actually, for bounded problems, it is better to clip the *evaluation*
                # but keep the *internal* distribution unbounded to avoid bias?
                # However, if the mean drifts out, we waste calls.
                # Standard approach: Clip the point before evaluation.
                # The CMA-ES update uses the *unclipped* point for the mean update?
                # If we clip x_new for evaluation, but use unclipped for mean update,
                # the mean will drift towards the boundary if the optimum is there.
                # This is generally acceptable.
                
                offspring.append((x_new, None)) # Placeholder for fitness
            
            # If we broke out of sampling loop due to budget, stop CMA-ES
            if len(offspring) < lam:
                break

            # 2. Evaluate offspring
            for i, (x_new, _) in enumerate(offspring):
                # Clip for evaluation
                x_eval = np.clip(x_new, -5.0, 5.0)
                try:
                    f_val = wrapped(x_eval)
                except (RuntimeError, Exception):
                    break
                offspring[i] = (x_new, f_val)
            
            # If evaluation failed midway, we might have fewer valid offspring.
            # For simplicity and robustness, if any evaluation fails (e.g. time out), we stop.
            if len(offspring) < lam or any(f is None for _, f in offspring):
                break

            # 3. Sort offspring by fitness
            offspring.sort(key=lambda x: x[1])
            
            # 4. Update Mean
            # Weighted recombination of the best mu individuals
            delta = np.zeros(dim)
            for i in range(mu):
                delta += weights[i] * offspring[i][0]
            delta /= np.sum(weights[:mu]) # weights are already normalized for mu?
            # Actually, weights sum to 1 over the first mu indices.
            x_mean_old = x_mean.copy()
            x_mean += cc * (delta - x_mean)

            # 5. Update Covariance Matrix
            # Calculate ps (sigma evolution path)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv @ (delta / sigma)
            
            # Calculate pc (covariance evolution path)
            # Note: pc update uses the *unclipped* delta
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * C_inv @ (x_mean - x_mean_old) / sigma
            
            # Rank-mu update
            v = np.zeros((dim, mu))
            for i in range(mu):
                v[:, i] = (offspring[i][0] - x_mean_old) / sigma
            
            C += c1 * (np.outer(pc, pc) - (1 - 2 * cs) * C) + cmu * (v @ (weights * v.T) - C)
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2
            # Eigenvalue check for stability
            # Simple check: if any eigenvalue is too small, add identity
            # For dim=5, this is fast
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    C += 1e-12 * np.eye(dim)
                    eigvals, eigvecs = np.linalg.eigh(C)
            except:
                pass

            # Update C_inv and C_sqrt
            # C_sqrt = eigvecs * sqrt(eigvals)
            C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
            C_inv = eigvecs @ np.diag(1.0 / np.maximum(eigvals, 1e-12)) @ eigvecs.T

            # 6. Update Sigma
            sigma *= np.exp(cs / chi_lambda * (np.linalg.norm(ps) / chi_lambda - 1))
            
            # Convergence check
            if sigma < 1e-12:
                break
            
            # Optional: Stop if progress is negligible
            # But AOCC rewards early progress, so we keep going as long as we have budget.

        # Phase 2: Local Polish
        # If we have remaining budget and time, use a simple (1+1) ES or L-BFGS-B
        # Since CMA-ES is already very precise, a simple (1+1) ES with small sigma is good.
        if call_count < budget and time.time() < deadline:
            x_es = best_x.copy()
            fx_es = best_f
            sigma_es = 0.05
            max_es_calls = min(50, budget - call_count)
            
            for _ in range(max_es_calls):
                if call_count >= budget or time.time() > deadline:
                    break
                y = np.clip(x_es + sigma_es * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except (RuntimeError, Exception):
                    break
                if fy < fx_es:
                    x_es, fx_es = y, fy
                    sigma_es *= 1.1
                else:
                    sigma_es *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
