"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with coordinate-wise adaptive step sizes and periodic gradient polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))  # lam = 8 for dim=5
        mu = lam // 2  # mu = 4

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        # Initial state
        m = np.zeros(dim)
        sigma = 1.0
        sigma_min = 1e-10
        sigma_max = 10.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Coordinate-wise adaptive step sizes (CMA-ES-lite)
        # Each dimension has its own step size
        coord_sigmas = np.ones(dim)
        # Success rate tracking for each dimension
        # We'll track recent successes/failures per dimension
        coord_history = [0] * dim  # 1 for success, 0 for failure
        history_len = 10

        # Best solution found
        best_x = m.copy()
        best_f = func(best_x)
        evals_used = 1

        # Gradient polish function
        def gradient_polish(x0, max_evals, evals_used):
            """Finite-difference gradient estimation and line search polish."""
            if max_evals <= 0:
                return x0.copy(), evals_used
            
            x = x0.copy()
            h = 1e-4  # Step size for finite differences
            
            for _ in range(max_evals):
                if time.time() - start_time > time_limit:
                    break
                
                # Estimate gradient
                grad = np.zeros(dim)
                for i in range(dim):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    x_plus = np.clip(x_plus, -5, 5)
                    x_minus = np.clip(x_minus, -5, 5)
                    fp = func(x_plus)
                    fm = func(x_minus)
                    evals_used += 2
                    grad[i] = (fp - fm) / (2 * h)
                
                # Check if we're out of budget
                if evals_used >= max_evals:
                    break
                
                # If gradient is very small, we've likely converged
                if np.linalg.norm(grad) < 1e-8:
                    break
                
                # Backtracking line search
                alpha = 0.1
                x_new = x - alpha * grad
                f_new = func(x_new)
                evals_used += 1
                
                # If it doesn't improve, reduce step size
                while f_new >= best_f and alpha > 1e-6:
                    alpha *= 0.5
                    x_new = x - alpha * grad
                    f_new = func(x_new)
                    evals_used += 1
                    if evals_used >= max_evals:
                        break
                
                if f_new < best_f:
                    x = x_new
                    best_x[:] = x
                    best_f = f_new
                else:
                    # If no improvement, break
                    break
            
            return best_x, evals_used

        # Main CMA-ES loop
        while evals_used < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break

            # Periodic gradient polish every 10 generations
            gen_count = evals_used // lam
            if gen_count > 0 and gen_count % 10 == 0:
                # Use 10% of remaining budget for polish, at most 20 evals
                polish_budget = min(20, (budget - evals_used) // 5)
                if polish_budget > 2:
                    _, evals_used = gradient_polish(best_x, polish_budget, evals_used)
                    continue

            # 1. Sample population
            ys = rng.standard_normal((lam, dim))
            
            # Whitening transformation
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            
            xs = m[None, :] + sigma * (ys @ L.T)
            
            # Apply coordinate-wise adaptive step sizes
            # Scale each dimension by its own sigma
            xs = xs * coord_sigmas[None, :]
            xs = np.clip(xs, -5, 5)
            
            # 2. Evaluate
            fs = np.array([func(x) for x in xs])
            evals_used += lam
            
            # Update best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = xs[idx_best].copy()
            
            # 3. Update coordinate-wise step sizes based on success rate
            # For each dimension, check if moving in the direction of the best
            # individual improved the objective
            for d in range(dim):
                # Compare best individual's coordinate to mean
                direction = np.sign(xs[idx_best, d] - m[d])
                if direction == 0:
                    direction = 1
                
                # Try a small move in that direction
                x_test = best_x.copy()
                x_test[d] += direction * coord_sigmas[d] * 0.1
                x_test = np.clip(x_test, -5, 5)
                f_test = func(x_test)
                evals_used += 1
                
                if f_test < best_f:
                    coord_history[d] = 1
                    coord_sigmas[d] = min(coord_sigmas[d] * 1.1, sigma_max)
                else:
                    coord_history[d] = 0
                    coord_sigmas[d] = max(coord_sigmas[d] * 0.9, sigma_min)
            
            # 4. Update CMA-ES state
            # Sort by fitness
            idx_sorted = np.argsort(fs)
            x_sorted = xs[idx_sorted]
            
            # Weighted mean
            new_m = np.dot(weights, x_sorted[:mu])
            
            # Update step size (CSA)
            diff = (new_m - m) / sigma
            v = np.linalg.solve(L.T, diff)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
            
            hsig = 1.0 if np.linalg.norm(ps) < (1.4 + 2.0 / (dim + 1.0)) * chiN else 0.0
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Update covariance
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * diff
            rank1 = np.outer(pc, pc)
            v_mu = (x_sorted[:mu] - m) / sigma
            rankmu = np.dot((weights[:, None] * v_mu).T, v_mu)
            
            C = (1 + c1 * hsig * (2 - hsig) - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            C = (C + C.T) / 2.0
            m = new_m
            
            # Check convergence
            if sigma < 1e-10:
                break

        # Final gradient polish with remaining budget
        remaining = budget - evals_used
        if remaining > 2:
            if time.time() - start_time < time_limit - 5:
                _, evals_used = gradient_polish(best_x, remaining, evals_used)

        return best_x, best_f

# EVOLVE-BLOCK-END
