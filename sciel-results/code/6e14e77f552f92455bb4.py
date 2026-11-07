# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget
        
        if budget < 20:
            # Fallback for very small budgets
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f, best_x = f, x
            return best_x, best_f

        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return None
            calls_made += 1
            return func(x)

        # Phase 1: Initial sampling to find a good starting point
        n_init = min(20, budget // 5)
        best_x = None
        best_f = np.inf
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, n)
            f = safe_call(x)
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_call(best_x)
            if best_f is None:
                return best_x, np.inf

        # Phase 2: CMA-ES with stagnation-based restarts
        # CMA-ES is the strongest general-purpose black-box optimizer for this type of problem.
        # We implement a robust version with:
        # - Population size lambda
        # - Rank-mu update for covariance matrix
        # - Path-based step-size adaptation
        # - Stagnation detection: if no improvement for a while, restart with a larger sigma
        
        def run_cmaes(start_x, start_sigma, max_calls):
            """Run CMA-ES starting from start_x with initial step size start_sigma."""
            nonlocal best_x, best_f, calls_made, deadline
            
            # CMA-ES parameters
            lam = 4 + int(3 * np.log(n))  # lambda = 4 + 3*ln(5) ≈ 9
            mu = lam // 2  # mu = 4
            
            # Weights for the (mu, lambda)-selection
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log((lam + 1) / 2) - np.log(i + 1)
            weights /= weights.sum()
            mueff = (weights.sum() ** 2) / np.sum(weights ** 2)
            
            # Adaptation rates
            cs = (mueff + 2) / (n + mueff + 5)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            # Initial state
            m = start_x.copy()
            sigma = start_sigma
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            hsig = np.sqrt((1 - cc) * (2 / (1 + cc)) + 1)
            
            # Expected norm of a standard normal in n dimensions
            e_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(32*n**2))
            
            calls_in_this_run = 0
            no_improve = 0
            stagnation_limit = 60  # restart if no improvement for 60 calls
            
            while calls_made < budget and time.time() < deadline:
                if calls_in_this_run >= max_calls:
                    break
                if no_improve >= stagnation_limit:
                    break
                    
                # Generate offspring
                Y = rng.standard_normal((lam, n))
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-10 * np.eye(n)
                    L = np.linalg.cholesky(C)
                
                X = np.zeros((lam, n))
                for i in range(lam):
                    X[i] = m + sigma * (L @ Y[i])
                
                # Evaluate offspring
                F = np.full(lam, np.inf)
                for i in range(lam):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    F[i] = safe_call(X[i])
                    calls_in_this_run += 1
                    
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Sort
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                
                # Update best
                old_best = best_f
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = X[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                # Update mean
                m_old = m.copy()
                m = np.sum(weights[:, None] * X[:mu], axis=0)
                
                # Update step size
                d = (m - m_old) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * cc) * (L.T @ d)
                sigma *= np.exp(cs / e_norm * (np.linalg.norm(ps) / e_norm - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update covariance matrix
                # pc update
                pc = (1 - cc) * pc + cc * hsig * (L.T @ d)
                delta_hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) / e_norm) < (1.4 + 2 / (n + 1))
                
                C = (1 + c1 * (2 - cc - (2 - cc) * delta_hsig - cmu) + cmu - c1) * C
                C += c1 * (2 - cc - (2 - cc) * delta_hsig) * (np.outer(pc, pc) - (1 - cc) * C)
                C += cmu * np.sum(weights[:, None, None] * np.outer(X[:mu] - m_old, X[:mu] - m_old) / sigma ** 2, axis=0)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(n)
                    try:
                        np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        
                if calls_made >= budget or time.time() > deadline:
                    break

            return best_x, best_f

        # Run CMA-ES with potential restarts
        # Start with a reasonable initial sigma
        sigma_init = 1.0
        run_cmaes(best_x, sigma_init, max_calls=100)
        
        # If we still have budget, do a few more CMA-ES runs with different sigmas
        for sigma_mult in [0.5, 2.0, 0.2, 5.0]:
            if calls_made >= budget or time.time() >= deadline:
                break
            sigma_try = best_f and (sigma_init * sigma_mult) or 1.0
            # Restart from the best point with a different sigma
            # Use a fresh CMA-ES run
            run_cmaes(best_x, sigma_try, max_calls=100)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
