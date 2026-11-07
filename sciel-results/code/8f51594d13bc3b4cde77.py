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

        best_x = None
        best_f = np.inf

        # Phase 1: Sobol Sampling
        n_init = min(50, budget // 4)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n, scramble=True)
            m = 1
            while m < n_init:
                m *= 2
            sobol_points = sampler.random(m)
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            sobol_points = rng.uniform(-5, 5, (n_init, n))

        for i in range(n_init):
            x = sobol_points[i % len(sobol_points)]
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

        # Phase 2: CMA-ES
        # CMA-ES is the state-of-the-art for ill-conditioned black-box optimization.
        # We implement it directly to have full control over budget and time.
        
        def run_cmaes(start_x, start_sigma, max_calls, restart_after_stagnation=40):
            nonlocal best_x, best_f, calls_made, deadline
            
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log((lam + 1) / 2) - np.log(i + 1)
            weights /= weights.sum()
            mueff = (weights.sum() ** 2) / np.sum(weights ** 2)
            
            cs = (mueff + 2) / (n + mueff + 5)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            m = start_x.copy()
            sigma = start_sigma
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            hsig = np.sqrt((1 - cc) * (2 / (1 + cc)) + 1)
            e_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(32*n**2))
            
            calls_in_this_run = 0
            no_improve = 0
            
            while calls_made < budget and time.time() < deadline:
                if calls_in_this_run >= max_calls:
                    break
                if no_improve >= restart_after_stagnation:
                    break
                    
                Y = rng.standard_normal((lam, n))
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-10 * np.eye(n)
                    L = np.linalg.cholesky(C)
                
                X = np.zeros((lam, n))
                for i in range(lam):
                    X[i] = m + sigma * (L @ Y[i])
                
                F = np.full(lam, np.inf)
                for i in range(lam):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    F[i] = safe_call(X[i])
                    calls_in_this_run += 1
                        
                if calls_made >= budget or time.time() > deadline:
                    break
                    
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = X[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                m_old = m.copy()
                m = np.sum(weights[:, None] * X[:mu], axis=0)
                
                d = (m - m_old) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * cc) * (L.T @ d)
                sigma *= np.exp(cs / e_norm * (np.linalg.norm(ps) / e_norm - 1))
                sigma = np.clip(sigma, 1e-10, 5.0)
                
                pc = (1 - cc) * pc + cc * hsig * (L.T @ d)
                delta_hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) / e_norm) < (1.4 + 2 / (n + 1))
                
                C = (1 + c1 * (2 - cc - (2 - cc) * delta_hsig - cmu) + cmu - c1) * C
                C += c1 * (2 - cc - (2 - cc) * delta_hsig) * (np.outer(pc, pc) - (1 - cc) * C)
                C += cmu * np.sum(weights[:, None, None] * np.outer(X[:mu] - m_old, X[:mu] - m_old) / sigma ** 2, axis=0)
                
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

        # Run CMA-ES with a moderate initial sigma
        # The Sobol phase gives us a good starting point, but CMA-ES needs to explore.
        # We use a slightly larger sigma than in the previous version to ensure escape from local minima.
        run_cmaes(best_x, 0.5, max_calls=200)
        
        # If we have budget left, run a second CMA-ES with a smaller sigma for polishing
        if calls_made < budget and time.time() < deadline:
            run_cmaes(best_x, 0.1, max_calls=100)
            
        # If we still have budget, try a few random restarts with Nelder-Mead
        if calls_made < budget and time.time() < deadline:
            from scipy.optimize import minimize
            
            def nm_func(x):
                if calls_made >= budget or time.time() > deadline:
                    return 1e300
                return safe_call(x)
                
            for _ in range(5):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Start from a random point near the best
                x0 = best_x + rng.normal(0, 0.5, n)
                x0 = np.clip(x0, -5, 5)
                
                try:
                    res = minimize(
                        nm_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'xatol': 1e-6,
                            'fatol': 1e-8
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
