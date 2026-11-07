"""CMA-ES with Restart and Nelder-Mead Polish for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer using CMA-ES with restarts and Nelder-Mead polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        call_count = [0]
        best_x = [np.zeros(dim)]
        best_f = [np.inf]

        def safe_call(x):
            if call_count[0] >= budget:
                return best_f[0]
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                call_count[0] += 1
                if not np.isfinite(f):
                    f = 1e10
                if f < best_f[0]:
                    best_f[0] = f
                    best_x[0] = x.copy()
                return f
            except Exception:
                return 1e10

        def nelder_mead(x0, max_iters=50):
            """Nelder-Mead optimizer for polishing."""
            n = dim
            # Initialize simplex
            simplex = [x0.copy() for _ in range(n + 1)]
            fvals = [safe_call(simplex[0])]
            for i in range(1, n + 1):
                simplex[i] = simplex[0].copy()
                simplex[i][i - 1] += 0.1 * (1.0 if simplex[i][i - 1] < 0 else -1.0)
                fvals.append(safe_call(simplex[i]))
                
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iters):
                if time.time() > deadline or call_count[0] >= budget:
                    break
                if np.std(fvals) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = safe_call(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = safe_call(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    f_c = safe_call(x_c)
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])
                        if call_count[0] >= budget:
                            break

                # Re-sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

            return simplex[0], fvals[0]

        def run_cma_es(mean, sigma, max_gens):
            """Run CMA-ES with stagnation detection."""
            n = dim
            lam = 12
            mu = lam // 2
            
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            C = np.eye(n)
            Csqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            
            z = np.random.randn(lam, n)
            pop = mean + sigma * (z @ Csqrt.T)
            fvals = np.array([safe_call(x) for x in pop])
            
            best_gen_f = fvals[0]
            stagnation_count = 0
            stagnation_threshold = 5

            for gen in range(max_gens):
                if time.time() > deadline or call_count[0] >= budget:
                    break
                
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                
                if fvals[0] < best_gen_f - 1e-10:
                    best_gen_f = fvals[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                if stagnation_count > stagnation_threshold:
                    # Restart logic: break and let caller restart
                    break

                if np.std(fvals[:mu]) < 1e-12 * (1 + np.abs(fvals[0])):
                    break

                w = weights
                new_mean = np.dot(w, pop[:mu])
                
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (Csqrt @ (new_mean - mean))
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**((gen + 1) * cs)) < (1.4 + 2 / (n + 1)) * chi_n)
                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma
                
                delta_sigma = np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = sigma * delta_sigma
                
                C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C)
                C += cmu * np.dot(w[:mu], (pop[:mu] - mean) / sigma).T @ (np.dot(w[:mu], (pop[:mu] - mean) / sigma))
                
                C = (C + C.T) / 2
                try:
                    Csqrt, _ = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    Csqrt = np.eye(n)
                    sigma *= 0.5
                    
                mean = new_mean
                
                z = np.random.randn(lam, n)
                pop = mean + sigma * (z @ Csqrt.T)
                fvals = np.array([safe_call(x) for x in pop])
                
                if sigma < 1e-15:
                    break

            return mean, sigma, stagnation_count

        # Main Loop
        # Start with a few random points to find a good initial mean
        rng = np.random.default_rng(42)
        
        # Initial screening
        n_screen = 5
        screen_points = []
        for _ in range(n_screen):
            x = rng.uniform(-4.0, 4.0, dim)
            f = safe_call(x)
            screen_points.append((f, x))
        
        if screen_points:
            screen_points.sort(key=lambda t: t[0])
            start_mean = screen_points[0][1]
        else:
            start_mean = np.zeros(dim)
            
        start_sigma = 2.0
        
        # Run CMA-ES with restarts
        max_restarts = 3
        for r in range(max_restarts):
            if call_count[0] >= budget - 50: # Leave budget for NM
                break
                
            # Estimate generations
            lam = 12
            remaining_budget = budget - call_count[0]
            max_gens = max(5, remaining_budget // lam)
            
            mean, sigma, stag = run_cma_es(start_mean, start_sigma, max_gens)
            
            # Update start point for next restart if needed
            # If stagnated, try a new random point or smaller sigma
            if stag > 0:
                # Try a new random point if we have budget
                if call_count[0] < budget - 50:
                    new_x = rng.uniform(-4.0, 4.0, dim)
                    new_f = safe_call(new_x)
                    if new_f < best_f[0]:
                        start_mean = new_x
                    else:
                        # Shrink sigma and restart from best
                        start_sigma *= 0.5
                        start_mean = best_x[0].copy()
            else:
                # Converged or finished
                break

        # Polish with Nelder-Mead
        if call_count[0] < budget - 10:
            nelder_mead(best_x[0], max_iters=20)

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
