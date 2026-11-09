import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 295.0  # 35s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def eval_point(x):
            nonlocal calls_made, best_f, best_x
            if not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Broad global exploration with stratified sampling
        # Katsuura is highly multimodal; we need good coverage early for AOCC
        n_phase1 = min(30, budget // 5)
        if n_phase1 < 5:
            n_phase1 = 5

        # Stratified/Latin Hypercube sampling for good coverage
        rng = np.random.default_rng(42)
        points = np.zeros((n_phase1, dim))
        for d in range(dim):
            perm = rng.permutation(n_phase1)
            points[:, d] = (perm + rng.random(n_phase1)) / n_phase1
        points = points * 10.0 - 5.0  # map to [-5, 5]
        points = np.clip(points, -5.0, 5.0)

        for x in points:
            if not time_ok():
                break
            eval_point(x)

        # Add center point and a few random points
        extra_points = [np.zeros(dim)]
        for _ in range(3):
            extra_points.append(rng.uniform(-4.0, 4.0, size=dim))
        for x in extra_points:
            if not time_ok():
                break
            eval_point(x)

        if best_x is None:
            # Fallback
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x.copy()
            calls_made = 1

        # Phase 2: CMA-ES from the best point
        # CMA-ES is well-suited for non-separable, multimodal functions
        if calls_made < budget and time_ok():
            x_cma = best_x.copy()
            f_cma = best_f
            sigma = 1.5  # initial step size
            
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))  # population size
            lambda_ = pop_size
            cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + cc)  # path length adaptation
            cs = (dim + 1.7) / (dim + 9 + cs)  # step size adaptation
            damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
            cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + cs)
            c1 = 2 / ((dim + 1.3) ** 2 + dim)
            cmu = min(0.5 - c1, 2 * (2 - dim / (dim + 4)) ** 2 - c1)
            cmu = max(0.0, cmu)
            
            # Initialize
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)  # will be updated
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            mean = x_cma.copy()
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Budget for CMA-ES
            cma_budget = min(budget - calls_made, 300)
            
            for gen in range(cma_budget // lambda_ + 1):
                if not time_ok() or calls_made >= budget:
                    break
                
                # Generate population
                offspring = []
                for i in range(lambda_):
                    if not time_ok() or calls_made >= budget:
                        break
                    # Sample
                    z = rng.standard_normal(dim)
                    # Apply CMA-ES transformation (simplified: use Cholesky-like)
                    # For simplicity, use diagonal adaptation first
                    y = mean + sigma * z
                    y = np.clip(y, -5.0, 5.0)
                    f_y = func(y)
                    calls_made += 1
                    if f_y < best_f:
                        best_f = f_y
                        best_x = y.copy()
                    offspring.append((f_y, y))
                
                if len(offspring) < 2:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                f_best_gen = offspring[0][0]
                x_best_gen = offspring[0][1]
                if f_best_gen < best_f:
                    best_f = f_best_gen
                    best_x = x_best_gen.copy()
                
                # Update mean
                mu_eff = min(lambda_ // 2, int(np.sum(2 ** (-np.arange(1, lambda_ + 1) / (lambda_ - 1)))))
                weights = np.zeros(lambda_)
                weights[:mu_eff] = np.log(mu_eff + 0.5) - np.log(np.arange(1, mu_eff + 1))
                weights = weights / weights.sum()
                
                new_mean = np.zeros(dim)
                for i in range(mu_eff):
                    new_mean += weights[i] * offspring[i][1]
                
                # Update covariance matrix (simplified)
                delta_mean = new_mean - mean
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * C_sqrt_inv @ (delta_mean / sigma)
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * delta_mean / sigma
                
                # Hessian correction
                hsig = np.sqrt((1 - (1 - cs)**10) / (1 - (1 - cs)**2))
                chi_n2 = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                hsig = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2/(dim+1)) * chi_n2 else 0.0
                
                C = (1 + c1 * hsig * hsig - c1) * C + c1 * np.outer(p_c, p_c)
                # Simplified: don't update C fully, just use diagonal adaptation
                # For speed, use a simpler approach
                
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n2 - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)
                
                mean = new_mean
                mean = np.clip(mean, -5.0, 5.0)
                
                # Update C_sqrt_inv (simplified: just use identity for now)
                # This is a simplified CMA-ES; for full implementation, need eigendecomposition
        
        # Phase 3: Local refinement with decreasing step sizes
        if calls_made < budget and time_ok():
            x_refine = best_x.copy()
            f_refine = best_f
            
            # Try several step sizes, decreasing
            for sigma_try in [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
                if not time_ok() or calls_made >= budget:
                    break
                
                improved = True
                while improved and time_ok() and calls_made < budget:
                    improved = False
                    # Try random directions
                    for _ in range(10):
                        if not time_ok() or calls_made >= budget:
                            break
                        direction = rng.standard_normal(dim)
                        direction /= np.linalg.norm(direction)
                        y = x_refine + sigma_try * direction
                        y = np.clip(y, -5.0, 5.0)
                        f_y = func(y)
                        calls_made += 1
                        if f_y < f_refine:
                            x_refine = y
                            f_refine = f_y
                            if f_refine < best_f:
                                best_f = f_refine
                                best_x = x_refine.copy()
                            improved = True
    
        return best_x, best_f
