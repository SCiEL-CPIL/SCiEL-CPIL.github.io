import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        budget = self.budget
        dim = self.dim

        calls = [0]
        best_x = np.zeros(dim)
        best_f = float('inf')
        all_xs = []
        all_fs = []

        def evaluate(x):
            nonlocal best_x, best_f
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            all_xs.append(x.copy())
            all_fs.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration
        # Lunacek bi-Rastrigin has two basins. We need to sample widely.
        # 30 initial samples.
        n_init = 30
        for _ in range(n_init):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: Bayesian Optimization / Surrogate-based search
        # We will use a simple surrogate model. Given the constraints (500 evals, 5D),
        # a full GP might be slow to retrain 470 times.
        # However, dim=5 is small.
        # Let's try a simpler approach: Random Forest or KNN-based surrogate?
        # Or just a simple interpolation.
        # Actually, with 500 evals, a simple "Infill" strategy based on distance to known points
        # and function values can work well.
        
        # Let's implement a simple surrogate-based search:
        # 1. Fit a simple model (e.g., linear regression or KNN) to current data.
        # 2. Predict function values on a set of candidate points.
        # 3. Choose the point that maximizes Expected Improvement (EI) or simply minimizes predicted value + uncertainty.
        
        # To keep it fast and robust without sklearn (though sklearn is often available, let's stick to numpy/scipy if possible, or simple logic):
        # We can use a simple K-Nearest Neighbors predictor with inverse distance weighting.
        # Uncertainty can be estimated as the variance of the K neighbors or distance to K-th neighbor.
        
        # Let's use a simple strategy:
        # Candidate generation:
        # 1. Perturbations of the current best point.
        # 2. Random points.
        # 3. Points interpolated between good points.
        
        # Actually, a more robust approach for 500 evals in 5D is a simple "Adaptive Search":
        # - Maintain a set of "active" points.
        # - Generate offspring from the best points.
        # - If no improvement, increase step size.
        
        # Let's try a population-based approach with a simple surrogate for selection.
        # But the directive says "Structural change... different design...".
        # The current code is (1+1)-ES with restarts.
        # Inspiration 1 (CMA-ES) scored 2.38.
        # Let's implement a lightweight CMA-ES or a similar covariance matrix adaptation.
        # CMA-ES is robust for multimodal if restarted properly.
        
        # Given the budget is small (500), a full CMA-ES with large population is expensive.
        # Population size lambda ~ 4 * dim = 20.
        # 500 / 20 = 25 generations.
        # This is reasonable.
        
        # Let's implement a simple (mu+lambda)-ES with covariance matrix adaptation (CMA-ES lite).
        # Or just a standard (mu+lambda)-ES with step size adaptation.
        
        # Let's go with a simple CMA-ES implementation.
        
        # Parameters
        mu = 10  # Number of parents
        lambda_ = 20  # Number of offspring
        # Initial mean
        m = best_x.copy()
        
        # Initial covariance
        C = np.eye(dim)
        
        # Initial step size
        sigma = 1.0
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff/dim - 1)) + c_sigma
        c_c = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        def cma_es_step():
            nonlocal m, C, sigma, ps, pc, best_x, best_f
            if calls[0] >= budget or time.time() > deadline:
                return False
            
            # Generate offspring
            ys = []
            xs = []
            for _ in range(lambda_):
                z = rng.multivariate_normal(np.zeros(dim), C)
                # Actually, standard normal transformed by C^0.5 is better for stability
                # But for simplicity, let's use Cholesky or just standard normal * sigma if C is identity?
                # No, we need to adapt C.
                # Let's use a simpler update for C to avoid numerical issues.
                # Standard CMA-ES update:
                # y = C^0.5 * z
                # We need sqrtm or cholesky.
                # Let's use a simpler approach: just use standard normal and scale by sigma,
                # and adapt the "direction" via the covariance of the successful steps.
                
                # Actually, let's just use a (mu+lambda)-ES with a simple covariance update.
                z = rng.standard_normal(dim)
                y = sigma * z
                x = m + y
                f = evaluate(x)
                ys.append(y)
                xs.append((x, f))
            
            # Sort by fitness
            xs.sort(key=lambda t: t[1])
            
            # Select parents
            parents = [xs[i][0] for i in range(mu)]
            ys_parents = [ys[i] for i in range(mu)]
            
            # Update mean
            new_m = np.sum([w * p for w, p in zip(weights, parents)], axis=0)
            
            # Update evolution path ps
            # h_sigma = 1 if norm(ps) < (1 + 2/(dim+2)) * sqrt(1 - (1-t_gen/t_end)) * sqrt(mu_eff)
            # Simplified:
            norm_ps = np.linalg.norm(ps)
            threshold = (1 + 2.0 / (dim + 2.0)) * np.sqrt(mu_eff)
            h_sigma = 1.0 if norm_ps < threshold * 0.85 else 0.0  # Heuristic
            
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                np.sum([w * y for w, y in zip(weights, ys_parents)], axis=0) / sigma
            )
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update evolution path pc
            # h_c = 1 if norm(pc) < (1 + 2/(dim+2)) * sqrt(1 - (1-t_gen/t_end)) * sqrt(c_c * (2 - c_c) * mu_eff)
            norm_pc = np.linalg.norm(pc)
            threshold_c = (1 + 2.0 / (dim + 2.0)) * np.sqrt(c_c * (2 - c_c) * mu_eff)
            h_c = 1.0 if norm_pc < threshold_c * 0.85 else 0.0
            
            # Compute delta_h
            delta_h = (h_c - 1) * c_c * (2 - c_c)
            
            # Update pc
            pc = (1 - c_c) * pc + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                (new_m - m) / sigma
            )
            
            # Update covariance C
            rank_one = np.outer(ps, ps)
            rank_mu = np.sum([w * np.outer(y, y) for w, y in zip(weights, ys_parents)], axis=0) / sigma**2
            
            C = (1 + c_1 * delta_h - c_1) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure symmetry and positive definiteness (simple fix)
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-6 * np.eye(dim)
            
            m = new_m
            
            return True

        # Run CMA-ES for the remaining budget
        # We need to check budget inside the loop.
        while calls[0] < budget and time.time() < deadline:
            if not cma_es_step():
                break
        
        return best_x, best_f
