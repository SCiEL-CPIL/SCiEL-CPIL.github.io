"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Optimizer:
    1. Global Phase: Deterministic Sobol quasi-random sequence for uniform space coverage.
    2. Local Phase: Lightweight CMA-ES (adapted step size, fixed covariance initially or simplified adaptation) 
       to refine the best point found.
    
    This addresses the "21 peaks" problem by ensuring broad, variance-free exploration 
    before committing budget to local exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, n, dim, seed=12345):
        """
        Generate a Sobol quasi-random sequence.
        Uses a simple implementation compatible with standard libraries.
        If scipy is available, we might use it, but to be safe and fast, 
        we implement a basic Sobol or use numpy's random if scipy is too heavy.
        However, the prompt says SciPy is available. Let's use scipy.stats.qmc.Sobol.
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
            # Generate n points in [0, 1]^dim
            points = sampler.random(n)
            # Map to [-5, 5]^dim
            return points * 10 - 5
        except ImportError:
            # Fallback to Halton or just random if scipy fails (unlikely)
            rng = np.random.default_rng(seed)
            return rng.uniform(-5, 5, (n, dim))

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s safety margin
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0
        dim = self.dim
        budget = self.budget

        # Phase 1: Global Exploration with Sobol
        # Allocate ~40% of budget to global search
        # 21 peaks in 5D. 500 points total. 200 points for global is ~1 point per 125 volume units?
        # Volume is 10^5 = 100,000. 200 points is 1 per 500. Might be sparse for 21 peaks?
        # Let's try 150 points for global, 350 for local.
        n_global = 150
        if budget > 200:
            n_global = min(200, budget // 3)
        else:
            n_global = budget // 2

        # Generate Sobol points
        sobol_points = self._sobol_sequence(n_global, dim, seed=42)
        
        # Evaluate global points
        for i in range(n_global):
            if time.time() > deadline or calls_made >= budget:
                break
            
            x = sobol_points[i]
            # Clip to box
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Local Refinement with CMA-ES
        # Use the best point found in Phase 1 as the center
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls_made += 1

        x = best_x.copy()
        fx = best_fx
        sigma = 1.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        ps = np.zeros(dim)  # Evolution path for sigma
        pc = np.zeros(dim)  # Evolution path for C
        cs = 1 + 2 * (dim - 2) / (dim + 1) ** 2
        ds = 1 + cs + 2 * max(0, (dim / (dim + 1) - 1) ** 0.5 - 1)
        cmu = 2 / ((dim + 1.3) ** 2 + mu) if False else 2 / ((dim + 1.3) ** 2 + 1) # Simplified cmu
        # Let's use a standard lightweight CMA-ES adaptation
        # mu = dim // 2 + 1
        mu = dim // 2 + 1
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        cmu = min(2 / ((dim + 1.3) ** 2 + mu_eff), 0.5 / dim)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (dim / (dim + 1) - 1) ** 0.5 - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.7) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for k in range(mu):
                y = rng.multivariate_normal(np.zeros(dim), C)
                z = y / (np.linalg.norm(y) + 1e-16) if np.linalg.norm(y) > 1e-16 else y
                # Actually, standard CMA-ES samples y ~ N(0, C)
                # Let's stick to simpler: y = rng.standard_normal(dim)
                # x_new = x + sigma * C @ y
                # But C adaptation is heavy. Let's use a simpler "1+1" style with CMA-ES sigma adaptation?
                # No, the prompt asks for a structural change. The current is 1+1.
                # Let's do a proper small-population CMA-ES.
                
                y_k = rng.standard_normal(dim)
                x_k = x + sigma * (C @ y_k)
                x_k = np.clip(x_k, -5, 5)
                f_k = func(x_k)
                calls_made += 1
                offspring.append((f_k, x_k))
                
                if calls_made >= budget:
                    break

            if len(offspring) == 0:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda item: item[0])
            
            # Update evolution paths
            y_w = np.zeros(dim)
            for i in range(mu):
                f_i, x_i = offspring[i]
                y_i = (x_i - x) / (sigma + 1e-16)
                y_w += weights[i] * y_i

            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // mu)))) < (1 + c_sigma) * chi_n * 1.14
            
            # Update pc
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (ps / chi_n)
            else:
                pc = (1 - c_c) * pc

            # Update C
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - delta_hsig) * C
            C += c_1 * (np.outer(pc, pc) - C)
            C += c_mu * sum(weights[i] * np.outer((x_i - x) / sigma, (x_i - x) / sigma) for i, (f_i, x_i) in enumerate(offspring[:mu]))
            
            # Ensure C is symmetric and positive definite (simple check)
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-16):
                C = C + 1e-6 * np.eye(dim)

            # Update x and sigma
            x = np.zeros(dim)
            for i in range(mu):
                x += weights[i] * offspring[i][1]
            x = np.clip(x, -5, 5)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update best
            f_new = func(x) if calls_made < budget else offspring[0][0]
            if calls_made < budget:
                calls_made += 1
            if f_new < best_fx:
                best_fx = f_new
                best_x = x.copy()
            elif offspring[0][0] < best_fx:
                best_fx = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Check for stagnation
            if sigma < 1e-6 or (best_fx - offspring[0][0]) < 1e-10:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
