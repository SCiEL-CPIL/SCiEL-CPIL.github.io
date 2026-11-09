"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-free optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Wrap func to count evaluations
        eval_count = [0]
        def wrapped_func(x):
            eval_count[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        # Population size: 4 + floor(3 * log(dim)) is a common choice
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        # Initial sigma (step size)
        sigma = 2.0  # Start with a reasonable step size for [-5,5]^5
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution path
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Learning rates
        mu = pop_size // 2  # Number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4. + mueff / dim) / (dim + 4. + 2. * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Damping for sigma update
        damp = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Initial chi_n for expected norm of standard normal
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        # Generate initial population
        def generate_population():
            """Generate a population of points."""
            points = []
            for i in range(pop_size):
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Apply Cholesky decomposition of C
                L = np.linalg.cholesky(C)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5.0, 5.0)
                points.append(x)
            return points

        # Main CMA-ES loop
        while eval_count[0] < budget:
            # Generate population
            population = generate_population()
            
            # Evaluate all points in population
            fvals = []
            for x in population:
                if eval_count[0] >= budget:
                    break
                fx = wrapped_func(x)
                fvals.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if len(fvals) < pop_size:
                break
            
            # Sort by fitness
            indices = np.argsort(fvals)
            sorted_pop = [population[i] for i in indices]
            sorted_fvals = [fvals[i] for i in indices]
            
            # Select top mu parents
            parents = sorted_pop[:mu]
            
            # Weighted recombination
            new_mean = np.zeros(dim)
            for i, w in enumerate(weights):
                new_mean += w * parents[i]
            
            # Update evolution paths
            diff = (new_mean - mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (eval_count[0] // pop_size + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            if hsig:
                pc = (1 - cc) * pc + cc / sigma * (new_mean - mean)
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + delta_hsig * C)
            for i in range(mu):
                zi = (parents[i] - mean) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(cs / damp * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Adaptive sigma: if we're not making progress, reduce sigma
            # Check if we've been stuck
            if eval_count[0] > 100 and eval_count[0] % 50 == 0:
                # Simple stagnation check: if best hasn't improved in last 50 evals, reduce sigma
                sigma *= 0.8

            # If sigma becomes too small, restart with larger sigma
            if sigma < 1e-6:
                sigma = 1.0
                C = np.eye(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                mean = best_x.copy() if best_x is not None else np.zeros(dim)

        # Final polish with local search around best point
        if best_x is not None and eval_count[0] < budget:
            # Local random search around best point
            for _ in range(min(20, budget - eval_count[0])):
                if eval_count[0] >= budget:
                    break
                perturbation = rng.normal(0, 0.05, dim)
                x_new = np.clip(best_x + perturbation, -5.0, 5.0)
                fx = wrapped_func(x_new)
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
