"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        # Initial population size, will shrink as we converge
        initial_pop_size = 4 * n + 2  # Standard CMA-ES: 4n+2
        min_pop_size = 2  # Minimum population size for stability
        
        # Initialize mean (center) and covariance
        mean = np.zeros(n)
        C = np.eye(n)
        sigma = 1.5  # Initial step size
        
        # CMA-ES adaptation parameters
        mu = initial_pop_size // 2  # Number of best individuals used for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # Path lengths
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Damping factor for covariance update
        d_amps = 1.0
        
        # Track convergence to adapt population size
        improvement_history = []
        last_improvement = 0
        gen = 0
        
        # Evaluate initial point
        fx = wrapped_func(mean)
        
        while calls_made[0] < budget:
            if time.time() > deadline:
                break
            
            # Determine current population size based on convergence
            # Shrink population as we get closer to convergence
            # Use a smooth function that reduces pop size when improvements slow down
            if len(improvement_history) > 10:
                recent_improvements = improvement_history[-10:]
                avg_recent = np.mean(recent_improvements)
                # If average improvement is small, reduce population
                if avg_recent < 1e-6:
                    pop_size = max(min_pop_size, initial_pop_size // 4)
                elif avg_recent < 1e-4:
                    pop_size = max(min_pop_size, initial_pop_size // 2)
                else:
                    pop_size = initial_pop_size
            else:
                pop_size = initial_pop_size
            
            # Generate population
            population = []
            for i in range(pop_size):
                z = rng.standard_normal(n)
                y = np.clip(mean + sigma * z, -5, 5)
                fy = wrapped_func(y)
                population.append((fy, y))
            
            # Sort by fitness
            population.sort(key=lambda x: x[0])
            
            # Update mean using weighted combination of best mu individuals
            new_mean = np.zeros(n)
            for i in range(min(mu, len(population))):
                new_mean += weights[i] * population[i][1]
            mean = new_mean
            
            # Update step size (sigma) using path evolution
            y_mean = (mean - (mean - sigma * 0)) / sigma if sigma > 0 else np.zeros(n)
            # Simplified: use the actual step
            y_step = (new_mean - mean) / sigma if sigma > 1e-10 else np.zeros(n)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_step
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) / chi_n < 1.4 + 2 / (n + 1)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_step
            
            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(min(mu, len(population))):
                y_i = (population[i][1] - mean) / sigma if sigma > 1e-10 else np.zeros(n)
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * (d_amps - 1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(1e-10, min(sigma, 10.0))
            
            gen += 1
            
            # Track improvement
            current_best = population[0][0]
            if last_improvement > 0:
                improvement = last_improvement - current_best
                improvement_history.append(max(0, improvement))
            last_improvement = current_best
            
            # Stall detection: if no significant improvement for many generations, restart
            if len(improvement_history) > 30:
                recent = improvement_history[-30:]
                if np.sum(recent) < 1e-8:
                    # Restart with new random center
                    mean = rng.uniform(-3, 3, n)
                    C = np.eye(n)
                    sigma = 1.5
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    improvement_history = []
                    last_improvement = wrapped_func(mean)
                    gen = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
