import numpy as np
import time
import random


class Optimizer:
    """CMA-ES with adaptive restarts for Katsuura function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_point(x):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or check_time():
                return
            x = np.clip(x, -5, 5)
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Initial sampling to get a rough idea of the landscape
        # Sample origin, axis points, and random points
        try_point(np.zeros(d))
        
        for sign in [-1, 1]:
            for i in range(d):
                if check_time() or used_calls >= budget:
                    break
                x = np.zeros(d)
                x[i] = sign * 3.0
                try_point(x)

        # Random sampling to find promising regions
        num_coarse = min(80, budget - used_calls)
        for _ in range(num_coarse):
            if used_calls >= budget or check_time():
                break
            x = rng.uniform(-4, 4, d)
            try_point(x)

        # Phase 2: CMA-ES implementation
        if best_x is not None and used_calls < budget and not check_time():
            self._cma_es(func, best_x, best_f, budget, d, used_calls, check_time, try_point, deadline, rng)

        # Phase 3: Fine-tuning with small perturbations
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls
            
            # Try small random steps with decreasing magnitude
            for scale in [0.05, 0.01, 0.001, 0.0001]:
                if used_calls >= budget or check_time():
                    break
                for _ in range(min(30, remaining)):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    y = np.clip(x + scale * z, -5, 5)
                    try_point(y)
                    if best_x is not None and best_f < fx:
                        x = best_x.copy()
                        fx = best_f

        return best_x, best_f

    def _cma_es(self, func, x0, f0, budget, d, used_calls, check_time, try_point, deadline, rng):
        """CMA-ES implementation with adaptive restarts."""
        nonlocal best_x, best_f, used_calls

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(d)))
        sigma = 1.0  # Initial step size
        c1 = (4 + np.log(d)) / ((d + 4)**2 + 4)
        cmu = min(1 - c1, 2 * (2 - d / (d + 2))**2 - 1)
        cs = (np.sqrt(d + 1.3) - 1) / (d + 3 + 2. / d)
        ds = 1 + cs + 2 * max(0, np.sqrt((cs - 1) * (d - 1) / (d + 1)) - 1)
        cc = (4 + d / d) / (4 + d)

        # State variables
        mean = x0.copy()
        C = np.eye(d)  # Covariance matrix
        C_sqrt = np.eye(d)  # Square root of C
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1. / (4 * d) + 1. / (21 * d**2))
        chi_d_sqrt = np.sqrt(d)

        # Eigenvalue decomposition for C
        def update_C_sqrt(C):
            nonlocal C_sqrt
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            except:
                C_sqrt = np.eye(d)

        update_C_sqrt(C)

        # Selection: keep top half
        lambda_ = pop_size
        mu = lambda_ // 2  # Number of points used for recombination

        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1. / np.sum(weights**2)

        # Maximum number of CMA-ES iterations
        max_iter = 100
        stall_count = 0
        stall_threshold = 10

        for iteration in range(max_iter):
            if used_calls >= budget or check_time():
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if used_calls >= budget or check_time():
                    break
                z = rng.standard_normal(d)
                y = C_sqrt @ z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                try_point(x)
                if best_x is not None:
                    offspring.append((best_f, x))

            if len(offspring) < mu:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Check for stall
            if offspring[0][0] >= best_f * (1 - 1e-8) and iteration > 5:
                stall_count += 1
            else:
                stall_count = 0

            if stall_count >= stall_threshold:
                # Restart with smaller sigma
                sigma *= 0.5
                stall_count = 0
                if sigma < 1e-6:
                    break

            # Update mean using weighted recombination
            new_mean = np.zeros(d)
            for i in range(mu):
                new_mean += weights[i] * offspring[i][1]
            mean = new_mean

            # Update evolution path for sigma
            y_mean = (new_mean - mean) / sigma if sigma > 1e-20 else np.zeros(d)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_mean

            # Update evolution path for covariance
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt @ y_mean)

            # Update covariance matrix
            delta_sig = (np.linalg.norm(p_sigma) / chi_d - 1) * min(1, 1 / (cs * ds))
            C = ((1 - c1 - cmu) * C +
                 c1 * (p_c @ p_c + (np.linalg.norm(p_c)**2 / chi_d**2 - 1) * C) +
                 cmu * np.sum(weights[i] * np.outer(offspring[i][1] - mean, offspring[i][1] - mean) for i in range(mu)))

            # Ensure positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(d)

            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            update_C_sqrt(C)

            # Check convergence
            if sigma < 1e-8:
                break

        return best_x, best_f
