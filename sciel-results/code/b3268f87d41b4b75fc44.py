"""CMA-ES based optimizer for Katsuura f23 with hidden shift/rotation/offset.
Uses a robust global search strategy with adaptive step sizes and multi-restart."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        start_time = time.time()
        DEADLINE = 300.0  # safety margin before 330s kill

        def evaluate(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            return float(func(x))

        # Data storage
        X = []
        Y = []

        def add_point(x, f):
            X.append(np.asarray(x, dtype=float).copy())
            Y.append(float(f))

        best_x = None
        best_f = float('inf')

        # --- Phase 1: Initial diverse sampling ---
        # Use a set of well-distributed initial points
        rng = np.random.RandomState(42)
        
        # Center and axis points
        candidates = [np.zeros(dim)]
        for v in [2.0, -2.0, 3.0, -3.0, 1.0, -1.0, 4.0, -4.0]:
            candidates.append(np.full(dim, v))
        for d in range(dim):
            for v in [2.0, -2.0, 4.0, -4.0]:
                x = np.zeros(dim)
                x[d] = v
                candidates.append(x)
        
        # Random points
        for _ in range(20):
            candidates.append(rng.uniform(-4.5, 4.5, size=dim))

        for x in candidates:
            if calls >= budget or (time.time() - start_time) > DEADLINE:
                break
            x = np.clip(x, -5.0, 5.0)
            fx = evaluate(x)
            add_point(x, fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: CMA-ES-like adaptive search ---
        # Implement a simplified CMA-ES with adaptive step sizes
        mean = best_x.copy()
        sigma = 1.5
        lambda_ = 2 * dim + 1  # population size
        
        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        
        no_improve_count = 0
        max_no_improve = 30
        step_multiplier = 1.0

        while calls < budget and (time.time() - start_time) < DEADLINE:
            # Check time
            if (time.time() - start_time) > DEADLINE - 5:
                break
            
            # Generate population
            pop = []
            pop_f = []
            
            for i in range(lambda_):
                if calls >= budget or (time.time() - start_time) > DEADLINE:
                    break
                
                # Sample from multivariate normal using current C
                z = rng.standard_normal(dim)
                # Transform using eigen decomposition
                x_sample = mean + sigma * eigvecs @ (eigvals ** 0.5 * z)
                x_sample = np.clip(x_sample, -5.0, 5.0)
                
                fx = evaluate(x_sample)
                add_point(x_sample, fx)
                pop.append(x_sample.copy())
                pop_f.append(fx)
            
            if len(pop) == 0:
                break
            
            # Sort by fitness
            sorted_idx = np.argsort(pop_f)
            pop_sorted = [pop[i] for i in sorted_idx]
            pop_f_sorted = [pop_f[i] for i in sorted_idx]
            
            # Update best
            if pop_f_sorted[0] < best_f:
                best_f = pop_f_sorted[0]
                best_x = pop_sorted[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update mean (weighted average of top half)
            mu = lambda_ // 2
            weights = np.zeros(mu)
            w_sum = 0
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
                w_sum += weights[i]
            weights /= w_sum
            
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * pop_sorted[i]
            mean = new_mean
            
            # Update covariance
            diff = np.zeros((mu, dim))
            for i in range(mu):
                diff[i] = pop_sorted[i] - mean
            
            C_new = np.eye(dim) * (1 - 2 / (dim + 1) ** 1.5) * C
            for i in range(mu):
                C_new += weights[i] * np.outer(diff[i], diff[i])
            C = C_new
            
            # Update sigma (simple adaptation)
            if no_improve_count > 10:
                sigma *= 0.8
            elif no_improve_count > 5:
                sigma *= 0.9
            else:
                sigma = min(sigma * 1.05, 5.0)
            
            # Re-decompose C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            
            # Restart if stuck
            if no_improve_count >= max_no_improve:
                # Try a new random start
                new_start = np.clip(rng.normal(0, 2.0, dim), -5.0, 5.0)
                fx = evaluate(new_start)
                add_point(new_start, fx)
                if fx < best_f:
                    best_f = fx
                    best_x = new_start.copy()
                mean = best_x.copy()
                sigma = 1.5
                C = np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                no_improve_count = 0
                max_no_improve = max(15, max_no_improve // 2)

        return best_x, best_f
# EVOLVE-BLOCK-END
