"""CMA-ES with adaptive restarts for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    CMA-ES with adaptive restarts and a global exploration phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin for 330s limit
        
        dim = self.dim
        total_budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        rng = np.random.default_rng(42)

        def record(x, fx):
            nonlocal n_calls, best_f, best_x
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float, copy=True)

        # Phase 1: Global Exploration
        # Use a mix of uniform and Latin hypercube-like sampling for better coverage
        n_explore = min(50, total_budget // 5)
        candidates = []
        
        # First, do some uniform random
        for _ in range(n_explore // 2):
            if n_calls >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            record(x, fx)
            candidates.append((fx, x))
        
        # Then, do structured sampling (grid-like with jitter)
        if n_explore > n_explore // 2:
            n_per_axis = max(2, int((n_explore - n_explore // 2) ** (1.0 / dim)))
            # Create a grid
            grid_points = []
            for i in range(n_per_axis ** dim):
                if n_calls >= total_budget or time.time() > deadline:
                    break
                x = np.zeros(dim)
                rem = i
                for d in range(dim):
                    x[d] = -5 + (2 * d + 1) * (10.0 / (2 * n_per_axis)) + rng.uniform(-0.5, 0.5) * (10.0 / n_per_axis)
                    rem //= n_per_axis
                x = np.clip(x, -5, 5)
                fx = func(x)
                record(x, fx)
                candidates.append((fx, x))
        
        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Select top unique candidates for restarts
        starts = []
        min_dist = 1.5
        for fx, x in candidates:
            if len(starts) >= 4:
                break
            is_distinct = True
            for _, sx in starts:
                if np.linalg.norm(x - sx) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append((fx, x))
        
        # If we didn't find enough distinct ones, just take the best ones
        if len(starts) < 2:
            for fx, x in candidates:
                if (fx, x) not in starts:
                    starts.append((fx, x))
                if len(starts) >= 2:
                    break

        # Phase 2: Multi-Start CMA-ES
        remaining_budget = total_budget - n_calls
        
        # Define CMA-ES parameters
        lambda_pop = 12
        mu = lambda_pop // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        for i, (f_start, x_start) in enumerate(starts):
            if n_calls >= total_budget or time.time() > deadline:
                break
            
            # Initialize CMA-ES state
            x = np.array(x_start, dtype=float)
            sigma = 1.5
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            gen = 0
            stagnation_counter = 0
            best_f_in_run = best_f
            no_improve_count = 0
            
            # Run CMA-ES
            while n_calls < total_budget and time.time() < deadline:
                # Check stagnation
                if n_calls % 10 == 0:
                    if best_f_in_run == best_f:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                        best_f_in_run = best_f
                
                if stagnation_counter > 8:  # 80 evals without improvement
                    break
                
                # If not enough budget for a full population, do a final random step
                if n_calls + lambda_pop > total_budget:
                    pop_size = total_budget - n_calls
                    for _ in range(pop_size):
                        if time.time() > deadline:
                            break
                        y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                        record(y, func(y))
                    break

                # Generate population
                ys = []
                fs = []
                for j in range(lambda_pop):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    n_calls += 1
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = np.array(y, dtype=float, copy=True)
                
                if len(ys) < lambda_pop:
                    break

                ys = np.array(ys)
                fs = np.array(fs)

                # Sort by fitness
                order = np.argsort(fs)
                ys = ys[order]
                fs = fs[order]

                # Weighted mean
                m_new = np.sum(weights[:, None] * ys[:mu], axis=0)

                # Update CMA matrix
                y_sigma = (m_new - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n) < (1.4 + 2 / (dim + 1))
                pc = (1 - c_c) * pc + hsig * c_c * y_sigma
                delta_hsig = (1 - hsig) * c_c * (1 - c_c)

                v_new = np.zeros((dim, dim))
                for k in range(mu):
                    v_new += weights[k] * np.outer(ys[k] - m_new, ys[k] - m_new)
                
                C = (1 + c_1 * delta_hsig + c_mu - c_1 - c_mu) * C \
                    + c_1 * (np.outer(pc, pc) + delta_hsig) \
                    + c_mu * v_new / (sigma ** 2)

                # Ensure positive definite
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-8 * np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ys[0] - m_new) / sigma - 1))
                sigma = np.clip(sigma, 0.01, 10.0)

                # Move center
                x = m_new
                gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
