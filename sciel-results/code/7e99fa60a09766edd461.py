"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-scale CMA-ES + L-BFGS-B hybrid for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        # Track all evaluated points for multi-start
        eval_history = []  # list of (x, f)

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            eval_history.append((x.copy(), f_val))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Multi-scale global exploration
        # Use a set of strategic points at different scales
        scout_points = []
        
        # Origin and standard candidates
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))
        
        # Corners of the box
        for i in range(2 ** dim):
            p = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    p[j] = 5.0
                else:
                    p[j] = -5.0
            scout_points.append(p)
        
        # Mid-scale points
        for val in [-2.5, 0.0, 2.5]:
            for d in range(dim):
                p = np.zeros(dim)
                p[d] = val
                scout_points.append(p)
        
        # Random points at multiple scales
        for scale in [1.0, 2.0, 4.0]:
            for _ in range(5):
                scout_points.append(rng.uniform(-scale, scale, dim))

        # Evaluate scout points
        for s in scout_points:
            if call_count >= budget or time.time() > deadline:
                break
            try:
                wrapped(s)
            except (RuntimeError, Exception):
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except Exception:
                best_f = np.inf

        # Phase 2: CMA-ES for global refinement
        # Use CMA-ES with a population that adapts to the landscape
        remaining_budget = budget - call_count
        if remaining_budget > 50 and time.time() < deadline:
            # Initialize CMA-ES from the best point
            x0 = best_x.copy()
            sigma0 = 1.0  # Initial step size
            
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            max_iter = min(remaining_budget // pop_size, 50)
            
            # Simple CMA-ES implementation
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)  # Covariance matrix
            B = np.eye(dim)  # Eigenvectors
            D = np.ones(dim)  # Eigenvalues
            lambda_ = pop_size
            weights = np.ones(lambda_) / lambda_
            mu = lambda_ // 2
            c_sigma = (mu + 2) / (dim + mu + 5)
            c_c = (4 + dim / dim) / (dim + 4 + 3 * dim / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * (sqrt(mu_eff) - 2) / ((dim + 2) ** 2 + mu_eff))
            mu_eff = 1 / np.sum(weights[:mu] ** 2)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            for gen in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_i = mean + sigma * y
                    offspring.append(x_i)
                
                # Evaluate offspring
                offspring_f = []
                for x_i in offspring:
                    if call_count >= budget or time.time() > deadline:
                        break
                    try:
                        f_i = wrapped(x_i)
                        offspring_f.append((x_i, f_i))
                    except (RuntimeError, Exception):
                        break
                
                if len(offspring_f) < mu:
                    break
                
                # Sort by fitness
                offspring_f.sort(key=lambda x: x[1])
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * (offspring_f[i][0] - mean) / sigma
                
                # Update mean
                mean = mean + c_1 * y_w
                
                # Update step size
                y_sigma = B.T @ (D ** (-1)) @ (B @ y_w)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update covariance
                h_sigma = (1 - c_sigma) ** (2 * gen) > (1 - c_c) ** 2
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean)
                
                delta_h = (1 - h_sigma) * (1 - c_c) + c_c
                C = (1 + c_1 * delta_h - c_1) * C + c_1 * np.outer(y_w, y_w)
                
                # Update B and D
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                
                # Update best
                if offspring_f[0][1] < best_f:
                    best_f = offspring_f[0][1]
                    best_x = offspring_f[0][0].copy()

        # Phase 3: Multi-start L-BFGS-B from top candidates
        remaining_budget = budget - call_count
        if remaining_budget > 20 and time.time() < deadline:
            # Get top 3 candidates from eval_history
            if eval_history:
                eval_history.sort(key=lambda x: x[1])
                top_candidates = [x for x, f in eval_history[:3]]
            else:
                top_candidates = [best_x]
            
            per_start_budget = max(5, remaining_budget // len(top_candidates))
            
            for x0 in top_candidates:
                if call_count >= budget or time.time() > deadline:
                    break
                
                try:
                    result = minimize(
                        wrapped,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 100,
                            'maxfev': min(per_start_budget, budget - call_count),
                            'gtol': 1e-8,
                            'ftol': 1e-10,
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)
                except (RuntimeError, Exception):
                    pass

        # Phase 4: Final (1+1)-ES polish
        remaining_budget = budget - call_count
        if remaining_budget > 10 and time.time() < deadline:
            x_es = best_x.copy()
            fx_es = best_f
            sigma = 0.05
            max_es_calls = min(30, remaining_budget)

            for _ in range(max_es_calls):
                if call_count >= budget or time.time() > deadline:
                    break
                y = np.clip(x_es + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except (RuntimeError, Exception):
                    break
                if fy < fx_es:
                    x_es, fx_es = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8

                if fx_es < best_f:
                    best_f = fx_es
                    best_x = x_es.copy()

        # Phase 5: Final Powell polish
        remaining_budget = budget - call_count
        if remaining_budget > 5 and time.time() < deadline:
            try:
                result2 = minimize(
                    wrapped,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining_budget,
                        'xtol': 1e-12,
                        'ftol': 1e-14,
                    }
                )
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = np.clip(result2.x, -5.0, 5.0)
            except (RuntimeError, Exception):
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except Exception:
                best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
