"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + ES optimizer for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        # Track best solution and function value
        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            if call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Broad scout with diverse starting points
        # Rosenbrock has a long narrow valley; we need to find it first.
        # Use a mix of structured and random starts including corners and midpoints.
        starts = []
        
        # Corners of the box
        for c1 in [-4.5, 4.5]:
            for c2 in [-4.5, 4.5]:
                starts.append(np.array([c1, c2, c1, c2, c1]))
        
        # Midpoints of faces
        for v in [-3, -1.5, 0, 1.5, 3]:
            starts.append(np.full(dim, float(v)))
        
        # Nominal minima candidates
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim))
        starts.append(np.full(dim, 0.5))
        starts.append(np.full(dim, -0.5))
        
        # Points along diagonals
        starts.append(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        starts.append(np.array([1.0, 1.0, 0.5, 0.5, 0.5]))
        starts.append(np.array([0.5, 0.5, 1.0, 1.0, 1.0]))
        
        # Random points
        for _ in range(15):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate all starting points
        for s in starts:
            if call_count >= self.budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            f_val = wrapped(s_clipped)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: Multi-start L-BFGS-B from top diverse scout points
        # L-BFGS-B leverages gradient-based efficiency for smooth Rosenbrock landscapes
        # Collect unique starting points for L-BFGS-B
        lbfgsb_starts = [best_x.copy()]
        
        # Add diverse candidates
        for s in starts:
            if len(lbfgsb_starts) >= 8:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            # Check diversity
            diverse = True
            for existing in lbfgsb_starts:
                if np.linalg.norm(s_clipped - existing) < 0.5:
                    diverse = False
                    break
            if diverse:
                lbfgsb_starts.append(s_clipped)

        # Run L-BFGS-B from each start
        for ls in lbfgsb_starts:
            if call_count >= self.budget - 30 or time.time() > deadline:
                break
            # Run L-BFGS-B from this start
            maxfev = max(10, (self.budget - call_count - 20) // max(1, len(lbfgsb_starts)))
            result = minimize(
                wrapped,
                ls.copy(),
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0)] * dim,
                options={
                    'maxiter': 100,
                    'maxfev': maxfev,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                }
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5.0, 5.0)

        # Phase 3: CMA-ES-like local refinement
        # Use a population-based approach for robust local optimization
        if call_count < self.budget - 20 and time.time() < deadline:
            # Population size for CMA-ES
            lambda_es = max(4, int(4 + 3 * np.log(dim)))
            # Initial sigma based on problem scale
            sigma = 0.3
            x_mean = best_x.copy()
            f_mean = best_f
            step_size = sigma
            
            es_calls = 0
            max_es_calls = min(150, self.budget - call_count - 10)
            no_improve = 0

            for gen in range(max_es_calls):
                if call_count >= self.budget - 5 or time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                for i in range(lambda_es):
                    y = rng.standard_normal(dim)
                    x_off = np.clip(x_mean + step_size * y, -5.0, 5.0)
                    offspring.append(x_off)
                
                # Evaluate offspring
                offspring_f = []
                for x_off in offspring:
                    if call_count >= self.budget or time.time() > deadline:
                        break
                    f_off = wrapped(x_off)
                    offspring_f.append(f_off)
                
                if len(offspring_f) == 0:
                    break
                
                # Sort by fitness
                idx = np.argsort(offspring_f)
                sorted_offspring = [offspring[i] for i in idx]
                sorted_f = [offspring_f[i] for i in idx]
                
                # Update mean (weighted average of top half)
                mu = lambda_es // 2
                weights = np.ones(mu)
                weights[0] = 2.0  # Extra weight for best
                weights /= np.sum(weights)
                
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * sorted_offspring[i]
                
                # Update sigma
                f_new_mean = sorted_f[0]
                if f_new_mean < f_mean:
                    step_size = min(step_size * 1.1, 5.0)
                    no_improve = 0
                else:
                    step_size = max(step_size * 0.9, 0.001)
                    no_improve += 1
                
                x_mean = new_mean
                f_mean = sorted_f[0]
                
                if f_mean < best_f:
                    best_f = f_mean
                    best_x = sorted_offspring[0].copy()
                
                es_calls += 1
                if no_improve > 30:
                    break

        # Phase 4: Final Nelder-Mead polish
        if call_count < self.budget - 5 and time.time() < deadline:
            result2 = minimize(
                wrapped,
                best_x.copy(),
                method='Nelder-Mead',
                options={
                    'maxiter': 50,
                    'maxfev': max(5, self.budget - call_count),
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                }
            )
            if result2.fun < best_f:
                best_f = result2.fun
                best_x = np.clip(result.x, -5.0, 5.0)

        # Phase 5: Final micro-step refinement
        if call_count < self.budget and time.time() < deadline:
            x_es = best_x.copy()
            fx_es = best_f
            sigma = 0.005
            max_es_calls = min(20, self.budget - call_count)
            
            for _ in range(max_es_calls):
                if call_count >= self.budget or time.time() > deadline:
                    break
                y = np.clip(x_es + sigma * rng.standard_normal(dim), -5, 5)
                fy = wrapped(y)
                if fy < fx_es:
                    x_es, fx_es = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                if fx_es < best_f:
                    best_f = fx_es
                    best_x = x_es.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
