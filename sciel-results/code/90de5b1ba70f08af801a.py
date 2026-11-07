"""Combined (1+1)-ES with CMA-ES-like covariance adaptation and L-BFGS-B polish for BBOB f12 (bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive (1+1)-ES with rank-1 covariance updates, then L-BFGS-B local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        # Track global best
        best_x = None
        best_f = np.inf
        total_calls = 0

        def record_call(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def check_budget():
            nonlocal total_calls
            if total_calls >= budget or time.time() > deadline:
                return False
            return True

        # Phase 1: Adaptive (1+1)-ES with covariance adaptation
        # This handles the global exploration and adapts to the anisotropic landscape
        
        # Reserve budget for L-BFGS-B polish
        # L-BFGS-B in dim=5 with ~50 maxiter uses roughly 50-100 function calls
        polish_budget = min(100, budget // 5)
        es_budget = budget - polish_budget

        # Initial point: random uniform in [-5, 5]
        x = rng.uniform(-5, 5, dim)
        total_calls += 1
        fx = func(x)
        record_call(x, fx)

        # Adaptive step size
        sigma = 2.0
        # Covariance matrix for anisotropic adaptation
        C = np.eye(dim)
        n = dim

        # Success rate tracking window
        window_size = 15
        successes = []

        # Run (1+1)-ES
        while total_calls < es_budget and time.time() < deadline:
            # Generate candidate step using covariance
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)

            z = rng.standard_normal(n)
            step = sigma * np.dot(L, z)
            y = np.clip(x + step, -5, 5)

            fy = func(y)
            total_calls += 1

            success = fy < fx
            successes.append(1 if success else 0)
            if len(successes) > window_size:
                successes.pop(0)

            if success:
                x = y
                fx = fy
                record_call(x, fx)

                # Rank-1 covariance update on success
                mu_c = 0.2
                C = (1 - mu_c) * C + mu_c * np.outer(z, z)
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (n / trace_C)

                # Increase sigma slightly on success
                sigma *= 1.05
            else:
                # Decrease sigma on failure
                sigma *= 0.95

            # Adapt sigma based on success rate in window
            if len(successes) >= 5:
                success_rate = np.mean(successes)
                # Target ~1/3 success rate
                if success_rate > 0.4:
                    sigma *= 1.02
                elif success_rate < 0.2:
                    sigma *= 0.98

            # Clamp sigma
            sigma = np.clip(sigma, 1e-5, 15.0)

            # Periodic re-centering to prevent covariance drift
            if total_calls % 20 == 0:
                C = 0.9 * C + 0.1 * np.eye(n)
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (n / trace_C)

            # Periodic eigenvalue clipping for numerical stability
            if total_calls % 50 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 0.01, 50.0)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (n / trace_C)

        # Phase 2: L-BFGS-B local polish from best point found
        remaining_budget = budget - total_calls
        if remaining_budget > 10 and time.time() < deadline and best_x is not None:
            nm_calls = [0]
            
            def lbfgsb_func(z):
                nonlocal total_calls
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining_budget:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                total_calls += 1
                # Clip to box to ensure validity
                z_clipped = np.clip(z, -5, 5)
                val = func(z_clipped)
                record_call(z_clipped, val)
                return val
            
            try:
                result = minimize(
                    lbfgsb_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfun': remaining_budget,
                        'ftol': 1e-12,
                        'gtol': 1e-10
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
