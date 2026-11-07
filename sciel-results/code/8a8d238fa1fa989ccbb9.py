import numpy as np
import time

class Optimizer:
    """Local quadratic surrogate model optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Broad random sampling to get a good starting point and initial history
        history = []
        call_count = 0

        def do_eval(x):
            nonlocal call_count
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            history.append((x.copy(), f))
            return f

        # Initial point at origin
        x0 = np.zeros(n)
        f0 = do_eval(x0)

        # Try a few random points to find a better starting point
        best_x = x0.copy()
        best_f = f0
        for _ in range(8):
            if not time_ok() or call_count >= budget - 20:
                break
            x_try = rng.uniform(-4, 4, n)
            f_try = do_eval(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Minimum points needed for quadratic fit: n*(n+1)/2 + 1 = 16 for n=5
        min_points = n * (n + 1) // 2 + 1
        max_points = 25

        # Phase 2: Surrogate model optimization
        while call_count < budget and time_ok():
            # If we have enough points, try to fit a quadratic surrogate
            if len(history) >= min_points:
                recent = history[-max_points:]
                X = np.array([pt for pt, _ in recent])
                F = np.array([f for _, f in recent])

                # Build design matrix for quadratic: [1, x_1..x_n, x_1^2, x_1*x_2, ..., x_n^2]
                m = n * (n + 1) // 2
                Q = np.ones((len(X), 1 + n + m))
                Q[:, 1:1 + n] = X
                idx = 1 + n
                for i in range(n):
                    for j in range(i, n):
                        Q[:, idx] = X[:, i] * X[:, j]
                        idx += 1

                try:
                    coeffs, _, rank, _ = np.linalg.lstsq(Q, F, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = None

                if coeffs is not None and rank >= 1 + n + m - 1:
                    c = coeffs[0]
                    b = coeffs[1:1 + n]
                    A = np.zeros((n, n))
                    idx = 1 + n
                    for i in range(n):
                        for j in range(i, n):
                            val = coeffs[idx]
                            if i == j:
                                A[i, i] = val
                            else:
                                A[i, j] = val / 2.0
                                A[j, i] = val / 2.0
                            idx += 1

                    # Check if A is positive definite (for a minimum)
                    try:
                        eigvals = np.linalg.eigvalsh(A)
                    except np.linalg.LinAlgError:
                        eigvals = None

                    if eigvals is not None and np.all(eigvals > -1e-6):
                        # Find minimizer: grad = b + 2*A*x = 0 => x* = -0.5 * A^{-1} * b
                        try:
                            A_reg = A + 1e-8 * np.eye(n)
                            x_star = -0.5 * np.linalg.solve(A_reg, b)
                        except np.linalg.LinAlgError:
                            x_star = None

                        if x_star is not None:
                            x_star = np.clip(x_star, -5, 5)
                            step = x_star - x
                            step_norm = np.linalg.norm(step)
                            if step_norm > 0:
                                # Damped step: limit to sigma
                                if step_norm > sigma:
                                    step = step / step_norm * sigma
                                x_new = np.clip(x + step, -5, 5)
                                fx_new = do_eval(x_new)

                                if fx_new < fx:
                                    x, fx = x_new, fx_new
                                    sigma = min(sigma * 1.3, 4.0)
                                else:
                                    sigma = max(sigma * 0.7, 0.05)
                                continue

            # Fallback: random perturbation
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = do_eval(y)

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.2, 4.0)
            else:
                sigma = max(sigma * 0.85, 0.05)

        return x, fx
