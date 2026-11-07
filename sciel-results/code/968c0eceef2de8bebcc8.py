import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            if time.time() > deadline:
                return best_f
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Quadratic Fit to exploit Sphere structure
        # f(x) = x^T Q x + q^T x + c
        # Parameters: dim + dim*(dim+1)/2 + 1 = 5 + 15 + 1 = 21 for dim=5
        # Use 25 points for stability
        
        n_points = 25
        if self.budget < n_points + 10:
            # Fallback: simple (1+1)-ES
            return self._simple_es(func, best_x, best_f)

        # Generate well-spread points for quadratic fit
        # Use a combination of structured and random points
        points = []
        rng = np.random.default_rng(42)

        # Center
        points.append(np.zeros(dim))

        # Axis-aligned points at ±2
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 2.0
            p2 = np.zeros(dim); p2[i] = -2.0
            points.append(p1)
            p2 = np.zeros(dim); p2[i] = -2.0
            points.append(p2)

        # Face-centered points (pairs of axes at ±2)
        for i in range(dim):
            for j in range(i + 1, dim):
                p = np.zeros(dim)
                p[i] = 2.0
                p[j] = 2.0
                points.append(p)

        # Random points to fill up to n_points
        while len(points) < n_points:
            p = rng.uniform(-4, 4, dim)
            points.append(p)

        points = np.array(points[:n_points])

        # Evaluate all points
        values = np.array([evaluate(p) for p in points])

        # Check if we got valid values
        if np.any(~np.isfinite(values)):
            # Fallback to ES if any evaluation failed
            return self._polish_with_es(func, best_x, best_f)

        # Construct feature vector for quadratic model
        def feature_vector(x):
            dim_local = len(x)
            quad_terms = []
            for i in range(dim_local):
                for j in range(i, dim_local):
                    quad_terms.append(x[i] * x[j])
            lin_terms = list(x)
            const = [1.0]
            return np.array(quad_terms + lin_terms + const)

        A = np.array([feature_vector(p) for p in points])
        b = values

        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

            # Extract Q, q, c
            n_quad = dim * (dim + 1) // 2
            Q_flat = coeffs[:n_quad]
            q = coeffs[n_quad:n_quad + dim]
            c = coeffs[n_quad + dim]

            # Reconstruct symmetric Q matrix
            Q = np.zeros((dim, dim))
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    val = Q_flat[idx]
                    Q[i, j] = val
                    Q[j, i] = val
                    idx += 1

            # Solve for optimum: Q x = -0.5 * q
            try:
                # Check if Q is positive definite (should be for Sphere)
                eigvals = np.linalg.eigvalsh(Q)
                if np.all(eigvals > 1e-10):
                    x_opt = np.linalg.solve(Q, -0.5 * q)
                else:
                    # Use pseudo-inverse if not positive definite
                    x_opt = np.linalg.pinv(Q) @ (-0.5 * q)
            except np.linalg.LinAlgError:
                x_opt = np.linalg.pinv(Q) @ (-0.5 * q)

            x_opt = np.clip(x_opt, -5, 5)
            f_opt = evaluate(x_opt)

            # Update best if this is better
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        except Exception:
            # If fitting fails, proceed with best found so far
            pass

        # Phase 2: Polish with (1+1)-ES starting from best_x
        x = best_x.copy()
        fx = best_f

        # Adaptive sigma with 1/5 success rule approximation
        sigma = 0.5
        rng = np.random.default_rng(0)

        # Track recent successes for better sigma adaptation
        recent_successes = 0
        window_size = 10

        while calls_made < self.budget and time.time() < deadline:
            # Generate step
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = evaluate(y)

            if fy < fx:
                x = y
                fx = fy
                recent_successes += 1
                # Increase sigma on success
                sigma *= 1.3
            else:
                # Decrease sigma on failure
                sigma *= 0.85

            # Clamp sigma
            if sigma > 5.0:
                sigma = 5.0
            if sigma < 1e-10:
                break

        return best_x, best_f

    def _simple_es(self, func, best_x, best_f):
        """Simple (1+1)-ES fallback for small budgets."""
        dim = self.dim
        calls_made = 0
        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        rng = np.random.default_rng(0)

        start_time = time.time()
        deadline = start_time + 280

        for _ in range(self.budget):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
            else:
                sigma *= 0.87

        return x, fx

    def _polish_with_es(self, func, best_x, best_f):
        """Polish with (1+1)-ES when quadratic fit fails."""
        dim = self.dim
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        rng = np.random.default_rng(0)

        start_time = time.time()
        deadline = start_time + 280
        calls_made = 0

        while calls_made < self.budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            if sigma < 1e-10:
                break

        return x, fx
