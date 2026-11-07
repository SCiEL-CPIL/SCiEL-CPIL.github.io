"""Quadratic surrogate model optimizer for BBOB-style Discus function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic surrogate model optimizer exploiting Discus quadratic structure."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Build initial quadratic model with 32 deterministic points
        # Center + spherical shell points for global quadratic structure
        n_init = 32
        X = []
        F = []
        
        # Add origin
        fx = try_eval(np.zeros(dim))
        if fx is not None:
            X.append(np.zeros(dim))
            F.append(fx)
        
        # Add spherical shell points: 2*dim along axes + additional for full coverage
        # For dim=5: 10 axis points + 22 more = 32 total (including origin)
        # Actually let's do: origin + 10 axis points (±e_i) + 21 more on sphere
        
        # Axis points: ±e_i * r for i in range(dim)
        r = 2.0  # radius for initial exploration
        for i in range(dim):
            for sign in [1.0, -1.0]:
                x = np.zeros(dim)
                x[i] = sign * r
                fx = try_eval(x)
                if fx is not None:
                    X.append(x)
                    F.append(fx)
        
        # Additional points on sphere to complete quadratic model
        # We need enough points to fit a full quadratic: 1 + dim + dim*(dim+1)/2 = 1+5+15=21 params
        # We have 1 + 10 = 11 so far, need 10 more
        rng = np.random.default_rng(42)
        while len(X) < n_init and call_count < budget:
            x = rng.uniform(-r, r, size=dim)
            # Normalize to sphere of radius r
            norm = np.linalg.norm(x)
            if norm > 1e-10:
                x = x / norm * r
            fx = try_eval(x)
            if fx is not None:
                X.append(x)
                F.append(fx)
        
        if len(X) < 2:
            # Fallback: just return what we have
            return best_x, best_f

        X = np.array(X)
        F = np.array(F)

        # Phase 2: Iteratively refine quadratic model
        # Fit quadratic: f(x) ≈ c + b^T x + x^T Q x
        # Design matrix: [1, x, x_i*x_j for i<=j]
        
        def build_design_matrix(X):
            """Build design matrix for quadratic model."""
            n, d = X.shape
            cols = [np.ones(n)]  # constant
            cols.append(X)  # linear terms
            # Quadratic terms: x_i*x_j for i<=j
            quad_terms = []
            for i in range(d):
                for j in range(i, d):
                    quad_terms.append(X[:, i] * X[:, j])
            if quad_terms:
                cols.append(np.column_stack(quad_terms))
            return np.hstack(cols)

        def fit_quadratic(X, F):
            """Fit quadratic model, return (c, b, Q)."""
            D = build_design_matrix(X)
            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(D, F, rcond=None)
            except np.linalg.LinAlgError:
                return None, None, None
            
            c = coeffs[0]
            b = coeffs[1:1+dim]
            q_coeffs = coeffs[1+dim:]
            
            # Reconstruct Q from coefficients
            # q_coeffs corresponds to x_i*x_j for i<=j
            Q = np.zeros((dim, dim))
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    Q[i, j] = q_coeffs[idx]
                    Q[j, i] = q_coeffs[idx]
                    idx += 1
            # Note: the quadratic term in the model is x^T Q x = sum_{i,j} Q[i,j] x_i x_j
            # But our basis has x_i*x_j for i<=j, so we need to be careful
            # Actually, x^T Q x = sum_i Q[i,i] x_i^2 + 2*sum_{i<j} Q[i,j] x_i x_j
            # Our basis: x_i^2 for i, and x_i*x_j for i<j
            # So coeffs[i,i] = Q[i,i], and coeffs[i,j] (i<j) = 2*Q[i,j]
            # Let me redo this properly
            
            Q = np.zeros((dim, dim))
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        Q[i, i] = q_coeffs[idx]
                    else:
                        Q[i, j] = q_coeffs[idx] / 2.0
                        Q[j, i] = q_coeffs[idx] / 2.0
                    idx += 1
            
            return c, b, Q

        def predict(x, c, b, Q):
            """Predict f(x) from quadratic model."""
            return c + b @ x + x @ Q @ x

        def find_min_quadratic(c, b, Q, bounds=(-5.0, 5.0)):
            """Find minimum of quadratic model within bounds."""
            # Unconstrained min: -Q^{-1} b
            try:
                Q_inv = np.linalg.inv(Q)
                x_min = -Q_inv @ b
            except np.linalg.LinAlgError:
                # Q is singular, use pseudo-inverse
                try:
                    Q_pinv = np.linalg.pinv(Q)
                    x_min = -Q_pinv @ b
                except:
                    return np.zeros(dim)
            
            # Clip to bounds
            x_min = np.clip(x_min, bounds[0], bounds[1])
            return x_min

        # Iterative refinement
        max_iterations = 50
        for iteration in range(max_iterations):
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break
            
            # Fit quadratic model
            c, b, Q = fit_quadratic(X, F)
            if c is None:
                break
            
            # Find predicted minimum
            x_pred = find_min_quadratic(c, b, Q)
            
            # Evaluate at predicted minimum
            fx = try_eval(x_pred)
            if fx is None:
                break
            
            # Add new point to model
            X = np.vstack([X, x_pred])
            F = np.append(F, fx)
            
            # If we've converged (predicted min is close to current best), stop
            if np.linalg.norm(x_pred - best_x) < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
