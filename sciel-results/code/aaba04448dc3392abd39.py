"""Combined optimizer: Quadratic model fitting for global localization, followed by Hessian-scaled (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic fitting + Hessian-scaled (1+1)-ES polish for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        call_count = 0
        best_x = None
        best_fx = np.inf

        def make_call(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline - 2.0:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Quadratic model fitting to locate the global minimum
        # For Sphere: f(x) = ||x - x*||^2 + f_opt
        # We fit a quadratic model using samples around the origin
        # Number of points needed for full quadratic in d dims: d*(d+1)/2 + d + 1
        # For d=5: 15 + 5 + 1 = 21 points
        
        n_pts = dim * (dim + 1) // 2 + dim + 1  # 21 for dim=5
        
        # Generate design points: center + axis points + cross points
        design = []
        # Center
        design.append(np.zeros(dim))
        
        # Scale for exploration
        scale = 3.0
        
        # Axis points (2*dim)
        for i in range(dim):
            for s in [-1, 1]:
                pt = np.zeros(dim)
                pt[i] = s * scale
                design.append(pt)
        
        # Cross points (dim*(dim-1)/2 pairs, 2 each = dim*(dim-1))
        for i in range(dim):
            for j in range(i + 1, dim):
                for s in [-1, 1]:
                    pt = np.zeros(dim)
                    pt[i] = s * scale / np.sqrt(2)
                    pt[j] = s * scale / np.sqrt(2)
                    design.append(pt)
        
        # Trim to n_pts
        design = design[:n_pts]
        
        # Evaluate design points
        X = np.zeros((len(design), dim))
        F = np.zeros(len(design))
        
        for i, pt in enumerate(design):
            fx = make_call(pt)
            if fx is None:
                break
            X[i] = pt
            F[i] = fx
        
        if call_count < 2:
            # Fallback: just do random search
            for _ in range(budget - call_count):
                if time.time() > deadline - 2.0:
                    break
                x = rng.uniform(-5, 5, dim)
                make_call(x)
            return best_x, best_fx
        
        # Fit quadratic model: f(x) = x^T A x + b^T x + c
        # Number of parameters: d*(d+1)/2 + d + 1
        n_params = dim * (dim + 1) // 2 + dim + 1
        
        # Build design matrix for quadratic fit
        def quad_features(x):
            """Return feature vector for quadratic model."""
            feats = []
            # Quadratic terms: x_i * x_j for i <= j
            for i in range(dim):
                for j in range(i, dim):
                    feats.append(x[i] * x[j])
            # Linear terms
            for i in range(dim):
                feats.append(x[i])
            # Constant
            feats.append(1.0)
            return np.array(feats)
        
        M = np.array([quad_features(x) for x in X])
        
        # Solve least squares
        try:
            params, residuals, rank, sv = np.linalg.lstsq(M, F, rcond=None)
        except np.linalg.LinAlgError:
            params = np.zeros(n_params)
        
        # Extract A, b, c from params
        # params[0:d*(d+1)/2] -> quadratic coefficients
        # params[d*(d+1)/2 : d*(d+1)/2 + d] -> linear coefficients
        # params[-1] -> constant
        
        A = np.zeros((dim, dim))
        idx = 0
        for i in range(dim):
            for j in range(i, dim):
                val = params[idx]
                A[i, j] = val
                if i != j:
                    A[j, i] = val
                idx += 1
        
        b = params[dim * (dim + 1) // 2 : dim * (dim + 1) // 2 + dim]
        c = params[-1]
        
        # For Sphere: f(x) = (x - x*)^T (x - x*) + f_opt
        # = x^T x - 2 x*^T x + x*^T x* + f_opt
        # So A should be ~0.5*I (since we used x_i*x_j features, the quadratic form is x^T A x)
        # Actually our model is x^T A x + b^T x + c
        # For sphere: x^T (0.5*I) x + (-x*)^T x + (x*^T x* + f_opt)
        # Wait: (x-x*)^T(x-x*) = x^T x - 2x*^T x + x*^T x*
        # So A = 0.5*I, b = -x*, c = x*^T x* + f_opt
        
        # Solve for x*: minimize x^T A x + b^T x + c
        # Gradient: 2Ax + b = 0 => x* = -0.5 * A^{-1} b
        
        try:
            if np.allclose(A, 0.5 * np.eye(dim), atol=1.0):
                # Standard sphere: x* = -b (since A = 0.5*I, so -0.5 * (2I)^{-1} * b = -b... wait)
                # 2Ax + b = 0 => x = -A^{-1} b / 2... no
                # d/dx (x^T A x + b^T x) = 2Ax + b = 0 => x = -A^{-1} b / 2
                # Hmm, let me be careful. The quadratic form is x^T A x.
                # d/dx (x^T A x) = (A + A^T) x = 2Ax if A is symmetric.
                # So 2Ax + b = 0 => x = -0.5 * A^{-1} b
                x_star = -0.5 * np.linalg.solve(A, b)
            else:
                x_star = -0.5 * np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback: use linear fit only
            try:
                x_star = -b / (2 * np.mean(np.diag(A)) if np.mean(np.diag(A)) != 0 else 1.0)
            except:
                x_star = np.zeros(dim)
        
        x_star = np.clip(x_star, -5, 5)
        
        # Evaluate at the estimated minimum
        fx_star = make_call(x_star)
        if fx_star is None:
            return best_x, best_fx
        
        # Phase 2: Hessian-scaled (1+1)-ES polish
        # Use the fitted Hessian (2A) to scale the step
        # For sphere, Hessian = 2A ≈ I, so scaling is mild
        # Compute eigenvalues of A for adaptive scaling
        
        try:
            eigvals_A = np.linalg.eigvalsh(A)
            eigvals_A = np.clip(eigvals_A, 1e-10, 1e10)
            # Step scale per dimension: 1/sqrt(2*lambda_i)
            step_scale = 1.0 / np.sqrt(2.0 * eigvals_A)
            # Normalize
            step_scale = step_scale / np.max(step_scale)
        except:
            step_scale = np.ones(dim) / np.sqrt(dim)
        
        # Initialize sigma for (1+1)-ES
        sigma = 0.5
        x_cur = x_star.copy()
        f_cur = fx_star
        
        # 1/5 success rule parameters
        sigma_up = 1.05
        sigma_down = 0.95
        window = 10
        successes = 0
        
        while call_count < budget:
            if time.time() > deadline - 2.0:
                break
            
            # Generate perturbation
            z = rng.standard_normal(dim)
            # Scale by sigma and step_scale
            delta = sigma * step_scale * z
            x_new = x_cur + delta
            x_new = np.clip(x_new, -5, 5)
            
            f_new = make_call(x_new)
            if f_new is None:
                break
            
            # Update best
            if f_new < best_fx:
                best_fx = f_new
                best_x = x_new.copy()
            
            # 1/5 rule
            if f_new < f_cur:
                successes += 1
                x_cur = x_new.copy()
                f_cur = f_new
            else:
                pass
            
            # Adjust sigma every 'window' steps
            if (call_count % window) == 0:
                if successes / window > 0.2:
                    sigma *= sigma_up
                else:
                    sigma *= sigma_down
                successes = 0
                sigma = np.clip(sigma, 1e-10, 10.0)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
