<<<<<<< SEARCH
"""Rewritten optimizer using a local quadratic surrogate model."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Local quadratic surrogate model optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Initialize with a few random points to build initial history
        rng = np.random.default_rng(42)

        # We'll keep a history of points and function values
        # For the quadratic fit, we need at least n*(n+1)/2 + 1 points
        # But we'll use a sliding window of the last 25 evaluations

        # Start with a reasonable initial point
        x = rng.uniform(-3, 3, n)
        fx = func(x)

        # History: list of (x, f) pairs
        history = [(x.copy(), fx)]
        call_count = 1

        # Initial step size
        sigma = 1.0

        # Minimum number of points needed for quadratic fit
        min_points = n * (n + 1) // 2 + 1  # 16 for n=5
        # Use up to 25 points for the fit
        max_points = 25

        while call_count < budget:
            # If we have enough points, try to fit a quadratic
            if len(history) >= min_points:
                # Take the last max_points evaluations
                recent = history[-max_points:]
                X = np.array([pt for pt, _ in recent])
                F = np.array([f for _, f in recent])

                # Fit quadratic: f(x) ≈ c + x^T b + x^T A x
                # Design matrix for quadratic terms
                # For each point x, create [1, x_1, ..., x_n, x_1^2, x_1*x_2, ..., x_n^2]
                m = n * (n + 1) // 2
                quad_terms = []
                for xi in X:
                    row = [1.0]
                    row.extend(xi)
                    # Squared and cross terms
                    for i in range(n):
                        for j in range(i, n):
                            row.append(xi[i] * xi[j])
                    quad_terms.append(row)
                Q = np.array(quad_terms)

                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(Q, F, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = None

                if coeffs is not None:
                    c = coeffs[0]
                    b = coeffs[1:1 + n]
                    # Reconstruct A from the quadratic coefficients
                    # The quadratic part is sum_{i<=j} coeffs[n+1 + idx(i,j)] * x_i * x_j
                    # where idx(i,j) = i*n - i*(i-1)//2 + (j - i) ... let's just index carefully
                    # Actually, let's reconstruct A properly
                    # The quadratic form x^T A x = sum_i A_ii x_i^2 + 2*sum_{i<j} A_ij x_i x_j
                    # Our design matrix has terms: x_i^2 and x_i*x_j for i<j
                    # So coeffs for x_i^2 correspond to A_ii
                    # coeffs for x_i*x_j (i<j) correspond to 2*A_ij

                    A = np.zeros((n, n))
                    idx = 1 + n  # start index for quadratic terms
                    for i in range(n):
                        for j in range(i, n):
                            val = coeffs[idx]
                            if i == j:
                                A[i, i] = val
                            else:
                                A[i, j] = val / 2.0
                                A[j, i] = val / 2.0
                            idx += 1

                    # Find minimizer of the quadratic model
                    # grad = b + 2*A*x = 0 => x* = -0.5 * A^{-1} * b
                    try:
                        # Add small regularization for stability
                        A_reg = A + 1e-6 * np.eye(n)
                        x_star = -0.5 * np.linalg.solve(A_reg, b)
                    except np.linalg.LinAlgError:
                        x_star = None

                    if x_star is not None:
                        # Clip to bounds
                        x_star = np.clip(x_star, -5, 5)

                        # Take a step toward x_star, but limit the step size
                        # Use damped step
                        step = x_star - x
                        step_norm = np.linalg.norm(step)
                        if step_norm > sigma:
                            step = step / step_norm * sigma

                        x_new = np.clip(x + step, -5, 5)
                        fx_new = func(x_new)
                        call_count += 1

                        if fx_new < fx:
                            x, fx = x_new, fx_new
                            # Adapt sigma: increase if we made progress
                            sigma = min(sigma * 1.3, 4.0)
                        else:
                            sigma = max(sigma * 0.7, 0.05)

                        history.append((x_new.copy(), fx_new))
                        continue

            # If quadratic fit failed or we don't have enough points,
            # do a random perturbation
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.2, 4.0)
            else:
                sigma = max(sigma * 0.85, 0.05)

            history.append((y.copy(), fy))

        return x, fx
# EVOLVE-BLOCK-END
=======
"""Rewritten optimizer using a local quadratic surrogate model."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Local quadratic surrogate model optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within the instance
        rng = np.random.default_rng(42)

        # History of (x, f) pairs
        history = []
        call_count = 0

        # Initial point
        x_best = rng.uniform(-3, 3, n)
        f_best = func(x_best)
        call_count += 1
        history.append((x_best.copy(), f_best))

        # Initial step size for exploration if model fails
        sigma = 1.0
        
        # Number of points to use for the fit
        # Quadratic in n dims has n(n+1)/2 + n + 1 coefficients? 
        # Model: f ~ c + b^T x + x^T A x
        # Coeffs: 1 + n + n(n+1)/2. For n=5, 1+5+15=21.
        # We need at least that many points. Let's use 25 for stability.
        min_points = n * (n + 1) // 2 + n + 1 
        fit_points = 25

        while call_count < budget:
            # Check if we have enough points to fit
            if len(history) >= min_points:
                # Select the 'fit_points' points closest to x_best
                # This ensures the model is local
                X_all = np.array([pt for pt, _ in history])
                dists = np.linalg.norm(X_all - x_best, axis=1)
                
                if len(history) > fit_points:
                    idx = np.argsort(dists)[:fit_points]
                    X_sel = X_all[idx]
                    F_sel = np.array([f for _, f in history])[idx]
                else:
                    X_sel = X_all
                    F_sel = np.array([f for _, f in history])
                
                # Center around x_best to improve numerical stability
                # z = x - x_best
                # y = f(x) - f_best
                Z = X_sel - x_best
                Y = F_sel - f_best
                
                # Build design matrix for quadratic fit of Y ~ Z^T H Z + W^T Z
                # We ignore the constant term because Y is centered (offset removed).
                # Model: Y = W^T Z + Z^T H Z
                
                n_quad = n * (n + 1) // 2
                D = np.zeros((len(Z), n + n_quad))
                
                # Linear part
                D[:, :n] = Z
                
                # Quadratic part
                idx = n
                for i in range(n):
                    for j in range(i, n):
                        D[:, idx] = Z[:, i] * Z[:, j]
                        idx += 1
                
                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(D, Y, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = None

                if coeffs is not None:
                    # Extract W and H coefficients
                    W = coeffs[:n]
                    H_coeffs = coeffs[n:]
                    
                    # Reconstruct H
                    H = np.zeros((n, n))
                    idx = 0
                    for i in range(n):
                        for j in range(i, n):
                            val = H_coeffs[idx]
                            if i == j:
                                H[i, i] = val
                            else:
                                H[i, j] = val / 2.0
                                H[j, i] = val / 2.0
                            idx += 1
                    
                    # Find minimizer of z^T H z + W^T z
                    # Grad: 2 H z + W = 0 => z* = -0.5 H^{-1} W
                    # Add regularization to handle non-convex or ill-conditioned cases
                    H_reg = H + 1e-4 * np.eye(n)
                    
                    try:
                        z_star = -0.5 * np.linalg.solve(H_reg, W)
                    except np.linalg.LinAlgError:
                        z_star = None

                    if z_star is not None:
                        # Propose new point
                        x_proposed = x_best + z_star
                        
                        # Clip to bounds
                        x_proposed = np.clip(x_proposed, -5, 5)
                        
                        # Check if it's a valid step (not identical to current)
                        dist = np.linalg.norm(x_proposed - x_best)
                        
                        if dist < 1e-6:
                            # Model suggests we are at the minimum.
                            # Do a random step to escape potential local minima or numerical noise
                            x_new = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
                            f_new = func(x_new)
                            call_count += 1
                            if f_new < f_best:
                                x_best, f_best = x_new, f_new
                                sigma = min(sigma * 1.2, 4.0)
                            else:
                                sigma = max(sigma * 0.8, 0.01)
                            history.append((x_new.copy(), f_new))
                            continue

                        # Evaluate proposed point
                        f_proposed = func(x_proposed)
                        call_count += 1
                        
                        if f_proposed < f_best:
                            x_best, f_best = x_proposed, f_proposed
                            # Success: keep sigma or adjust? 
                            # We can slightly decrease sigma to refine if we are close?
                            # Or keep it for fallback.
                            pass
                        else:
                            # Failure: The model was wrong.
                            # Do a random step to explore
                            x_new = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
                            f_new = func(x_new)
                            call_count += 1
                            if f_new < f_best:
                                x_best, f_best = x_new, f_new
                                sigma = min(sigma * 1.2, 4.0)
                            else:
                                sigma = max(sigma * 0.8, 0.01)
                            history.append((x_new.copy(), f_new))
                            continue
                        
                        history.append((x_proposed.copy(), f_proposed))
                        continue

            # If not enough points, do random exploration
            x_new = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
            f_new = func(x_new)
            call_count += 1
            
            if f_new < f_best:
                x_best, f_best = x_new, f_new
                sigma = min(sigma * 1.2, 4.0)
            else:
                sigma = max(sigma * 0.8, 0.01)
            
            history.append((x_new.copy(), f_new))

        return x_best, f_best
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
