# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Derivative-free optimizer using adaptive coordinate descent with
    dynamic rotation of the search basis via principal components of
    recent evaluation history.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        B = self.budget

        # Initialize at a random point in the box
        x = rng.uniform(-4, 4, d)
        fx = func(x)
        calls = 1

        # History of points for PCA
        # We store the last 'hist_size' points
        hist_size = 20
        points = [x.copy()]

        # Initial basis: identity
        basis = np.eye(d)
        
        # Step sizes for each axis in the rotated frame
        # Start with moderate steps
        steps = np.ones(d) * 0.5

        # PCA update interval
        pca_interval = 10
        # Minimum number of points before rotating
        min_hist = 10

        for i in range(B - 1):
            # Determine search directions: current basis vectors
            # For each axis in the rotated frame, try a line search
            for k in range(d):
                if calls >= B:
                    break
                direction = basis[:, k]
                
                # Adaptive line search along this direction
                best_x = x
                best_f = fx
                improved = False
                
                # Try positive direction
                trial = x + steps[k] * direction
                trial = np.clip(trial, -5, 5)
                # Check if movement is significant
                if np.linalg.norm(trial - x) > 1e-8:
                    ft = func(trial)
                    calls += 1
                    if ft < best_f:
                        best_f = ft
                        best_x = trial
                        improved = True

                # Try negative direction
                trial = x - steps[k] * direction
                trial = np.clip(trial, -5, 5)
                if np.linalg.norm(trial - x) > 1e-8:
                    ft = func(trial)
                    calls += 1
                    if ft < best_f:
                        best_f = ft
                        best_x = trial
                        improved = True

                if improved:
                    x = best_x
                    fx = best_f
                    # Increase step size on success, but cap it
                    steps[k] = min(steps[k] * 1.1, 4.0)
                else:
                    # Shrink step size on failure
                    steps[k] = max(steps[k] * 0.7, 0.01)

            # Update history
            points.append(x.copy())
            if len(points) > hist_size:
                points.pop(0)

            # Periodically update the basis using PCA of recent points
            if len(points) >= min_hist and (i + 1) % pca_interval == 0:
                pts = np.array(points)
                # Center the points
                mean_pt = np.mean(pts, axis=0)
                centered = pts - mean_pt
                # Covariance matrix
                cov = np.cov(centered, rowvar=False)
                
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    # Sort by eigenvalue descending
                    idx = np.argsort(eigvals)[::-1]
                    new_basis = eigvecs[:, idx]
                    
                    # Orthonormalize to ensure stability
                    Q, R = np.linalg.qr(new_basis)
                    if np.linalg.det(Q) < 0:
                        Q[:, 0] = -Q[:, 0]
                    basis = Q
                    
                except np.linalg.LinAlgError:
                    pass

        return x, fx
# EVOLVE-BLOCK-END
