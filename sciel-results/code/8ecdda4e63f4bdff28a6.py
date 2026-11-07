"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Optimizer:
    """Local quadratic surrogate model with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Initialize with a central point
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # Store evaluations: (x, fx)
        evals = deque([(x.copy(), fx)], maxlen=50)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0
        n_params = 1 + dim + dim * (dim + 1) // 2

        while calls < budget:
            if time.time() > deadline:
                break

            y = None
            # Try to fit a local quadratic model if we have enough points
            # Use model every 10 steps or if we have enough points
            if len(evals) >= n_params and (calls % 10 == 0 or calls < 20):
                try:
                    n_fit = min(len(evals), 30)
                    points = list(evals)[-n_fit:]
                    X = np.array([p[0] for p in points])
                    Y = np.array([p[1] for p in points])
                    
                    # Build design matrix
                    A = np.ones((n_fit, n_params))
                    A[:, 1:1+dim] = X
                    col = 1 + dim
                    for i_idx in range(dim):
                        for j_idx in range(i_idx, dim):
                            A[:, col] = X[:, i_idx] * X[:, j_idx]
                            col += 1
                    
                    # Solve least squares
                    coeffs, residuals, rank, sv = np.linalg.lstsq(A, Y, rcond=None)
                    if rank >= n_params - 1:
                        c = coeffs[0]
                        b = coeffs[1:1+dim]
                        H = np.zeros((dim, dim))
                        col = 1 + dim
                        for i_idx in range(dim):
                            for j_idx in range(i_idx, dim):
                                val = coeffs[col]
                                H[i_idx, j_idx] = val
                                if i_idx != j_idx:
                                    H[j_idx, i_idx] = val
                                col += 1
                        
                        try:
                            # Regularize Hessian to ensure positive definite if needed
                            H_reg = H + 1e-4 * np.eye(dim)
                            x_pred = -np.linalg.solve(H_reg, b)
                            x_pred = np.clip(x_pred, -5, 5)
                            
                            dist = np.linalg.norm(x_pred - x)
                            if dist < 5.0:
                                y = x_pred
                            else:
                                step = x_pred - x
                                step /= (np.linalg.norm(step) + 1e-10)
                                y = np.clip(x + 0.5 * dist * step, -5, 5)
                        except np.linalg.LinAlgError:
                            y = None
                except Exception:
                    y = None

            if y is None:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)

            fy = func(y)
            calls += 1
            evals.append((y.copy(), fy))
            
            if fy < fx:
                x, fx = y, fy
                if fy < best_fx:
                    best_x, best_fx = y.copy(), fy
                sigma *= 1.1
            else:
                sigma *= 0.95
                if sigma < 0.01:
                    sigma = 0.1

        return best_x, best_fx
# EVOLVE-BLOCK-END
