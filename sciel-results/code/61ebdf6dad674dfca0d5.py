"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Start at a well-chosen point (not too close to boundary).
    2. Estimate the gradient via finite differences (dim+1 calls total).
    3. Jump directly to the optimal corner of the box.
    4. Verify with a few nearby points and use remaining budget for refinement.
    5. Deadline guard to ensure we finish within 300s (with margin).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin
        calls_made = 0
        dim = self.dim
        budget = self.budget

        # Start at a point away from boundaries to avoid clipping issues in finite differences
        x1 = np.zeros(dim)
        f1 = func(x1)
        calls_made += 1
        best_x = x1.copy()
        best_fx = f1

        # Estimate gradient using central differences where possible,
        # forward/backward differences at boundaries.
        eps = 0.5
        a_est = np.zeros(dim)

        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break

            # Try central difference: x1[i] + eps and x1[i] - eps
            xp = x1.copy()
            xp[i] += eps
            xm = x1.copy()
            xm[i] -= eps

            xp_clipped = np.clip(xp, -5, 5)
            xm_clipped = np.clip(xm, -5, 5)

            if np.allclose(xp_clipped, xp) and np.allclose(xm_clipped, xm):
                # Both points are in bounds - use central difference
                fp = func(xp_clipped)
                calls_made += 1
                fm = func(xm_clipped)
                calls_made += 1
                a_est[i] = (fp - fm) / (2 * eps)
            else:
                # One or both are clipped - use one-sided difference
                if not np.allclose(xp_clipped, x1):
                    # Use forward difference
                    fp = func(xp_clipped)
                    calls_made += 1
                    actual_dx = xp_clipped[i] - x1[i]
                    a_est[i] = (fp - f1) / actual_dx
                elif not np.allclose(xm_clipped, x1):
                    # Use backward difference
                    fm = func(xm_clipped)
                    calls_made += 1
                    actual_dx = xm_clipped[i] - x1[i]
                    a_est[i] = (fm - f1) / actual_dx
                else:
                    # Can't move in either direction (shouldn't happen at x1=0)
                    a_est[i] = 0.0

        # Jump to the optimal corner based on estimated gradient
        # For a linear function f(x) = a^T x + b, the minimum over [-5,5]^d is at:
        # x_i = -5 if a_i > 0, else x_i = 5 if a_i < 0, else either
        x_corner = np.where(a_est > 0, -5.0, 5.0)
        x_corner = np.clip(x_corner, -5, 5)

        f_corner = func(x_corner)
        calls_made += 1
        if f_corner < best_fx:
            best_fx = f_corner
            best_x = x_corner.copy()

        # Verify: check the corner and a few neighbors to ensure we're at the true minimum
        # For a linear function, the minimum is exactly at a vertex, so this is just verification
        # Check adjacent vertices (flip one coordinate) to handle edge cases
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            neighbor = best_x.copy()
            # Flip the i-th coordinate to the other bound
            if neighbor[i] > 0:
                neighbor[i] = -5.0
            else:
                neighbor[i] = 5.0
            f_neighbor = func(neighbor)
            calls_made += 1
            if f_neighbor < best_fx:
                best_fx = f_neighbor
                best_x = neighbor.copy()

        # Use remaining budget for a simple refinement just in case
        # (e.g., if the function has any non-linear component or numerical issues)
        if calls_made < budget and time.time() < deadline:
            sigma = 0.1
            x_cur = best_x.copy()
            fx_cur = best_fx
            remaining = budget - calls_made

            for _ in range(remaining):
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x_cur + sigma * np.random.randn(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx_cur:
                    x_cur = y
                    fx_cur = fy
                    sigma *= 1.1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-4:
                        break

        return best_x, best_fx

# EVOLVE-BLOCK-END
