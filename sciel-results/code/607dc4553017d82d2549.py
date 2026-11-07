"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """RBF trust-region surrogate with coordinate-wise Hooke-Jeeves refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        d = self.dim
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        # Store evaluations for surrogate model
        # RBF model needs at least d+1 points for interpolation, but more is better
        hist_X = []
        hist_f = []
        max_hist = 40  # larger window for better global structure capture

        # Phase 1: Aggressive exploration
        sigma = 2.5
        sigma_cap = 15.0
        growth = 1.3
        decay = 0.8
        phase1_end = 150  # switch to polish after ~150 calls

        calls = 1

        for i in range(self.budget - 1):
            if time.time() > deadline:
                break

            # Add current point to history
            hist_X.append(x.copy())
            hist_f.append(fx)
            if len(hist_X) > max_hist:
                hist_X.pop(0)
                hist_f.pop(0)

            # Phase transition: switch to polish mode
            if i >= phase1_end:
                sigma = 0.05
                growth = 1.1
                decay = 0.7
                sigma_cap = 1.0

            # Try to use RBF surrogate model if we have enough data
            used_surrogate = False
            if len(hist_X) >= 15 and i > 10:
                try:
                    X = np.array(hist_X)
                    f_vals = np.array(hist_f)
                    
                    # Build RBF model: f(x) = sum_j w_j * phi(||x - x_j||) + c
                    # Using Gaussian RBF: phi(r) = exp(-beta * r^2)
                    n = len(X)
                    
                    # Choose beta adaptively based on point spacing
                    # Compute median distance between points
                    dists = []
                    for a in range(min(n, 10)):
                        for b in range(a+1, min(n, 10)):
                            dists.append(np.linalg.norm(X[a] - X[b]))
                    if dists:
                        median_dist = np.median(dists)
                        beta = 1.0 / (median_dist ** 2) if median_dist > 1e-6 else 1.0
                    else:
                        beta = 1.0
                    
                    # Build RBF matrix
                    A = np.zeros((n, n + 1))
                    for a in range(n):
                        for b in range(n):
                            r2 = np.sum((X[a] - X[b]) ** 2)
                            A[a, b] = np.exp(-beta * r2)
                        A[a, n] = 1.0  # constant term
                    
                    # Solve for weights
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(A, f_vals, rcond=None)
                        w = coeffs[:n]
                        c0 = coeffs[n]
                    except np.linalg.LinAlgError:
                        continue
                    
                    # Predict function at several candidate points
                    # 1. Current best
                    # 2. Random points in trust region
                    # 3. Points along coordinate axes (for Hooke-Jeeves style)
                    
                    def predict_rbf(xq):
                        r2 = np.sum((X - xq) ** 2, axis=1)
                        return np.dot(w, np.exp(-beta * r2)) + c0
                    
                    # Generate candidate points
                    candidates = []
                    
                    # Trust region radius
                    trust_radius = max(sigma * 2.0, 0.5)
                    
                    # 1. Random points in trust region
                    for _ in range(10):
                        xq = x + trust_radius * rng.standard_normal(d)
                        xq = np.clip(xq, -5, 5)
                        candidates.append(xq)
                    
                    # 2. Coordinate-wise moves (Hooke-Jeeves style)
                    for coord in range(d):
                        for direction in [-1, 1]:
                            step = trust_radius / np.sqrt(d)
                            xq = x.copy()
                            xq[coord] += direction * step
                            xq = np.clip(xq, -5, 5)
                            candidates.append(xq)
                    
                    # 3. Gradient-like direction from RBF model
                    # Compute numerical gradient at x
                    eps = 1e-4
                    grad = np.zeros(d)
                    for coord in range(d):
                        xp = x.copy()
                        xm = x.copy()
                        xp[coord] += eps
                        xm[coord] -= eps
                        xp = np.clip(xp, -5, 5)
                        xm = np.clip(xm, -5, 5)
                        grad[coord] = (predict_rbf(xp) - predict_rbf(xm)) / (2 * eps)
                    
                    # Move in negative gradient direction
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 1e-8:
                        xq = x - trust_radius * grad / grad_norm
                        xq = np.clip(xq, -5, 5)
                        candidates.append(xq)
                    
                    # 4. Best point from history (in case it's better)
                    best_idx = np.argmin(f_vals)
                    candidates.append(X[best_idx])
                    
                    # Score candidates using RBF prediction
                    if candidates:
                        candidates = np.array(candidates)
                        preds = np.array([predict_rbf(xq) for xq in candidates])
                        best_cand_idx = np.argmin(preds)
                        x_pred = candidates[best_cand_idx]
                        
                        # Only take the step if it's not too far
                        dist = np.linalg.norm(x_pred - x)
                        max_dist = max(trust_radius * 1.5, 1.0)
                        if dist < max_dist and dist > 1e-8:
                            fy = func(x_pred)
                            calls += 1
                            if fy < fx:
                                x, fx = x_pred, fy
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                used_surrogate = True
                            else:
                                # Surrogate was wrong, fall back to ES
                                pass
                except Exception:
                    pass

            if not used_surrogate:
                # Standard (1+1)-ES step
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= growth
                else:
                    sigma *= decay

            sigma = np.clip(sigma, 1e-6, sigma_cap)

        return best_x, best_f
# EVOLVE-BLOCK-END
