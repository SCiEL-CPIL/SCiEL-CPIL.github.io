import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0 # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # State
        # We will maintain a set of points
        # Initial guess
        x_c = np.zeros(d)
        f_c = func(x_c)
        
        # Initial Delta
        delta = 1.0
        
        # Store points and values
        # We want at least 21 points for a full quadratic fit in d=5
        # Let's start with a grid of 2d+1 points
        points = [x_c]
        values = [f_c]
        
        # Generate initial neighbors
        for i in range(d):
            for sign in [-1, 1]:
                x_new = x_c.copy()
                x_new[i] += sign * delta
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                points.append(x_new)
                values.append(f_new)
                if time.time() > deadline: return x_c, f_c
                
        # Update best
        best_x = x_c
        best_f = f_c
        for i in range(len(values)):
            if values[i] < best_f:
                best_f = values[i]
                best_x = points[i]
        
        # Set center to best
        x_c = best_x
        f_c = best_f
        
        # Main loop
        calls_made = len(values)
        
        # We need a function to fit the quadratic model
        def fit_quadratic(pts, vals, center):
            # pts: list of arrays, vals: list of floats
            # center: array
            # Returns g, H, b (where f(x) ~ b + g^T(x-c) + 0.5 (x-c)^T H (x-c))
            n = len(pts)
            if n < 21: # Not enough points for full fit, use regularization or simpler model
                # For now, if not enough, we might just do a gradient step based on available points
                # But we start with 11. Let's add more points in the first few iterations if needed.
                # Actually, 11 points is not enough for 21 params.
                # We will add points as we go.
                pass
                
            # Build design matrix
            # Features: 1, z, z_i z_j
            # z = x - center
            Z = np.array([p - center for p in pts])
            m = Z.shape[0]
            d = Z.shape[1]
            
            # Number of features: 1 + d + d(d+1)/2
            n_feat = 1 + d + d*(d+1)//2
            A = np.zeros((m, n_feat))
            
            # Column 0: 1
            A[:, 0] = 1.0
            
            # Columns 1 to d: z_i
            A[:, 1:1+d] = Z
            
            # Columns d+1 to end: z_i z_j
            idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    A[:, idx] = Z[:, i] * Z[:, j]
                    idx += 1
            
            # Solve least squares
            # A * coeff = vals
            coeff, residuals, rank, s = np.linalg.lstsq(A, np.array(vals), rcond=None)
            
            # Extract g, H, b
            b = coeff[0]
            g = coeff[1:1+d].copy()
            
            # H is symmetric
            H = np.zeros((d, d))
            idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    val = coeff[idx]
                    if i == j:
                        H[i, i] = val # Note: model has 0.5 * z^T H z, so coefficient for z_i^2 is H_ii
                        # Wait, the feature is z_i^2. The term is 0.5 * H_ii * z_i^2.
                        # So coeff[idx] = 0.5 * H_ii => H_ii = 2 * coeff[idx]
                        H[i, i] = 2 * val
                    else:
                        H[i, j] = val # Term is 0.5 * (H_ij z_i z_j + H_ji z_j z_i) = H_ij z_i z_j
                        # So coeff[idx] = H_ij
                        H[j, i] = val
                    idx += 1
            
            return b, g, H

        # Function to find trust region step
        def trust_region_step(g, H, delta):
            # Minimize 0.5 z^T H z + g^T z s.t. ||z|| <= delta
            # Check if H is PD
            try:
                # Add small regularization for stability
                H_reg = H + 1e-6 * np.eye(self.dim)
                # Unconstrained min
                z_star = -np.linalg.solve(H_reg, g)
                
                if np.linalg.norm(z_star) <= delta:
                    return z_star
                else:
                    # Solve for lambda such that ||(H+lambda I)^-1 (-g)|| = delta
                    # Use bisection
                    lo = 0.0
                    hi = 1e6
                    # Check if H+hi I is invertible
                    for _ in range(100):
                        mid = (lo + hi) / 2
                        try:
                            z_mid = -np.linalg.solve(H_reg + mid * np.eye(self.dim), g)
                            norm_mid = np.linalg.norm(z_mid)
                            if norm_mid > delta:
                                lo = mid
                            else:
                                hi = mid
                        except:
                            hi = mid
                    z_final = -np.linalg.solve(H_reg + hi * np.eye(self.dim), g)
                    return z_final
            except:
                # Fallback: steepest descent step
                norm_g = np.linalg.norm(g)
                if norm_g > 1e-12:
                    return -g / norm_g * delta
                else:
                    return np.zeros(self.dim)

        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Fit model using current points
            # We should ensure we have enough points.
            # If we have < 21, we can't fit a full quadratic reliably.
            # Let's ensure we maintain a set of points.
            # For the first few iterations, we might just use the initial 11 points + new ones.
            
            b, g, H = fit_quadratic(points, values, x_c)
            
            # Compute step
            z_step = trust_region_step(g, H, delta)
            x_new = np.clip(x_c + z_step, -5, 5)
            
            f_new = func(x_new)
            calls_made += 1
            
            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new
            
            # Update points and center
            # Strategy: Keep the best 25 points? Or just add new point and remove worst?
            # Let's keep a pool of points.
            points.append(x_new)
            values.append(f_new)
            
            # If pool is too large, remove the worst (highest f) point, but keep the center?
            # Better: Re-center around the best point found so far.
            # Let's re-center if the best point has moved significantly.
            # For simplicity, let's just keep the last 50 points.
            if len(points) > 50:
                # Remove the oldest point
                points.pop(0)
                values.pop(0)
                
            # Update center to best point found in the pool?
            # Or keep x_c as the reference for the model?
            # The model is fit relative to x_c.
            # If x_c is far from the best point, the model might be poor.
            # Let's update x_c to best_x if best_x is in the pool.
            if best_x in points:
                 # Find index
                 # This is slow. Let's just track best_idx
                 pass
                 
            # Let's track best_idx explicitly
            # Actually, let's just re-center if the step was successful.
            # If f_new < f_c, we moved to a better place.
            if f_new < f_c:
                # Successful step
                # Increase delta
                delta *= 1.5
                # Move center to new point
                x_c = x_new
                f_c = f_new
            else:
                # Unsuccessful step
                # Decrease delta
                delta *= 0.5
                # Keep center
            
            # If delta is very small, we are converged
            if delta < 1e-5:
                break
                
        return best_x, best_f
