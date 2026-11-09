import numpy as np
import time
from scipy.optimize import minimize

def construct():
    N = 200
    start_time = time.time()
    deadline = 115.0  # 5s safety margin for 120s limit

    def compute_score(centers):
        if not np.all(np.isfinite(centers)):
            return -np.inf
        if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
            return -np.inf
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)

    def soft_min_score(centers, alpha=100.0):
        """
        Compute a smooth approximation of the score using log-sum-exp for min.
        score = min( min_pair_dist, min_wall_dist )
        We maximize this, so we minimize -soft_min.
        The gradient of the soft min is well-defined.
        """
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        
        # We want min of all pairwise dists and all wall dists.
        # Let's collect all values into one vector? No, too big and memory heavy.
        # Instead, use the property: min(A, B) where A = min(pairs), B = min(walls).
        # But we need the gradient of the *global* min.
        # The soft min of a set of values V is -1/alpha * log(sum(exp(-alpha * V))).
        # Here V consists of {d_ij for i<j} U {w_i for all i}.
        
        # To compute the gradient efficiently, we compute the weights.
        # w_ij = exp(-alpha * d_ij) / Z
        # w_i = exp(-alpha * w_i) / Z
        # Z = sum(w_ij) + sum(w_i)
        
        # Gradient of soft_min wrt c_p:
        # sum_{constraints c involving p} w_c * grad_c(soft_min_value)
        # Wait, soft_min = -1/alpha log(sum(exp(-alpha V_c)))
        # d(soft_min)/d(c_p) = -1/alpha * (1/Z) * sum_{c involves p} (-alpha * exp(-alpha V_c) * dV_c/dc_p)
        # = (1/Z) * sum_{c involves p} exp(-alpha V_c) * dV_c/dc_p
        # = sum_{c involves p} w_c * dV_c/dc_p
        
        # So we just need the weighted sum of the gradients of the individual constraint values.
        
        # 1. Pairwise terms
        # V_c = d_ij = ||c_i - c_j||
        # dV_c/dc_i = (c_i - c_j) / d_ij
        # dV_c/dc_j = (c_j - c_i) / d_ij = -dV_c/dc_i
        
        # 2. Wall terms
        # V_c = w_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # This is the distance to the closest wall.
        # Gradient is the unit vector pointing away from the closest wall.
        
        # Compute exp(-alpha * d_ij)
        # To avoid underflow/overflow, we can shift, but since we only care about the min,
        # and alpha is moderate, it should be okay.
        # However, for stability, let's compute the log-sum-exp properly or just use the weights directly if they don't underflow.
        # With alpha=100 and d~0.05, exp(-5) ~ 0.006. Not underflow.
        # With d~0.1, exp(-10) ~ 4e-5.
        # With d~0.2, exp(-20) ~ 2e-9.
        # So only very close pairs contribute significantly.
        
        pair_weights_raw = np.exp(-alpha * dists)
        np.fill_diagonal(pair_weights_raw, 0.0)
        
        wall_weights_raw = np.exp(-alpha * wall_dists)
        
        Z = np.sum(pair_weights_raw) + np.sum(wall_weights_raw)
        
        if Z < 1e-100:
            # All weights are effectively 0? This shouldn't happen if distances are small enough.
            # If it does, the gradient is undefined or zero.
            return 0.0, np.zeros_like(centers)
            
        pair_weights = pair_weights_raw / Z
        wall_weights = wall_weights_raw / Z
        
        grads = np.zeros_like(centers)
        
        # Pairwise gradient contribution
        # grad_i += sum_j w_ij * (c_i - c_j) / d_ij
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vecs = diff / dists[:, :, np.newaxis]
            np.fill_diagonal(unit_vecs[:, :, 0], 0)
            np.fill_diagonal(unit_vecs[:, :, 1], 0)
            unit_vecs = np.nan_to_num(unit_vecs)
            
        weighted_vecs = pair_weights[:, :, np.newaxis] * unit_vecs
        grads += np.sum(weighted_vecs, axis=1)
        
        # Wall gradient contribution
        for idx in range(N):
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            
            min_wall_val = min(dl, dr, db, dt)
            
            gx = 0
            gy = 0
            
            if dl == min_wall_val:
                gx = 1
            elif dr == min_wall_val:
                gx = -1
            elif db == min_wall_val:
                gy = 1
            else: # dt == min_wall_val
                gy = -1
            
            # If there's a tie (corner), we should probably add the gradients.
            # e.g. if dl == dr == min, then it's a corner? No, dl and dr can't both be min unless x is such that x+0.5 = 0.5-x -> x=0.
            # If x=0, dl=0.5, dr=0.5. If y=0, db=0.5, dt=0.5.
            # If we are at a corner, e.g. x=0.5, y=0.5. dl=1, dr=0, db=1, dt=0.
            # min is 0. dr and dt are both 0.
            # So we should add the gradients for all active constraints.
            
            # Let's re-evaluate to handle ties properly.
            active_grads = []
            if dl == min_wall_val: active_grads.append((1, 0))
            if dr == min_wall_val: active_grads.append((-1, 0))
            if db == min_wall_val: active_grads.append((0, 1))
            if dt == min_wall_val: active_grads.append((0, -1))
            
            if active_grads:
                total_gx = sum(g[0] for g in active_grads)
                total_gy = sum(g[1] for g in active_grads)
                norm = np.sqrt(total_gx**2 + total_gy**2)
                if norm > 0:
                    total_gx /= norm
                    total_gy /= norm
                grads[idx, 0] += wall_weights[idx] * total_gx
                grads[idx, 1] += wall_weights[idx] * total_gy
        
        # The value of the soft min
        # soft_min = -1/alpha * log(Z) + (shift?)
        # Actually, we just need the value for tracking, but the optimizer uses the negative.
        # Let's compute the soft min value properly.
        # min(V) approx -1/alpha log(sum(exp(-alpha V)))
        # We have Z = sum(exp(-alpha V)).
        # So soft_min = -1/alpha * log(Z).
        # But this is only correct if we didn't shift.
        # Since we didn't shift, and exp can underflow, this might be inaccurate for large V.
        # However, for the purpose of the optimizer, the gradient is the most important part.
        # The value is used for comparison.
        # To be safe, let's just return the hard min for the value, but use the soft gradient.
        # This is a common trick: use smooth gradient, hard value.
        val = compute_score(centers)
        
        return val, grads

    def soft_min_neg(centers, alpha=100.0):
        val, grads = soft_min_score(centers, alpha)
        return -val, -grads

    def hex_init(N):
        r_target = 0.036
        dx = 2 * r_target
        dy = np.sqrt(3) * r_target
        
        points = []
        y = -0.5 + r_target
        row = 0
        while y <= 0.5 - r_target + 1e-6:
            x_offset = 0 if row % 2 == 0 else dx / 2
            x = -0.5 + r_target + x_offset
            while x <= 0.5 - r_target + 1e-6:
                points.append((x, y))
                x += dx
            y += dy
            row += 1
            
        points = np.array(points)
        
        if len(points) >= N:
            dists_to_center = np.linalg.norm(points, axis=1)
            sorted_indices = np.argsort(dists_to_center)
            return points[sorted_indices[:N]]
        else:
            remaining = N - len(points)
            random_points = np.random.uniform(-0.5 + r_target, 0.5 - r_target, (remaining, 2))
            return np.vstack([points, random_points])

    def grid_init(N):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    inits = [
        hex_init(N),
        grid_init(N),
        np.random.uniform(-0.4, 0.4, (N, 2))
    ]
    
    best_centers = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() - start_time > deadline:
            break
            
        # Phase 1: Gradient Ascent using BFGS on the soft-min objective
        # We minimize -soft_min_score.
        # Since soft_min_score returns (val, grads), we need a function that returns (f, g).
        # scipy.optimize.minimize can take jac.
        
        def objective(x):
            centers = x.reshape(N, 2)
            val, grads = soft_min_score(centers, alpha=100.0)
            return -val, -grads.reshape(-1)
        
        x0 = init.flatten()
        
        try:
            result = minimize(objective, x0, method='BFGS', jac=True,
                             options={'maxiter': 500, 'gtol': 1e-8})
            
            if time.time() - start_time < deadline:
                new_centers = result.x.reshape(N, 2)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_score = compute_score(new_centers)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
        except Exception:
            pass

        # Phase 2: Local refinement with Nelder-Mead on the hard score
        # This helps to fine-tune the positions to maximize the exact min distance.
        if best_centers is not None and time.time() - start_time < deadline - 10:
            def hard_objective(x):
                centers = x.reshape(N, 2)
                if np.any(centers < -0.5) or np.any(centers > 0.5):
                    return 100.0
                return -compute_score(centers)
            
            x0 = best_centers.flatten()
            try:
                result2 = minimize(hard_objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if time.time() - start_time < deadline:
                    new_centers = result2.x.reshape(N, 2)
                    new_centers = np.clip(new_centers, -0.5, 0.5)
                    new_score = compute_score(new_centers)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
            except Exception:
                pass

    # Final fallback
    if best_centers is None:
        best_centers = grid_init(N)
        best_score = compute_score(best_centers)
        
    return best_centers
