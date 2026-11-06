import numpy as np
import time
import scipy.optimize as opt

N = 200
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # 1. Generate a good initial hexagonal lattice configuration
    # Hexagonal packing is the densest known for infinite planes.
    # For 200 points in a unit square, we estimate the spacing.
    # Area per point ~ 1/200 = 0.005.
    # Hex cell area = d^2 * sqrt(3)/2.
    # d^2 * 0.866 ~ 0.005 => d^2 ~ 0.00577 => d ~ 0.076.
    
    d = 0.076
    points_list = []
    dy = d * np.sqrt(3) / 2
    
    # Generate points on a hexagonal grid covering the square with some margin
    n_rows = int(np.ceil(1.0 / dy)) + 2
    for j in range(-n_rows, n_rows + 1):
        y = j * dy
        if y < -0.55 or y > 0.55:
            continue
        x_offset = (j % 2) * d / 2
        n_cols = int(np.ceil(1.0 / d)) + 2
        for i in range(-n_cols, n_cols + 1):
            x = i * d + x_offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points_list.append([x, y])
    
    points = np.array(points_list)
    
    # Adjust to exactly N points
    if len(points) > N:
        # Randomly sample N points from the lattice to break symmetry
        # and provide a diverse starting point for the optimizer.
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Add random points if we don't have enough from the grid
        remaining = N - len(points)
        new_pts = np.random.uniform(-0.45, 0.45, size=(remaining, 2))
        points = np.vstack([points, new_pts])
        
    # Ensure points are strictly inside initially to avoid immediate large penalties
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    def get_radius(pts):
        """Calculate the actual maximum radius for a given set of points."""
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_c = np.min(dist) / 2.0
        
        # Boundary distances
        dx = np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0])
        dy = np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
        min_b = np.min(np.minimum(dx, dy))
        
        return min(min_c, min_b)
    
    current_r = get_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Vectorized Objective and Gradient for L-BFGS-B
    # We minimize a penalty function: sum of squared violations.
    # Violations are:
    # 1. Boundary: r_target - dist_to_wall > 0
    # 2. Pair overlap: 2*r_target - dist_ij > 0
    
    def objective_and_grad(p_flat, r_target):
        p = p_flat.reshape(N, 2)
        E = 0.0
        grad = np.zeros_like(p)
        
        # 1. Boundary Violations (Vectorized)
        # Distances to walls
        d_left = p[:, 0] + 0.5
        d_right = 0.5 - p[:, 0]
        d_bottom = p[:, 1] + 0.5
        d_top = 0.5 - p[:, 1]
        
        d_bounds = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
        min_b = np.min(d_bounds, axis=1)
        
        viol_b = np.maximum(0.0, r_target - min_b)
        E += np.sum(viol_b**2)
        
        # Gradient of penalty w.r.t. position
        # d(viol_b^2)/dp = 2 * viol_b * d(viol_b)/dp
        # d(viol_b)/dp is -1 in the direction of the closest wall, 0 otherwise.
        min_idx = np.argmin(d_bounds, axis=1)
        
        gx = np.zeros(N)
        gy = np.zeros(N)
        
        # If closest wall is Left (idx 0), moving left (dx<0) increases distance? No.
        # d_left = x + 0.5. d(d_left)/dx = 1.
        # viol = max(0, r - d_left). d(viol)/dx = -d(viol)/d(d_left) = -1 if active.
        # So grad_x += -2 * viol * (-1) = 2 * viol? 
        # Let's check: if x is too close to left wall, we want to push it right (+x).
        # Penalty increases if x decreases. So gradient should point in +x direction.
        # dE/dx = 2 * viol * (-1) * d(d_left)/dx ? No.
        # E = viol^2. dE/dx = 2 * viol * d(viol)/dx.
        # viol = r - (x+0.5). d(viol)/dx = -1.
        # dE/dx = -2 * viol. This points in -x direction (towards wall). That's wrong for minimization.
        # Wait, minimization moves AGAINST the gradient.
        # If gradient is negative, update x = x - lr * grad => x increases. Correct.
        
        gx[min_idx == 0] = 1.0   # Left wall active: grad_x component from this term
        gx[min_idx == 1] = -1.0  # Right wall active
        gy[min_idx == 2] = 1.0   # Bottom wall active
        gy[min_idx == 3] = -1.0  # Top wall active
        
        # Note: The gradient of the penalty term 2*viol*dir is:
        # dE/dx_i = 2 * viol_i * (d(viol_i)/dx_i)
        # If left wall: viol = r - (x+0.5). d(viol)/dx = -1.
        # So term is -2*viol.
        # We defined gx=1 for left. So we add -2*viol * 1? No.
        # Let's just compute:
        # dE/dx = -2 * viol_b * (sign of x direction to wall)
        # Left wall: move right (+x) reduces viol. Grad should be negative (so x increases).
        # d(viol)/dx = -1. dE/dx = 2*viol*(-1) = -2*viol.
        # So if min_idx==0 (left), grad_x contribution is -2*viol.
        # My gx vector: I set gx=1 for left.
        # So I should do: grad_x += -2 * viol_b * gx?
        # If gx=1, grad_x += -2*viol. Correct.
        
        grad[:, 0] += -2.0 * viol_b * gx
        grad[:, 1] += -2.0 * viol_b * gy
        
        # 2. Pairwise Violations (Vectorized)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Violation if dist < 2*r_target
        viol_pairs = np.maximum(0.0, 2.0 * r_target - dist)
        
        # Only sum upper triangle to avoid double counting
        i_idx, j_idx = np.triu_indices(N, k=1)
        v = viol_pairs[i_idx, j_idx]
        d = dist[i_idx, j_idx]
        
        E += np.sum(v**2)
        
        mask = v > 0
        if np.any(mask):
            i_m = i_idx[mask]
            j_m = j_idx[mask]
            v_m = v[mask]
            d_m = d[mask]
            
            # Safe division
            d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
            
            # Vector from j to i
            vec = p[i_m] - p[j_m]
            unit = vec / d_safe[:, np.newaxis]
            
            # Gradient of v^2 w.r.t p_i:
            # v = 2r - dist.
            # d(v)/dp_i = - (p_i - p_j) / dist = -unit.
            # d(v^2)/dp_i = 2 * v * (-unit) = -2 * v * unit.
            # We want to minimize, so move p_i in direction of -grad.
            # If p_i is too close to p_j, unit points from j to i.
            # -2*v*unit points from i to j. This pushes them together?
            # Wait. If they overlap, we want to push them apart.
            # p_i should move away from p_j. Direction is unit (from j to i).
            # So gradient should be -unit?
            # Let's re-evaluate.
            # E = v^2.
            # dE/dp_i = 2 v (dv/dp_i).
            # dv/dp_i = d(2r - ||p_i - p_j||)/dp_i = - (p_i - p_j)/||p_i - p_j|| = -unit.
            # So dE/dp_i = -2 v unit.
            # Update: p_i = p_i - lr * grad = p_i - lr * (-2 v unit) = p_i + 2 lr v unit.
            # This moves p_i in direction of unit (away from p_j). Correct.
            
            # For p_j:
            # dv/dp_j = d(2r - ||p_i - p_j||)/dp_j = - (-(p_i - p_j))/||...|| = unit.
            # dE/dp_j = 2 v unit.
            # Update: p_j = p_j - lr * (2 v unit). Moves p_j opposite to unit (away from p_i). Correct.
            
            np.add.at(grad, i_m, -2.0 * v_m[:, np.newaxis] * unit)
            np.add.at(grad, j_m,  2.0 * v_m[:, np.newaxis] * unit)
            
        return E, grad.flatten()

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimization Loop: Increase target radius gradually
    # Start from current radius, incrementally push it higher.
    # This helps escape local minima.
    
    r = current_r
    # Step size for radius increase
    # We want to reach ~0.0366.
    # If current is ~0.03, we need ~0.006 increase.
    # Let's use a dynamic step or fixed small steps.
    
    # Strategy: 
    # 1. Optimize at current r.
    # 2. Increase r by small delta.
    # 3. Optimize at new r.
    # 4. If success, update best_r and continue.
    # 5. If fail (objective doesn't decrease or time runs out), break.
    
    # To be efficient, we can try larger steps initially if the config is good.
    # But small steps are safer for convergence.
    
    # Let's try a sequence of target radii.
    # We don't know the exact optimal, but 0.0366 is the benchmark.
    # We will try to climb towards it.
    
    # Initial optimization at current radius to settle the configuration
    if time.time() < deadline - 20:
        p_flat = best_points.flatten()
        try:
            res = opt.minimize(
                objective_and_grad, 
                p_flat, 
                args=(r,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 100, 'gtol': 1e-5, 'ftol': 1e-7}
            )
            new_points = res.x.reshape(N, 2)
            new_actual_r = get_radius(new_points)
            if new_actual_r > best_r:
                best_r = new_actual_r
                best_points = new_points.copy()
                r = new_actual_r
        except Exception:
            pass
            
    # Now climb the radius
    # We will try to increase r by small amounts until we can't anymore or time runs out.
    # A good heuristic is to increase by a fraction of the current radius or a fixed small amount.
    # Since we are close to the optimum, fixed small amounts are better.
    
    step_r = 0.0005  # 0.05% of 1.0, or ~1.4% of 0.035
    
    while time.time() < deadline - 10:
        new_r = r + step_r
        
        # Don't try to exceed the known best by too much in one go if we are already high
        # But we don't know the "known best" strictly, we just know 0.0366 is a target.
        
        p_flat = best_points.flatten()
        
        try:
            # Use more iterations for higher radii as the problem becomes tighter
            max_iter = 200 if new_r < 0.035 else 500
            gtol = 1e-6 if new_r < 0.035 else 1e-8
            
            res = opt.minimize(
                objective_and_grad, 
                p_flat, 
                args=(new_r,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-8}
            )
            
            new_points = res.x.reshape(N, 2)
            new_actual_r = get_radius(new_points)
            
            if new_actual_r > best_r:
                best_r = new_actual_r
                best_points = new_points.copy()
                r = new_actual_r
                # If we made progress, we can try again.
                # Maybe increase step size slightly if we are far from 0.0366?
                # For now, keep it constant for stability.
            else:
                # If we failed to improve, try a smaller step?
                # Or break?
                # If we can't fit a slightly larger radius, we are likely at a local optimum
                # for this configuration. 
                # We could try a random perturbation, but time is limited.
                # Let's try a smaller step once.
                step_r *= 0.5
                if step_r < 1e-5:
                    break
        except Exception:
            break

    # Final Polish: Try to squeeze out a bit more with very tight tolerances
    if time.time() < deadline - 5:
        p_flat = best_points.flatten()
        try:
            # Use the current best radius as target, but with very strict convergence
            # This often finds small improvements by adjusting points that were "almost" optimal.
            res = opt.minimize(
                objective_and_grad, 
                p_flat, 
                args=(best_r,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            new_points = res.x.reshape(N, 2)
            new_actual_r = get_radius(new_points)
            if new_actual_r > best_r:
                best_r = new_actual_r
                best_points = new_points.copy()
        except Exception:
            pass

    # One last check: try to increase radius slightly from the polished state
    if time.time() < deadline - 2:
        try:
            new_r = best_r + 0.0001
            p_flat = best_points.flatten()
            res = opt.minimize(
                objective_and_grad, 
                p_flat, 
                args=(new_r,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            new_points = res.x.reshape(N, 2)
            new_actual_r = get_radius(new_points)
            if new_actual_r > best_r:
                best_r = new_actual_r
                best_points = new_points.copy()
        except Exception:
            pass

    # Ensure final points are strictly inside (safety net)
    best_points[:, 0] = np.clip(best_points[:, 0], -0.49999, 0.49999)
    best_points[:, 1] = np.clip(best_points[:, 1], -0.49999, 0.49999)
    
    return best_points

# Test
if __name__ == "__main__":
    pts = construct()
    # Print score for local testing
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_c = np.min(dist) / 2.0
    dx = np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0])
    dy = np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    min_b = np.min(np.minimum(dx, dy))
    print(f"Radius: {min(min_c, min_b)}")
