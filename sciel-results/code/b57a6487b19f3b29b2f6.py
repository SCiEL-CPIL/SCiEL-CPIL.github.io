"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def remaining_time():
        return max(0, deadline - time.time())

    def compute_radius(pts):
        """Compute the actual radius: min of all pair distances / 2 and wall distances."""
        if np.any(pts < -0.5) or np.any(pts > 0.5) or not np.all(np.isfinite(pts)):
            return 0.0

        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0

        # Wall distances
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)

        return min(min_pair_dist, min_wall_dist)

    def hex_grid_init(d):
        """Generate a hexagonal grid of points with spacing d."""
        pts = []
        # Number of rows to fit in height
        row_height = d * np.sqrt(3) / 2
        n_rows = int((1.0 - d) / row_height) + 1
        # Adjust to fit
        while n_rows > 1:
            total_height = (n_rows - 1) * row_height
            if total_height + d <= 1.0:
                break
            n_rows -= 1

        y_start = -0.5 + d/2 + total_height/2 if n_rows > 1 else 0.0
        if n_rows == 1:
            y_start = 0.0

        y_positions = np.linspace(-0.5 + d/2, 0.5 - d/2, n_rows) if n_rows > 1 else np.array([0.0])

        for i, y_i in enumerate(y_positions):
            offset = 0 if i % 2 == 0 else d/2
            # x positions
            x_start = -0.5 + d/2 - offset
            n_x = int((1.0 - d + 1e-9) / d) + 1
            x_positions = x_start + np.arange(n_x) * d
            # Filter to be within bounds
            x_positions = x_positions[(x_positions >= -0.5 + d/2 - 1e-9) & (x_positions <= 0.5 - d/2 + 1e-9)]

            for x_i in x_positions:
                pts.append([x_i, y_i])

        pts = np.array(pts)
        return pts

    def initial_hex_grid():
        """Generate a good initial hexagonal grid."""
        # Try several spacings and pick the one that gives closest to N points
        best_pts = None
        best_count_diff = N + 1

        for d in np.arange(0.065, 0.080, 0.001):
            pts = hex_grid_init(d)
            count_diff = abs(len(pts) - N)
            if count_diff < best_count_diff:
                best_count_diff = count_diff
                best_pts = pts
                best_d = d

        pts = best_pts.copy()

        if len(pts) < N:
            # Add random points
            while len(pts) < N:
                p = np.random.uniform(-0.4, 0.4, 2)
                pts = np.vstack([pts, [p]])
        elif len(pts) > N:
            pts = pts[:N]

        return pts

    def optimize_repulsion(pts, time_limit):
        """
        Optimize points using repulsive forces with adaptive parameters.
        """
        current_time = time.time()
        v = np.zeros_like(pts)
        best_pts = pts.copy()
        best_radius = compute_radius(pts)

        lr = 0.0005
        repulsion_strength = 0.002
        wall_strength = 0.05
        damping = 0.85
        eps = 1e-8

        no_improve_count = 0
        last_best_radius = best_radius

        steps = 0
        while time.time() - current_time < time_limit:
            # Compute forces
            forces = np.zeros_like(pts)

            # Pairwise repulsion
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)

            # F = (c_i - c_j) / (d_ij^2 + eps)
            inv_d2 = 1.0 / (dists**2 + eps)
            np.fill_diagonal(inv_d2, 0)

            forces += repulsion_strength * np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)

            # Wall repulsion - push towards center when close to walls
            dist_to_wall = 0.5 - np.abs(pts)
            # Use a smooth potential: V = 1/(dist + eps), F = 1/(dist + eps)^2
            wall_forces = wall_strength / (dist_to_wall + eps)**2
            # Direction: towards center, i.e., -sign(pts)
            forces += wall_forces * (-np.sign(pts))

            # Update
            v = damping * v + (1 - damping) * forces
            pts += lr * v

            # Clip to square (hard constraint)
            pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)

            # Check radius periodically
            if steps % 10 == 0:
                r = compute_radius(pts)
                if r > best_radius:
                    best_radius = r
                    best_pts = pts.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Adaptive learning rate
                if no_improve_count > 50:
                    lr *= 0.9
                    no_improve_count = 0
                    # Occasionally random kick
                    if np.random.random() < 0.1:
                        pts += np.random.randn(*pts.shape) * 0.001
                        pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
                        pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)

            steps += 1

        return best_pts

    def soft_min_polish(pts, time_limit):
        """
        High-precision polish using soft-min gradient descent on the minimum distance.
        The objective is to maximize min(min_pair_dist/2, min_wall_dist).
        We use a log-sum-exp soft-min to get a differentiable surrogate.
        """
        current_time = time.time()
        best_pts = pts.copy()
        best_radius = compute_radius(pts)

        # Increase beta over time for sharper minimum
        beta_start = 20.0
        beta_end = 200.0
        lr = 0.002
        mu = 0.9
        v = np.zeros_like(pts)

        def soft_min_objective(pts):
            """
            Returns the soft-min value and gradient.
            We want to maximize min over all constraints.
            Constraints: pair distances / 2 and wall distances.
            """
            # Pair distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_constraints = dists / 2.0

            # Wall distances
            wall_dists = 0.5 - np.abs(pts)
            wall_constraints = np.min(wall_dists, axis=1)

            # Collect all constraint values
            all_vals = np.concatenate([pair_constraints.ravel(), wall_constraints])
            all_vals = np.where(np.isfinite(all_vals), all_vals, np.inf)

            # Soft min: m - (1/beta) * log(sum(exp(-beta*(x - m))))
            m = np.min(all_vals)
            exp_vals = np.exp(-beta * (all_vals - m))
            sum_exp = np.sum(exp_vals)
            soft_min_val = m - (1.0 / beta) * np.log(sum_exp + 1e-30)
            weights = exp_vals / (sum_exp + 1e-30)

            # Gradient
            grad = np.zeros_like(pts)

            # Wall gradients
            wall_weights = weights[N*N:]
            abs_x = np.abs(pts[:, 0])
            abs_y = np.abs(pts[:, 1])
            dist_x = 0.5 - abs_x
            dist_y = 0.5 - abs_y

            # For each point, the wall constraint is min(dist_x, dist_y)
            # d(min(dist_x, dist_y))/dx: if dist_x < dist_y, d(dist_x)/dx = -sign(x)
            # if dist_y < dist_x, d(dist_y)/dx = 0
            # if equal, average

            sign_x = np.sign(pts[:, 0])
            sign_y = np.sign(pts[:, 1])

            # Gradient contribution from wall constraints
            # d wall_constraint_i / d pts_i[0]
            mask_x_less = dist_x < dist_y
            mask_y_less = dist_y < dist_x
            mask_eq = np.isclose(dist_x, dist_y, atol=1e-10)

            grad[:, 0] += np.where(mask_x_less, wall_weights, 0) * (-sign_x)
            grad[:, 0] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_x)
            grad[:, 1] += np.where(mask_y_less, wall_weights, 0) * (-sign_y)
            grad[:, 1] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_y)

            # Pair gradients
            pair_weights = weights[:N*N].reshape(N, N)
            # d (d_ij/2) / d c_i = (c_i - c_j) / (2 * d_ij)
            with np.errstate(divide='ignore', invalid='ignore'):
                coeff = 0.5 * pair_weights / dists
            coeff = np.where(np.eye(N, dtype=bool), 0, coeff)
            coeff = np.where(np.isfinite(coeff), coeff, 0)

            grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            grad += grad_pair

            return soft_min_val, grad

        steps = 0
        total_steps_est = int(time_limit / 0.05)  # rough estimate
        while time.time() - current_time < time_limit:
            # Anneal beta
            progress = min(1.0, steps / max(1, total_steps_est))
            beta = beta_start + (beta_end - beta_start) * progress

            val, grad = soft_min_objective(pts)

            v = mu * v + (1 - mu) * grad
            pts += lr * v

            # Clip to square
            pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)

            # Check radius periodically
            if steps % 5 == 0:
                r = compute_radius(pts)
                if r > best_radius:
                    best_radius = r
                    best_pts = pts.copy()

            steps += 1
            # Decay learning rate
            lr *= 0.9995

        # Final check
        r = compute_radius(pts)
        if r > best_radius:
            best_radius = r
            best_pts = pts.copy()

        return best_pts

    # Main logic
    # 1. Initialize with hex grid
    pts = initial_hex_grid()
    best_pts = pts.copy()
    best_radius = compute_radius(pts)

    # 2. Global Repulsion Optimization
    t1 = remaining_time()
    if t1 > 30:
        pts = optimize_repulsion(pts, t1 - 15)
        r = compute_radius(pts)
        if r > best_radius:
            best_radius = r
            best_pts = pts.copy()

    # 3. Soft-min Polish
    t2 = remaining_time()
    if t2 > 15:
        pts = soft_min_polish(pts, t2 - 5)
        r = compute_radius(pts)
        if r > best_radius:
            best_radius = r
            best_pts = pts.copy()

    # Final validation
    r_final = compute_radius(best_pts)
    if r_final <= 0 or not np.all(np.isfinite(best_pts)):
        # Fallback to initial grid
        return initial_hex_grid()

    return best_pts

# EVOLVE-BLOCK-END
