import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Ensure finite and normalized
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    def compute_energy(pts):
        """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
        dist_sq = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2 * pts @ pts.T
        dist_sq = np.clip(dist_sq, 1e-12, None)
        dist = np.sqrt(dist_sq)

        # Extract upper triangle
        i_upper, j_upper = np.triu_indices(N, k=1)
        d_pairs = dist[i_upper, j_upper]
        # Energy is sum of log(1/d) = -sum of log(d)
        return -np.sum(np.log(d_pairs))

    def compute_gradient(pts):
        """
        Compute gradient of E = -sum_{i<j} log(d_ij) w.r.t. each point.
        For point k: grad_k = sum_{j != k} (x_k - x_j) / d_kj^2
        """
        dist_sq = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2 * pts @ pts.T
        dist_sq = np.clip(dist_sq, 1e-12, None)

        # inv_d2[i,j] = 1/d_ij^2, with diagonal 0
        inv_d2 = 1.0 / dist_sq
        np.fill_diagonal(inv_d2, 0.0)

        # grad_i = sum_j (x_i - x_j) / d_ij^2
        # = x_i * sum_j inv_d2[i,j] - sum_j inv_d2[i,j] * x_j
        row_sums = np.sum(inv_d2, axis=1)
        grad = pts * row_sums[:, None] - inv_d2 @ pts

        return grad

    def project_to_tangent(pts, grad):
        """Project gradient onto tangent space of sphere at each point."""
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - dot * pts
        return grad_tangent

    def objective_and_grad(x_flat):
        """Objective and gradient for flattened point array with tangent projection."""
        pts = x_flat.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms

        energy = compute_energy(pts)

        # Compute gradient on sphere
        grad = compute_gradient(pts)
        # Project gradient to tangent space
        grad_tangent = project_to_tangent(pts, grad)

        return energy, grad_tangent.flatten()

    # Try to use scipy L-BFGS
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except ImportError:
        has_scipy = False

    best_energy = compute_energy(points)
    best_points = points.copy()

    if has_scipy:
        # Multiple restarts with different initializations
        num_restarts = 6
        best_energy_so_far = best_energy
        best_points_so_far = best_points.copy()

        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break

            if restart == 0:
                # Use the initial Fibonacci spiral
                x0 = points.flatten()
            elif restart == 1:
                # Small random perturbation of best
                noise = np.random.randn(N, 3) * 0.05
                x0 = (best_points_so_far + noise).flatten()
            elif restart == 2:
                # Larger perturbation
                noise = np.random.randn(N, 3) * 0.15
                x0 = (best_points_so_far + noise).flatten()
            elif restart == 3:
                # Ring-based initialization
                num_rings = 10
                pts_ring = []
                for ring in range(num_rings):
                    z_val = -1.0 + 2.0 * ring / (num_rings - 1)
                    r_val = np.sqrt(max(1.0 - z_val**2, 0.0))
                    if r_val < 0.01:
                        r_val = 0.01
                    num_per_ring = N // num_rings
                    for j in range(num_per_ring):
                        angle = 2.0 * np.pi * j / num_per_ring + ring * 0.3
                        x_val = r_val * np.cos(angle)
                        y_val = r_val * np.sin(angle)
                        pts_ring.append([x_val, y_val, z_val])
                # Pad if needed
                while len(pts_ring) < N:
                    pts_ring.append([np.random.randn(3)])
                pts_ring = np.array(pts_ring[:N])
                pts_ring /= np.linalg.norm(pts_ring, axis=1, keepdims=True)
                x0 = pts_ring.flatten()
            elif restart == 4:
                # Different Fibonacci with offset
                i2 = np.arange(N) + 0.5
                z2 = 1.0 - 2.0 * i2 / N
                phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + 0.3)
                r2 = np.sqrt(1.0 - z2 * z2)
                pts2 = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
                pts2 /= np.linalg.norm(pts2, axis=1, keepdims=True)
                x0 = pts2.flatten()
            else:
                # Random initial points
                x0_arr = np.random.randn(N, 3)
                x0_arr /= np.linalg.norm(x0_arr, axis=1, keepdims=True)
                x0 = x0_arr.flatten()

            # Use L-BFGS-B
            try:
                result = minimize(
                    lambda x: objective_and_grad(x)[0],
                    x0,
                    jac=lambda x: objective_and_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )

                new_pts = result.x.reshape(N, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_energy = compute_energy(new_pts)

                if new_energy < best_energy_so_far:
                    best_energy_so_far = new_energy
                    best_points_so_far = new_pts.copy()

            except Exception:
                pass

        points = best_points_so_far.copy()
        best_energy = best_energy_so_far

    # Final polish: gradient descent with adaptive step size in tangent space
    current_energy = compute_energy(points)
    step_size = 0.01
    min_step_size = 1e-10
    max_step_size = 0.1
    no_improve_count = 0
    max_iterations = 3000

    for it in range(max_iterations):
        if time.time() > deadline:
            break

        grad = compute_gradient(points)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < 1e-12:
            break

        # Project gradient to tangent space
        grad_tangent = project_to_tangent(points, grad)

        # Proposed update
        candidate_points = points - step_size * grad_tangent
        candidate_points /= np.linalg.norm(candidate_points, axis=1, keepdims=True)

        candidate_energy = compute_energy(candidate_points)

        if candidate_energy < current_energy:
            points = candidate_points
            current_energy = candidate_energy
            no_improve_count = 0
            step_size = min(step_size * 1.05, max_step_size)
        else:
            step_size *= 0.7
            no_improve_count += 1
            if no_improve_count > 50:
                if step_size < min_step_size:
                    break
                step_size = 0.01
                no_improve_count = 0

    # Final normalization
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    # Check for validity
    if not np.all(np.isfinite(points)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    return points

# EVOLVE-BLOCK-END
