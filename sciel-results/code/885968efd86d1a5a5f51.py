"""
Fifty equal circles in a unit circle - hybrid contact-graph optimization.
Combines soft-min gradient descent (current) with Newton-Raphson contact refinement (inspiration).
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Pre-compute pair indices
    iu = np.triu_indices(N, 1)
    num_pairs = len(iu[0])

    def _score_raw(centers):
        """Compute the actual score (minimum radius) for a set of centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0

        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms >= 1.0):
            return 0.0

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)

        min_pair_dist = np.min(dists)
        min_edge_dist = np.min(1.0 - norms)

        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(r, 0.0)

    def _objective(centers, beta=300.0):
        """Smooth approximation of negative score using log-sum-exp."""
        centers = centers.reshape(-1, 2)

        norms = np.sqrt(np.sum(centers**2, axis=1))
        edge_constraints = 1.0 - norms

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        pair_constraints = dists[iu] / 2.0

        all_constraints = np.concatenate([edge_constraints, pair_constraints])

        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / beta

        return -soft_min_val

    def _gradient(centers, beta=300.0):
        """Compute gradient of the objective - vectorized."""
        centers = centers.reshape(-1, 2)
        grad = np.zeros_like(centers)

        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-15
        edge_constraints = 1.0 - norms

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-15
        pair_constraints = dists[iu] / 2.0

        all_constraints = np.concatenate([edge_constraints, pair_constraints])

        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp

        # Edge gradient (vectorized)
        edge_grad = -centers / norms[:, np.newaxis]
        grad += -weights[:N, np.newaxis] * beta * edge_grad

        # Pair gradients (vectorized using scatter)
        i_idx = iu[0]
        j_idx = iu[1]
        w_pair = weights[N:N+num_pairs]

        diff_pairs = centers[i_idx] - centers[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]

        np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * beta * dir_vecs)
        np.add.at(grad, j_idx, w_pair[:, np.newaxis] * beta * dir_vecs)

        return grad.reshape(-1)

    def _newton_refine(centers, r_target, max_iter=200, tol=1e-10):
        """
        Newton-Raphson refinement assuming contact graph.
        For each contact pair, enforce |c_i - c_j| = 2*r
        For boundary contacts, enforce |c_i| = 1 - r
        """
        centers = centers.copy()
        r = r_target

        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            # Compute all distances
            norms = np.sqrt(np.sum(centers**2, axis=1))
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)

            # Identify active constraints (near contact)
            # Pair contacts: dist ≈ 2r
            contact_threshold = 2 * r * 1.005
            pair_contact_mask = dists < contact_threshold
            np.fill_diagonal(pair_contact_mask, False)

            # Boundary contacts: norm ≈ 1-r
            boundary_contact_mask = norms > (1 - r) * 0.995

            # Build constraint system
            # For each contact pair (i,j): (c_i - c_j) · (c_i - c_j) = (2r)^2
            # For each boundary contact (i): c_i · c_i = (1-r)^2

            # Get contact pairs
            contact_i, contact_j = np.where(np.triu(pair_contact_mask, 1))
            n_pair_contacts = len(contact_i)
            n_boundary_contacts = np.sum(boundary_contact_mask)
            n_constraints = n_pair_contacts + n_boundary_contacts

            if n_constraints == 0:
                break

            # Build Jacobian and residual
            # Variables: 2*N (x_0, y_0, x_1, y_1, ...)
            # Residuals: for pair (i,j): |c_i - c_j|^2 - (2r)^2
            #            for boundary i: |c_i|^2 - (1-r)^2

            # Compute residual
            residual = np.zeros(n_constraints)

            # Pair residuals
            if n_pair_contacts > 0:
                ci = centers[contact_i]
                cj = centers[contact_j]
                pair_diff = ci - cj
                pair_dists_sq = np.sum(pair_diff**2, axis=1)
                residual[:n_pair_contacts] = pair_dists_sq - (2 * r)**2

            # Boundary residuals
            if n_boundary_contacts > 0:
                boundary_indices = np.where(boundary_contact_mask)[0]
                boundary_norms_sq = norms[boundary_indices]**2
                residual[n_pair_contacts:] = boundary_norms_sq - (1 - r)**2

            # Check convergence
            if np.max(np.abs(residual)) < tol:
                break

            # Build Jacobian (sparse-like, but we'll use dense for simplicity with N=50)
            # J has shape (n_constraints, 2*N)
            J = np.zeros((n_constraints, 2 * N))

            # Pair constraint (i,j): f = |c_i - c_j|^2 - (2r)^2
            # df/d(c_i) = 2*(c_i - c_j)
            # df/d(c_j) = -2*(c_i - c_j)
            if n_pair_contacts > 0:
                for k in range(n_pair_contacts):
                    i = contact_i[k]
                    j = contact_j[k]
                    d = centers[i] - centers[j]
                    J[k, 2*i] = 2 * d[0]
                    J[k, 2*i+1] = 2 * d[1]
                    J[k, 2*j] = -2 * d[0]
                    J[k, 2*j+1] = -2 * d[1]

            # Boundary constraint (i): f = |c_i|^2 - (1-r)^2
            # df/d(c_i) = 2*c_i
            if n_boundary_contacts > 0:
                for k in range(n_boundary_contacts):
                    i = boundary_indices[k]
                    J[n_pair_contacts + k, 2*i] = 2 * centers[i][0]
                    J[n_pair_contacts + k, 2*i+1] = 2 * centers[i][1]

            # Solve J * delta = -residual using least squares
            try:
                delta, residuals_ls, rank, sv = np.linalg.lstsq(J, -residual, rcond=None)
            except np.linalg.LinAlgError:
                break

            # Damping to ensure stability
            step_norm = np.linalg.norm(delta)
            if step_norm > 0.05:
                delta *= 0.05 / step_norm

            # Update
            new_centers = centers.copy()
            new_centers[:, 0] += delta[0::2]
            new_centers[:, 1] += delta[1::2]

            # Check if this improves or maintains validity
            new_norms = np.sqrt(np.sum(new_centers**2, axis=1))
            if np.any(new_norms >= 1.0):
                # Reject if any circle goes outside
                break

            # Check pair distances
            new_diff = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
            new_dists = np.sqrt(np.sum(new_diff**2, axis=2))
            np.fill_diagonal(new_dists, np.inf)
            if np.min(new_dists) < 2 * r * 0.999:
                # Reject if circles overlap
                break

            centers = new_centers

        return centers

    def _soft_min_optimize(centers, beta=300.0, max_iter=5000):
        """
        Optimize using L-BFGS-B with soft-min objective.
        Returns optimized centers.
        """
        bounds = [(-1.0, 1.0)] * (2 * N)

        def obj_flat(x):
            return _objective(x, beta=beta)

        def jac_flat(x):
            return _gradient(x, beta=beta)

        result = minimize(
            obj_flat,
            centers.flatten(),
            jac=jac_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 2, 'ftol': 1e-12, 'gtol': 1e-10}
        )

        return result.x.reshape(N, 2)

    def _hexagonal_seed():
        """Generate a hexagonal lattice packing as a seed."""
        spacing = 0.2
        points_list = []
        for j in range(-5, 6):
            for i in range(-5, 6):
                x = i * spacing + (0.5 * spacing if j % 2 else 0)
                y = j * spacing * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points_list.append([x, y])

        points = np.array(points_list)
        # Sort by distance from center to prefer central points
        dists = np.linalg.norm(points, axis=1)
        points = points[np.argsort(dists)]

        # Select N points with largest minimum spacing
        if len(points) >= N:
            # Greedy selection: pick point farthest from all selected
            selected = [points[np.argmin(dists)]]
            while len(selected) < N:
                sel = np.array(selected)
                min_dists = np.min(np.linalg.norm(points[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
                selected.append(points[np.argmax(min_dists)])
            return np.array(selected)
        return points

    def _random_seed():
        """Generate a random valid packing."""
        for _ in range(10000):
            points = np.random.uniform(-1, 1, size=(N, 2))
            norms = np.linalg.norm(points, axis=1)
            max_norm = norms.max()
            if max_norm > 0:
                points *= (0.95 / max_norm)
            r = _score_raw(points)
            if r > 0.1:
                break
        return points

    def _local_refinement(points, n_steps=2000, step_size=0.005):
        """Fast coordinate descent local refinement."""
        points = points.copy()
        current_r = _score_raw(points)
        for step in range(n_steps):
            if time.time() > deadline - 5.0:
                break
            i = random.randint(0, N - 1)
            best_new_r = current_r
            best_new_point = points[i].copy()
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_point = points[i].copy()
                    new_point[0] += dx
                    new_point[1] += dy
                    if np.linalg.norm(new_point) >= 1.0:
                        continue
                    points[i] = new_point
                    new_r = _score_raw(points)
                    if new_r > best_new_r:
                        best_new_r = new_r
                        best_new_point = new_point.copy()
                    else:
                        points[i] = points[i] - np.array([dx, dy])
            if best_new_r > current_r:
                points[i] = best_new_point
                current_r = best_new_r
        return points

    # Main search portfolio
    best_points = None
    best_radius = 0.0

    # Start with hexagonal seed
    seed_hex = _hexagonal_seed()
    r_hex = _score_raw(seed_hex)
    best_points = seed_hex.copy()
    best_radius = r_hex

    # Generate multiple random seeds
    n_seeds = 12
    for i in range(n_seeds):
        if time.time() > deadline - 30.0:
            break
        seed = _random_seed()
        r = _score_raw(seed)
        if r > best_radius:
            best_points = seed.copy()
            best_radius = r

    # Try optimizing from several seeds with multiple beta values
    seeds_to_optimize = [best_points.copy()]
    seeds_to_optimize.append(seed_hex.copy())
    for i in range(6):
        if time.time() > deadline - 50.0:
            break
        s = _random_seed()
        seeds_to_optimize.append(s)

    beta_values = [300.0, 500.0, 800.0]
    
    for seed in seeds_to_optimize:
        if time.time() > deadline - 20.0:
            break
        for beta in beta_values:
            if time.time() > deadline - 15.0:
                break
            try:
                # Soft-min optimization
                optimized = _soft_min_optimize(seed, beta=beta, max_iter=2000)
                r = _score_raw(optimized)
                if r > best_radius:
                    best_points = optimized.copy()
                    best_radius = r
            except Exception:
                pass

    # Newton-Raphson refinement on best solution
    if time.time() < deadline - 10.0:
        try:
            refined = _newton_refine(best_points, best_radius, max_iter=300, tol=1e-12)
            r_refined = _score_raw(refined)
            if r_refined > best_radius:
                best_points = refined.copy()
                best_radius = r_refined
        except Exception:
            pass

    # Local refinement
    if time.time() < deadline - 5.0:
        best_points = _local_refinement(best_points, n_steps=2000, step_size=0.002)
        best_radius = _score_raw(best_points)

    # Final Newton-Raphson polish
    if time.time() < deadline - 3.0:
        try:
            polished = _newton_refine(best_points, best_radius, max_iter=150, tol=1e-14)
            r_polished = _score_raw(polished)
            if r_polished > best_radius:
                best_points = polished.copy()
                best_radius = r_polished
        except Exception:
            pass

    # Final validation and clipping
    best_points = np.asarray(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] >= 1.0:
            best_points[i] *= 0.999 / norms[i]

    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = _hexagonal_seed()

    return best_points

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    points = construct()
    print("Final radius:", _score_raw(points))
