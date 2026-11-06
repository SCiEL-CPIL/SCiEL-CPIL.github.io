"""
Fifty equal circles in a circle - ring-structure optimization with full-coordinate refinement.
Uses ring-based initialization with optimized ring radii and angular offsets,
then refines all 100 coordinates with L-BFGS-B on a smooth soft-min objective,
followed by Newton-Raphson contact-graph polish for high-precision final refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    best_score = 0.0
    best_centers = None

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
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0

        all_constraints = np.concatenate([edge_constraints, pair_constraints])

        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / beta

        return -soft_min_val

    def _newton_polish(centers, max_iter=50, tol=1e-14):
        """Newton-Raphson polish using contact graph of near-active constraints.
        For each active constraint (pair distance or edge), enforce equality
        and solve the linear system for displacement increments.
        """
        centers = centers.copy().astype(np.float64)
        n = len(centers)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute all distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            norms = np.sqrt(np.sum(centers**2, axis=1))
            edge_dists = 1.0 - norms
            
            # Find the minimum constraint value
            iu = np.triu_indices(n, 1)
            pair_dists = dists[iu]
            min_pair = np.min(pair_dists)
            min_edge = np.min(edge_dists)
            
            # Determine the active threshold (within tolerance of the minimum)
            r_current = min(min_pair / 2.0, min_edge)
            threshold = r_current + 1e-8  # constraints close to active
            
            # Build constraint system: for each active constraint, 
            # the directional derivative should be zero (or constraint satisfied)
            A_rows = []
            b_vals = []
            
            # Pair constraints: |c_i - c_j| = 2r
            # Gradient w.r.t. c_i: (c_i - c_j) / |c_i - c_j|
            # Gradient w.r.t. c_j: -(c_i - c_j) / |c_i - c_j|
            # For constraint g = |c_i - c_j| - 2r = 0, we want to move along gradient
            # to increase g (push apart)
            active_pairs = []
            for idx in range(len(iu[0])):
                i, j = iu[0][idx], iu[1][idx]
                d = dists[i, j]
                if d < 2.0 * (r_current + 1e-6):
                    active_pairs.append((i, j, d))
            
            for (i, j, d) in active_pairs:
                if d < 1e-12:
                    continue
                dir_vec = (centers[i] - centers[j]) / d
                # Row for c_i: +dir_vec, for c_j: -dir_vec
                row = np.zeros(2 * n)
                row[2*i:2*i+2] = dir_vec
                row[2*j:2*j+2] = -dir_vec
                # RHS: we want to increase distance, so target is 2*r_current
                # Current distance is d, need to increase by (2*r_current - d)
                # But in Newton, we solve A*delta = -g where g = d - 2*r_target
                # Actually, let's use: minimize sum of (d_ij - 2r)^2 subject to feasibility
                # Simpler: just push apart along the direction
                A_rows.append(row)
                b_vals.append(2.0 * r_current - d)  # positive means need to increase
            
            # Edge constraints: |c_i| = 1 - r
            for i in range(n):
                norm_i = norms[i]
                if norm_i > 1.0 - (r_current + 1e-6):
                    if norm_i < 1e-12:
                        dir_vec = np.array([1.0, 0.0])
                    else:
                        dir_vec = centers[i] / norm_i
                    row = np.zeros(2 * n)
                    row[2*i:2*i+2] = dir_vec
                    A_rows.append(row)
                    b_vals.append(1.0 - r_current - norm_i)  # negative means need to decrease radius
            
            if len(A_rows) == 0:
                break
            
            A = np.array(A_rows)
            b = np.array(b_vals)
            
            # Solve least-squares: A * delta = b
            try:
                delta, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                break
            
            delta = delta.reshape(n, 2)
            
            # Check if step is too large
            step_norm = np.sqrt(np.sum(delta**2))
            if step_norm < 1e-15:
                break
            
            # Damping
            alpha = 1.0
            for _ in range(20):
                new_centers = centers + alpha * delta
                new_norms = np.sqrt(np.sum(new_centers**2, axis=1))
                if np.any(new_norms >= 1.0):
                    alpha *= 0.5
                    continue
                new_diff = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
                new_dists = np.sqrt(np.sum(new_diff**2, axis=2))
                np.fill_diagonal(new_dists, np.inf)
                if np.min(new_dists) < 1e-10:
                    alpha *= 0.5
                    continue
                break
            
            centers = centers + alpha * delta
            
            # Check convergence
            if step_norm * alpha < tol:
                break
        
        return centers

    def _gradient(centers, beta=300.0):
        """Compute gradient of the objective - vectorized.
        
        The objective is -soft_min(constraints) where soft_min approximates min(constraints).
        d/dx [-soft_min] = -d/dx[soft_min] = -sum_i w_i * d(constraint_i)/dx
        where w_i = softmax(-beta * constraint_i) / sum(softmax(-beta * constraint_i))
        
        For edge constraint c_i = 1 - |x_i|: d/dx_i = -x_i/|x_i|
        For pair constraint c_ij = |x_i - x_j|/2: 
            d/dx_i = (x_i - x_j) / (2*|x_i - x_j|)
            d/dx_j = -(x_i - x_j) / (2*|x_i - x_j|)
        """
        centers = centers.reshape(-1, 2)
        grad = np.zeros_like(centers)

        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-15
        edge_constraints = 1.0 - norms

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-15
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0

        all_constraints = np.concatenate([edge_constraints, pair_constraints])

        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp

        # Edge gradient: d(1-|x_i|)/dx_i = -x_i/|x_i|
        # d/dx [-soft_min] += -w_i * beta * (-x_i/|x_i|) = w_i * beta * x_i/|x_i|
        edge_grad = -centers / norms[:, np.newaxis]
        grad += -weights[:N, np.newaxis] * beta * edge_grad

        # Pair gradients: d(|x_i-x_j|/2)/dx_i = (x_i-x_j)/(2|xi-xj|)
        # d/dx [-soft_min] += -w_ij * beta * (x_i-x_j)/(2|xi-xj|) for xi
        #                       +w_ij * beta * (x_i-x_j)/(2|xi-xj|) for xj
        n_pairs = len(pair_constraints)
        i_idx = iu[0]
        j_idx = iu[1]
        w_pair = weights[N:N+n_pairs]

        diff_pairs = centers[i_idx] - centers[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        # (x_i - x_j) / (2 * |x_i - x_j|)
        dir_vecs = diff_pairs / (2.0 * dist_pairs[:, np.newaxis])

        np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * beta * dir_vecs)
        np.add.at(grad, j_idx, w_pair[:, np.newaxis] * beta * dir_vecs)

        return grad.reshape(-1)

    def _make_ring_config(ring_radii, ring_counts, angles_offsets=None):
        """Create a ring-based configuration."""
        centers = []

        # Center
        if ring_counts[0] > 0:
            centers.append([0.0, 0.0])

        for ring_idx in range(1, len(ring_radii)):
            r = ring_radii[ring_idx]
            count = ring_counts[ring_idx]
            offset = angles_offsets[ring_idx] if angles_offsets else 0.0
            for k in range(count):
                theta = 2 * np.pi * k / count + offset
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                centers.append([x, y])

        centers = np.array(centers[:N])
        return centers

    # Generate diverse initial configurations
    configs = []

    # Configuration 1: Standard rings with known-good structure
    radii1 = [0.0, 0.25, 0.50, 0.75, 0.95]
    counts1 = [1, 6, 12, 18, 13]
    configs.append(_make_ring_config(radii1, counts1))

    # Configuration 2: Slightly compressed
    radii2 = [0.0, 0.23, 0.48, 0.73, 0.94]
    counts2 = [1, 6, 12, 18, 13]
    configs.append(_make_ring_config(radii2, counts2))

    # Configuration 3: Expanded
    radii3 = [0.0, 0.27, 0.52, 0.74, 0.93]
    counts3 = [1, 6, 12, 18, 13]
    configs.append(_make_ring_config(radii3, counts3))

    # Configuration 4: Angle offset on ring 2
    offsets4 = [0.0, 0.0, np.pi/12, 0.0, 0.0]
    configs.append(_make_ring_config(radii1, counts1, offsets4))

    # Configuration 5: Angle offset on ring 3
    offsets5 = [0.0, 0.0, 0.0, np.pi/18, 0.0]
    configs.append(_make_ring_config(radii1, counts1, offsets5))

    # Configuration 6: Different ring structure - 4 rings
    radii6 = [0.0, 0.30, 0.60, 0.90]
    counts6 = [1, 8, 18, 23]
    configs.append(_make_ring_config(radii6, counts6))

    # Configuration 7: 4 rings with offsets
    offsets7 = [0.0, 0.0, np.pi/8, 0.0]
    configs.append(_make_ring_config(radii6, counts6, offsets7))

    # Configuration 8: 3 rings
    radii8 = [0.0, 0.45, 0.88]
    counts8 = [1, 14, 35]
    configs.append(_make_ring_config(radii8, counts8))

    # Configuration 9: 5 rings with different distribution
    radii9 = [0.0, 0.22, 0.45, 0.68, 0.92]
    counts9 = [1, 7, 13, 17, 12]
    configs.append(_make_ring_config(radii9, counts9))

    # Configuration 10: Multiple angle offsets
    offsets10 = [0.0, np.pi/6, np.pi/12, -np.pi/18, np.pi/24]
    configs.append(_make_ring_config(radii1, counts1, offsets10))

    # Configuration 6: Hexagonal lattice
    def _hexagonal_config(spacing=0.26):
        points = []
        for j in range(-8, 9):
            for i in range(-8, 9):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x*x + y*y)
                if r <= 1.0 - 0.13 - 1e-9:
                    points.append([x, y])

        points = np.array(points)
        if len(points) < N:
            return None

        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]

    hex_cfg = _hexagonal_config()
    if hex_cfg is not None:
        configs.append(hex_cfg)

    # Random configurations
    for seed in range(4):
        np.random.seed(42 + seed * 7)
        cfg = np.random.uniform(-0.85, 0.85, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        if np.any(mask):
            cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        configs.append(cfg)

    # Optimize each configuration
    for idx, x0 in enumerate(configs):
        if time.time() > deadline - 10.0:
            break

        # Progressive beta refinement
        for beta in [80.0, 200.0, 400.0]:
            if time.time() > deadline - 15.0:
                break

            try:
                result = minimize(
                    _objective,
                    x0.flatten(),
                    args=(beta,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )

                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
            except Exception:
                continue

    # Final refinement passes on the best solution
    if best_centers is not None:
        for polish_beta in [500.0, 700.0, 1000.0]:
            if time.time() > deadline - 20.0:
                break

            try:
                result = minimize(
                    _objective,
                    best_centers.flatten(),
                    args=(polish_beta,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 60}
                )
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
            except Exception:
                continue

        # Perturbation-based local search
        np.random.seed(12345)
        for _ in range(12):
            if time.time() > deadline - 25.0:
                break

            perturbed = best_centers + np.random.normal(0, 0.002, size=(N, 2))
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.98
            if np.any(mask):
                perturbed[mask] *= 0.98 / norms[mask, np.newaxis]

            try:
                result = minimize(
                    _objective,
                    perturbed.flatten(),
                    args=(600.0,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
            except Exception:
                continue

        # Newton-Raphson contact-graph polish
        if time.time() < deadline - 5.0:
            try:
                polished = _newton_polish(best_centers, max_iter=80, tol=1e-15)
                score = _score_raw(polished)
                if score > best_score:
                    best_score = score
                    best_centers = polished.copy()
            except Exception:
                pass

        # One more L-BFGS pass after Newton polish
        if time.time() < deadline - 3.0 and best_centers is not None:
            try:
                result = minimize(
                    _objective,
                    best_centers.flatten(),
                    args=(1200.0,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 60}
                )
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
            except Exception:
                pass

        # Final Newton polish
        if time.time() < deadline - 2.0 and best_centers is not None:
            try:
                polished = _newton_polish(best_centers, max_iter=50, tol=1e-16)
                score = _score_raw(polished)
                if score > best_score:
                    best_score = score
                    best_centers = polished.copy()
            except Exception:
                pass

    # Final validation
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers

    # Fallback
    return _make_ring_config([0.0, 0.25, 0.50, 0.75, 0.95], [1, 6, 12, 18, 13])
