"""
Fifty equal circles in a circle - rigid-linkage contact-graph optimization.
Represents the packing as a mechanical linkage where circles are nodes connected
by rigid bars of length 2r. Optimizes r by iteratively solving for a valid embedding
using force-directed layout with soft constraint penalties.
"""
import numpy as np
import time
from scipy.optimize import minimize, Bounds

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def _score_raw(centers):
        """Compute the raw score: minimum radius for valid packing."""
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

    def _make_ring_config(ring_radii, ring_counts, angles_offset=0.0):
        """Generate a ring-based initial configuration."""
        centers = []
        if len(ring_counts) > 0 and ring_counts[0] > 0:
            centers.append([0.0, 0.0])
        for ring_idx in range(1, len(ring_radii)):
            if ring_idx >= len(ring_counts):
                break
            r = ring_radii[ring_idx]
            count = ring_counts[ring_idx]
            for k in range(count):
                theta = 2 * np.pi * k / count + angles_offset * ring_idx
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                centers.append([x, y])
        return np.array(centers[:N])

    def _hexagonal_config(spacing=0.26):
        """Generate a hexagonal lattice configuration."""
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x*x + y*y)
                if r <= 1.0 - 0.12 - 1e-9:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]

    def _force_directed_relax(centers, r_target, max_iter=3000, lr_init=0.005,
                              lr_decay=0.999, min_lr=1e-7, wall_strength=50.0,
                              repulsion_alpha=8.0):
        """
        Force-directed relaxation with rigid-bar constraints.
        Circles are nodes; pairs closer than 2*r_target get repulsive forces.
        Wall potential keeps circles inside the unit disk.
        """
        centers = centers.copy().reshape(N, 2)
        lr = lr_init
        prev_energy = np.inf

        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break

            energy = 0.0
            grad = np.zeros_like(centers)

            # Pair repulsion: strong when too close
            for i in range(N):
                for j in range(i + 1, N):
                    dvec = centers[i] - centers[j]
                    d = np.sqrt(np.sum(dvec**2))
                    if d < 1e-12:
                        d = 1e-12
                        dvec = np.array([1e-12, 0.0])
                    if d < 2.0 * r_target * 1.2:
                        # Rigid-bar-like: very stiff below 2*r_target
                        if d < 2.0 * r_target:
                            ratio = r_target / d
                            energy += 10.0 * ratio ** repulsion_alpha
                            coeff = 10.0 * repulsion_alpha * (r_target ** repulsion_alpha) * (d ** (-(repulsion_alpha + 1)))
                        else:
                            # Soft repulsion between 2*r_target and 2.4*r_target
                            excess = d - 2.0 * r_target
                            energy += excess * excess
                            coeff = 2.0 * excess / d
                        grad[i] += coeff * dvec
                        grad[j] -= coeff * dvec

            # Wall potential
            for i in range(N):
                norm = np.sqrt(np.sum(centers[i]**2))
                if norm < 1e-12:
                    continue
                margin = 1.0 - norm
                if margin < r_target:
                    excess = r_target - margin
                    if excess > 0:
                        energy += wall_strength * excess * excess
                        grad[i] -= 2.0 * wall_strength * excess * (centers[i] / norm)

            # Adaptive step
            if energy > prev_energy + 1e-12:
                lr *= 0.5
            else:
                lr = min(lr * 1.01, lr_init)

            centers -= lr * grad
            prev_energy = energy

            # Project back inside disk
            norms = np.sqrt(np.sum(centers**2, axis=1))
            mask = norms > 0.999
            if np.any(mask):
                centers[mask] *= 0.999 / norms[mask, np.newaxis]

            if lr < min_lr and it > 50:
                break

        return centers

    def _contact_graph_newton(centers, max_iter=200, tol=1e-12):
        """
        Newton-Raphson refinement using contact graph structure.
        Identifies active constraints (pairs at distance ~2r, points near boundary)
        and solves the linear system for optimal displacement.
        """
        centers = centers.copy().reshape(N, 2)

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            norms = np.sqrt(np.sum(centers**2, axis=1))
            edge_constraints = 1.0 - norms
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_constraints = dists / 2.0

            r = min(np.min(pair_constraints), np.min(edge_constraints))

            # Find active constraints
            tol_active = max(1e-9, 0.001 * r)
            active_pairs = []
            active_edges = []

            for i in range(N):
                if edge_constraints[i] < r + tol_active:
                    active_edges.append(i)
            for i in range(N):
                for j in range(i + 1, N):
                    if pair_constraints[i, j] < r + tol_active:
                        active_pairs.append((i, j))

            n_active = len(active_pairs) + len(active_edges)
            if n_active < 3:
                break

            # Build KKT system: maximize r subject to active constraints
            # Variables: 2N center coords + r
            # We use a simplified approach: perturb along gradient of soft-min

            beta = 3000.0
            all_constraints = np.concatenate([edge_constraints, pair_constraints[
                np.triu_indices(N, 1)]])
            neg_beta_x = -beta * all_constraints
            max_val = np.max(neg_beta_x)
            exp_vals = np.exp(neg_beta_x - max_val)
            sum_exp = np.sum(exp_vals)
            weights = exp_vals / sum_exp

            grad = np.zeros_like(centers)
            edge_w = weights[:N]
            for i in range(N):
                if norms[i] > 1e-12:
                    grad[i] += edge_w[i] * (-centers[i] / norms[i])

            iu = np.triu_indices(N, 1)
            pair_w = weights[N:]
            for k in range(len(pair_w)):
                i, j = iu[0][k], iu[1][k]
                if dists[i, j] > 1e-12:
                    dir_vec = (centers[i] - centers[j]) / dists[i, j]
                    grad[i] += pair_w[k] * dir_vec
                    grad[j] -= pair_w[k] * dir_vec

            grad_norm = np.sqrt(np.sum(grad**2))
            if grad_norm < 1e-15:
                break

            # Line search
            step = 0.02
            improved = False
            for _ in range(25):
                new_centers = centers + step * grad / grad_norm
                new_norms = np.sqrt(np.sum(new_centers**2, axis=1))
                if np.any(new_norms >= 1.0):
                    step *= 0.5
                    continue
                new_diff = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
                new_dists = np.sqrt(np.sum(new_diff**2, axis=2))
                np.fill_diagonal(new_dists, np.inf)
                new_r = min(np.min(new_dists / 2.0), np.min(1.0 - new_norms))
                if new_r > r:
                    centers = new_centers
                    r = new_r
                    improved = True
                    break
                else:
                    step *= 0.5

            if not improved and step < 1e-13:
                break

        return centers

    def _scipy_optimize(centers, r_target):
        """Use scipy L-BFGS-B to optimize the soft-min objective."""
        def objective(x):
            c = x.reshape(N, 2)
            if not np.all(np.isfinite(c)):
                return 1e10
            norms = np.sqrt(np.sum(c**2, axis=1))
            if np.any(norms >= 1.0):
                return 1e10
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            # Soft-min approximation
            constraints = np.concatenate([1.0 - norms, dists[np.triu_indices(N, 1)] / 2.0])
            beta = 1500.0
            neg_beta_x = -beta * constraints
            max_val = np.max(neg_beta_x)
            exp_vals = np.exp(neg_beta_x - max_val)
            sum_exp = np.sum(exp_vals)
            # Negative soft-min (we want to maximize min, so minimize -soft-min)
            return -np.log(sum_exp) / beta + max_val / beta

        def gradient(x):
            c = x.reshape(N, 2)
            if not np.all(np.isfinite(c)):
                return np.zeros_like(x)
            norms = np.sqrt(np.sum(c**2, axis=1))
            if np.any(norms >= 1.0):
                return np.zeros_like(x)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)

            constraints = np.concatenate([1.0 - norms, dists[np.triu_indices(N, 1)] / 2.0])
            beta = 1500.0
            neg_beta_x = -beta * constraints
            max_val = np.max(neg_beta_x)
            exp_vals = np.exp(neg_beta_x - max_val)
            sum_exp = np.sum(exp_vals)
            weights = exp_vals / sum_exp

            grad_c = np.zeros_like(c)
            edge_w = weights[:N]
            for i in range(N):
                if norms[i] > 1e-12:
                    grad_c[i] += edge_w[i] * (-c[i] / norms[i])

            iu = np.triu_indices(N, 1)
            pair_w = weights[N:]
            for k in range(len(pair_w)):
                i, j = iu[0][k], iu[1][k]
                if dists[i, j] > 1e-12:
                    dir_vec = (c[i] - c[j]) / dists[i, j]
                    grad_c[i] += pair_w[k] * dir_vec
                    grad_c[j] -= pair_w[k] * dir_vec

            return grad_c.flatten()

        x0 = centers.flatten()
        bounds = [(-0.99, 0.99)] * (2 * N)

        try:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 500, 'maxls': 100,
                                       'ftol': 1e-14, 'gtol': 1e-12})
            if result.success or result.fun < objective(x0):
                return result.x.reshape(N, 2)
        except Exception:
            pass

        return centers

    # Generate diverse initial configurations
    initial_configs = []

    # Ring configurations with various parameters
    ring_params = [
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.0),
        ([0.0, 0.22, 0.48, 0.72, 0.94], [1, 6, 12, 18, 13], 0.0),
        ([0.0, 0.28, 0.52, 0.74, 0.93], [1, 6, 12, 18, 13], 0.0),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.2),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.5),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.3),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.1),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.7),
        ([0.0, 0.26, 0.51, 0.76, 0.96], [1, 6, 12, 18, 13], 0.4),
        ([0.0, 0.24, 0.49, 0.74, 0.94], [1, 6, 12, 18, 13], 0.6),
    ]
    for radii, counts, offset in ring_params:
        initial_configs.append(_make_ring_config(radii, counts, offset))

    # Hexagonal configurations
    for spacing in [0.255, 0.26, 0.265, 0.27]:
        hex_cfg = _hexagonal_config(spacing)
        if hex_cfg is not None:
            initial_configs.append(hex_cfg)

    # Random configurations
    for seed in range(8):
        np.random.seed(2000 + seed * 6271)
        cfg = np.random.uniform(-0.88, 0.88, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.88
        if np.any(mask):
            cfg[mask] *= 0.88 / norms[mask, np.newaxis]
        initial_configs.append(cfg)

    # Process each initial config with multiple optimization strategies
    best_score = 0.0
    best_centers = None

    for idx, x0 in enumerate(initial_configs):
        if time.time() > deadline - 20.0:
            break

        # Strategy 1: Force-directed relaxation
        if time.time() < deadline - 15.0:
            try:
                relaxed = _force_directed_relax(x0, r_target=0.126, max_iter=2000)
                score = _score_raw(relaxed)
                if score > best_score:
                    best_score = score
                    best_centers = relaxed.copy()
            except Exception:
                pass

        # Strategy 2: Newton refinement on relaxed config
        if time.time() < deadline - 10.0 and best_centers is not None:
            try:
                refined = _contact_graph_newton(relaxed, max_iter=150)
                score = _score_raw(refined)
                if score > best_score:
                    best_score = score
                    best_centers = refined.copy()
            except Exception:
                pass

        # Strategy 3: Scipy optimization
        if time.time() < deadline - 8.0:
            try:
                optimized = _scipy_optimize(x0, r_target=0.126)
                score = _score_raw(optimized)
                if score > best_score:
                    best_score = score
                    best_centers = optimized.copy()
            except Exception:
                pass

    # Final polish on best solution
    if best_centers is not None and time.time() < deadline - 5.0:
        for _ in range(4):
            if time.time() > deadline - 3.0:
                break
            try:
                polished = _contact_graph_newton(best_centers, max_iter=200)
                score = _score_raw(polished)
                if score > best_score:
                    best_score = score
                    best_centers = polished.copy()
                else:
                    break
            except Exception:
                break

    # Final validation
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers

    # Fallback: ring configuration
    return _make_ring_config([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13])
