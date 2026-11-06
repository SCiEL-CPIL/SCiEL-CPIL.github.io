"""
Fifty equal circles in a circle - ring-structure optimization with Newton-Raphson
contact-constraint refinement and soft-min L-BFGS-B polish.
Combines ring-based initialization, soft-min gradient optimization, and
Newton-Raphson on active contact/boundary constraints to break local minima.
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

    # Pre-compute pair indices for vectorization
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

    def _newton_refine(centers, r_target, max_iter=50, tol=1e-12):
        """
        Newton-Raphson refinement on active contact/boundary constraints.
        For a given target radius r_target, find positions where all active
        constraints (pairwise distance = 2*r_target or boundary distance = r_target)
        are satisfied.
        """
        centers = centers.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            # Compute current score
            current_r = _score_raw(centers)
            if current_r <= r_target - tol:
                # Try to increase r_target slightly
                r_target = max(current_r + 1e-6, r_target)
            else:
                r_target = min(r_target, current_r)
            
            # Identify active constraints: those within tolerance of the minimum
            norms = np.sqrt(np.sum(centers**2, axis=1))
            edge_dists = 1.0 - norms
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_dists = dists[iu]
            
            # Active constraints are those close to the minimum value
            min_constraint = min(np.min(pair_dists)/2.0, np.min(edge_dists))
            active_tol = max(1e-6, 0.05 * min_constraint)
            
            # Build constraint system
            # For each active pair constraint: |c_i - c_j| - 2*r = 0
            # For each active boundary constraint: 1 - |c_i| - r = 0
            
            active_pairs = []
            active_bounds = []
            
            for k in range(num_pairs):
                if abs(pair_dists[k]/2.0 - min_constraint) < active_tol:
                    active_pairs.append(k)
            
            for i in range(N):
                if abs(edge_dists[i] - min_constraint) < active_tol:
                    active_bounds.append(i)
            
            if len(active_pairs) + len(active_bounds) < 2:
                break
            
            # Build Jacobian and residual
            # Variables: 2*N coordinates
            # Constraints: |c_i - c_j| - 2*r for pairs, 1 - |c_i| - r for bounds
            
            n_constraints = len(active_pairs) + len(active_bounds)
            if n_constraints > 2 * N:
                # Too many constraints, pick a subset
                active_pairs = active_pairs[:2*N//2]
                active_bounds = active_bounds[:2*N - 2*len(active_pairs)]
                n_constraints = len(active_pairs) + len(active_bounds)
            
            if n_constraints < 2:
                break
            
            J = np.zeros((n_constraints, 2*N))
            res = np.zeros(n_constraints)
            
            r = min_constraint
            
            # Pair constraints
            for idx, k in enumerate(active_pairs):
                i = iu[0][k]
                j = iu[1][k]
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-15:
                    continue
                unit = d / dist
                res[idx] = dist - 2*r
                # d/dx_i = unit, d/dx_j = -unit
                J[idx, 2*i:2*i+2] = unit
                J[idx, 2*j:2*j+2] = -unit
            
            # Boundary constraints
            for idx, i in enumerate(active_bounds):
                row_idx = len(active_pairs) + idx
                norm_i = np.linalg.norm(centers[i])
                if norm_i < 1e-15:
                    continue
                unit_r = centers[i] / norm_i
                res[row_idx] = 1.0 - norm_i - r
                # d/dx_i = -unit_r
                J[row_idx, 2*i:2*i+2] = -unit_r
            
            # Solve least squares
            try:
                delta, residuals, rank, sv = np.linalg.lstsq(J, -res, rcond=None)
            except np.linalg.LinAlgError:
                break
            
            if np.max(np.abs(delta)) > 0.1:
                delta *= 0.1 / np.max(np.abs(delta))
            
            new_centers = centers.reshape(-1) + delta
            new_centers = new_centers.reshape(N, 2)
            
            # Check if this improves
            new_score = _score_raw(new_centers)
            if new_score > current_r - 1e-10:
                centers = new_centers
            else:
                # Try smaller step
                new_centers = centers.reshape(-1) + 0.5 * delta
                new_centers = new_centers.reshape(N, 2)
                new_score = _score_raw(new_centers)
                if new_score > current_r - 1e-10:
                    centers = new_centers
                else:
                    break
            
            if np.max(np.abs(delta)) < tol:
                break
        
        return centers

    def _hill_climb(centers, current_r, max_steps=200, deadline=deadline):
        """Exact hill climbing with random single-point moves."""
        centers = centers.copy()
        step_size = 0.003
        no_improve = 0
        max_no_improve = 50
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, size=2)
            new_c = centers[i] + delta
            
            if np.linalg.norm(new_c) >= 1.0:
                continue
            
            old_c = centers[i].copy()
            centers[i] = new_c
            
            new_r = _score_raw(centers)
            if new_r > current_r + 1e-12:
                current_r = new_r
                no_improve = 0
                step_size = min(step_size * 1.02, 0.01)
            else:
                centers[i] = old_c
                no_improve += 1
                if no_improve > max_no_improve:
                    step_size *= 0.9
                    no_improve = 0
                    if step_size < 1e-5:
                        break
        
        return centers, current_r

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

    # Hexagonal lattice
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

        # Progressive beta refinement with L-BFGS-B
        for beta in [100.0, 300.0, 600.0]:
            if time.time() > deadline - 15.0:
                break

            try:
                result = minimize(
                    _objective,
                    x0.flatten(),
                    args=(beta,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )

                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
            except Exception:
                continue

    # Newton-Raphson refinement on the best solution
    if best_centers is not None:
        current_r = best_score
        for _ in range(5):
            if time.time() > deadline - 15.0:
                break
            
            refined = _newton_refine(best_centers, current_r, max_iter=30)
            refined_score = _score_raw(refined)
            if refined_score > current_r:
                current_r = refined_score
                best_centers = refined
                best_score = refined_score
            else:
                break

        # Hill climbing
        np.random.seed(999)
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            
            hc_centers, hc_score = _hill_climb(best_centers, best_score, max_steps=100)
            if hc_score > best_score:
                best_centers = hc_centers
                best_score = hc_score

    # Final polish passes
    if best_centers is not None:
        for polish_beta in [800.0, 1200.0]:
            if time.time() > deadline - 8.0:
                break

            try:
                result = minimize(
                    _objective,
                    best_centers.flatten(),
                    args=(polish_beta,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
                )
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
            except Exception:
                continue

        # Final Newton refinement
        refined = _newton_refine(best_centers, best_score, max_iter=40)
        refined_score = _score_raw(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined

        # Perturbation-based local search with Newton
        np.random.seed(12345)
        for _ in range(8):
            if time.time() > deadline - 5.0:
                break

            perturbed = best_centers + np.random.normal(0, 0.003, size=(N, 2))
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.98
            if np.any(mask):
                perturbed[mask] *= 0.98 / norms[mask, np.newaxis]

            try:
                result = minimize(
                    _objective,
                    perturbed.flatten(),
                    args=(500.0,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                
                if score > best_score - 1e-6:
                    # Newton refine this candidate
                    refined = _newton_refine(centers, score, max_iter=20)
                    refined_score = _score_raw(refined)
                    if refined_score > best_score:
                        best_score = refined_score
                        best_centers = refined
            except Exception:
                continue

    # Final validation
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers

    # Fallback
    return _make_ring_config([0.0, 0.25, 0.50, 0.75, 0.95], [1, 6, 12, 18, 13])
