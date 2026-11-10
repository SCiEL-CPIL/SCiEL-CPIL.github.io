import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def gen_superlattice(rotation, shear, scale):
        """Generate a hex lattice, apply affine transform, prune to N points inside unit circle."""
        # Base hex lattice with fine spacing
        base_spacing = 0.045  # dense enough to have >600 points in unit circle
        k = int(1.2 / base_spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * base_spacing
                y = j * base_spacing * np.sqrt(3.0) / 2.0
                rows.append([x, y])
        base_pts = np.array(rows)

        # Apply shear: x' = x + shear*y, y' = y
        base_pts[:, 0] += shear * base_pts[:, 1]

        # Apply rotation
        c, s = np.cos(rotation), np.sin(rotation)
        rot = np.array([[c, -s], [s, c]])
        base_pts = base_pts @ rot.T

        # Apply uniform scaling
        base_pts *= scale

        # Keep only points inside unit circle with margin
        radii = np.linalg.norm(base_pts, axis=1)
        mask = radii < 1.0
        base_pts = base_pts[mask]

        if len(base_pts) < N:
            return None

        # Select N points closest to center
        radii = np.linalg.norm(base_pts, axis=1)
        idx = np.argsort(radii, kind="stable")[:N]
        return base_pts[idx].copy()

    def force_directed_optimize(points, start_time, deadline, max_iter=3000):
        """Run force-directed optimization, return best points and score."""
        best_score = compute_score(points)
        best_points = points.copy()
        spacing_est = 0.053
        step_size = spacing_est * 0.1
        lr_decay = 0.9995

        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                break

            n = len(points)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            dist[dist < 1e-10] = 1e-10
            np.fill_diagonal(dist, np.inf)

            close_mask = (dist < 5 * spacing_est) & (dist < np.inf)
            inv_dist_sq = np.zeros_like(dist)
            inv_dist_sq[close_mask] = 1.0 / (dist[close_mask] ** 2)

            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

            radii = np.linalg.norm(points, axis=1)
            boundary_force = np.zeros_like(points)
            boundary_margin = 0.1
            over = radii > (1.0 - boundary_margin)
            if np.any(over):
                mag = (radii[over] - (1.0 - boundary_margin)) * 5.0
                boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]

            forces += boundary_force

            force_norms = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norms)
            if max_force > 1e-10:
                forces /= max_force

            points += step_size * forces

            radii = np.linalg.norm(points, axis=1)
            mask_out = radii > 1.0
            if np.any(mask_out):
                points[mask_out] /= radii[mask_out, np.newaxis]

            current_score = compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()

            step_size *= lr_decay
            if step_size < 1e-7:
                break

        return best_points, best_score

    def gradient_polish(points, start_time, deadline, max_steps=5000):
        """Gradient-based fine-tuning."""
        best_score = compute_score(points)
        best_points = points.copy()
        polish_step = 0.004
        polish_decay = 0.999

        for step in range(max_steps):
            if time.time() > deadline - 5:
                break

            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)

            min_pair_dist = np.min(dists)
            norms = np.linalg.norm(points, axis=1)
            min_boundary_dist = np.min(1.0 - norms)

            current_score = min(min_pair_dist / 2.0, min_boundary_dist)

            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()

            flat_dists = dists.flatten()
            k = min(25, N * (N - 1) // 2)
            part_idx = np.argpartition(flat_dists, k)[:k]

            pair_grad = np.zeros_like(points)
            for idx in part_idx:
                i = int(idx // N)
                j = int(idx % N)
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-6:
                    continue
                vec = (points[i] - points[j]) / d
                pair_grad[i] += vec
                pair_grad[j] -= vec

            boundary_grad = np.zeros_like(points)
            close_to_boundary = norms > (1.0 - current_score - 0.003)
            for i in np.where(close_to_boundary)[0]:
                if norms[i] > 1e-6:
                    boundary_grad[i] = -points[i] / norms[i]

            pair_score = min_pair_dist / 2.0
            bound_score = min_boundary_dist

            if pair_score <= bound_score:
                w_pair, w_bound = 1.0, 0.2
            else:
                w_pair, w_bound = 0.2, 1.0

            grad = w_pair * pair_grad + w_bound * boundary_grad
            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-6:
                grad = grad / g_norm

            points += polish_step * grad

            norms_new = np.linalg.norm(points, axis=1)
            outside = norms_new > 0.999
            if np.any(outside):
                points[outside] = points[outside] * (0.999 / norms_new[outside, np.newaxis])

            polish_step = max(1e-5, polish_step * polish_decay)

        return best_points, best_score

    def scipy_polish(points, start_time, deadline):
        """Final scipy BFGS polish on closest pairs."""
        if time.time() > deadline - 3:
            return points, compute_score(points)

        from scipy.optimize import minimize

        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)

        flat_dist = dist.flatten()
        k = min(30, n * (n - 1) // 2)
        smallest_indices = np.argpartition(flat_dist, k)[:k]

        pairs = []
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            if i < j:
                pairs.append((i, j))

        def objective(x):
            pts = x.reshape(-1, 2)
            penalty = 0.0
            for i, j in pairs:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 0.11:
                    penalty += (0.11 - d) ** 2
            r = np.linalg.norm(pts, axis=1)
            over = r > 0.95
            if np.any(over):
                penalty += np.sum((r[over] - 0.95) ** 2) * 10
            return penalty

        x0 = points.flatten()
        best_score = compute_score(points)
        best_points = points.copy()
        try:
            result = minimize(objective, x0, method='BFGS', options={'maxiter': 80, 'gtol': 1e-8})
            new_points = result.x.reshape(-1, 2)
            radii = np.linalg.norm(new_points, axis=1)
            mask_out = radii > 1.0
            if np.any(mask_out):
                new_points[mask_out] /= radii[mask_out, np.newaxis]
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except:
            pass

        return best_points, best_score

    # === Superlattice strategy: try multiple affine transformations ===
    overall_best_points = None
    overall_best_score = -1.0

    # List of (rotation, shear, scale) candidates
    # Start near optimal hex packing parameters, explore variations
    candidates = []

    # Baseline: standard hex packing (no shear, small rotation)
    candidates.append((0.0, 0.0, 1.0))

    # Small rotations
    for rot in [np.pi/12, -np.pi/12, np.pi/6, -np.pi/6, np.pi/18, -np.pi/18, np.pi/30, -np.pi/30]:
        candidates.append((rot, 0.0, 1.0))

    # Small shears
    for sh in [0.05, -0.05, 0.1, -0.1, 0.15, -0.15]:
        candidates.append((0.0, sh, 1.0))

    # Combined rotation + shear
    for rot in [np.pi/12, -np.pi/12, np.pi/18]:
        for sh in [0.05, -0.05, 0.1]:
            candidates.append((rot, sh, 1.0))

    # Slight scaling variations
    for sc in [0.99, 1.01, 0.98, 1.02]:
        candidates.append((0.0, 0.0, sc))

    # More diverse orientations
    for rot in [np.pi/8, 3*np.pi/16, np.pi/5, 2*np.pi/9]:
        candidates.append((rot, 0.05, 1.0))
        candidates.append((rot, -0.05, 1.0))

    # Random additional candidates
    rng = np.random.default_rng(42)
    for _ in range(15):
        rot = rng.uniform(0, np.pi/3)
        sh = rng.uniform(-0.2, 0.2)
        sc = rng.uniform(0.97, 1.03)
        candidates.append((rot, sh, sc))

    # Time budget per candidate: use remaining time divided by number of candidates
    # But prioritize early candidates; use a time-per-candidate approach
    time_per_candidate = 4.0  # seconds per candidate for initial pass
    candidates_evaluated = 0

    for rot, sh, sc in candidates:
        if time.time() > deadline - 30:
            break

        pts = gen_superlattice(rot, sh, sc)
        if pts is None or len(pts) < N:
            continue

        candidates_evaluated += 1

        # Force-directed optimization
        pts, score = force_directed_optimize(pts, start_time, deadline - 25, max_iter=2000)

        if score > overall_best_score:
            overall_best_score = score
            overall_best_points = pts.copy()

        # Gradient polish for the best candidate so far
        if score > overall_best_score * 0.999:
            pts, score = gradient_polish(pts, start_time, deadline - 10, max_steps=3000)
            if score > overall_best_score:
                overall_best_score = score
                overall_best_points = pts.copy()

    # Final polish on the overall best
    if overall_best_points is not None and time.time() < deadline - 5:
        overall_best_points, overall_best_score = gradient_polish(
            overall_best_points, start_time, deadline - 5, max_steps=4000)

    if overall_best_points is not None and time.time() < deadline - 2:
        overall_best_points, overall_best_score = scipy_polish(overall_best_points, start_time, deadline)

    # Fallback: if superlattice failed, use simple hex
    if overall_best_points is None:
        spacing = 2.0 * np.sqrt(0.907 / N)
        while True:
            k = int(1.0 / spacing) + 3
            pts_list = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) <= 1.0 - spacing * 0.5:
                        pts_list.append((x, y))
            pts = np.array(pts_list)
            if len(pts) >= N:
                break
            spacing *= 0.98
        radii = np.hypot(pts[:, 0], pts[:, 1])
        overall_best_points = pts[np.argsort(radii, kind="stable")[:N]].copy()
        overall_best_score = compute_score(overall_best_points)
        if time.time() < deadline - 5:
            overall_best_points, overall_best_score = force_directed_optimize(
                overall_best_points, start_time, deadline - 5, max_iter=2000)

    return overall_best_points
