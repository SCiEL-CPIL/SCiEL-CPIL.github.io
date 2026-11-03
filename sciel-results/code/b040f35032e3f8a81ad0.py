"""Rewritten program: Lennard-Jones particle dynamics with multi-phase annealing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # 5s safety margin

    def _elapsed():
        return time.time() - start_time

    def _compute_radius(c):
        """Compute the minimum radius (score) for centers c of shape (N,3)."""
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        # Wall distances
        wall = np.minimum(np.minimum(c + 0.5, 0.5 - c), np.minimum(c + 0.5, 0.5 - c))
        # Actually: distance from c to nearest wall in each axis
        dist_walls = np.minimum(c + 0.5, 0.5 - c)  # shape (N,3)
        min_wall = np.min(dist_walls)
        return min(min_pair, min_wall)

    def _lj_forces(c, sigma, k_spring, anchors):
        """
        Compute forces on all particles using vectorized operations.
        LJ repulsion: F_ij = 24*sigma^2/d^3 - 24*sigma^12/d^11 (repulsive part only for stability)
        We use a soft repulsion: F = k_rep * (sigma/d)^12 * (d_vec/d) for d < 2*sigma
        Spring: F = -k_spring * (c - anchor)
        """
        n = len(c)
        forces = np.zeros_like(c)

        # Pairwise: for each pair i<j, compute force on i and j
        # Vectorized: compute all pair differences
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (n,n,3)
        d = np.sqrt(np.sum(diff * diff, axis=2))  # (n,n)
        np.fill_diagonal(d, np.inf)

        # Repulsive force: only for pairs closer than cutoff
        cutoff = 2.5 * sigma
        mask = (d < cutoff) & (d > 1e-12)

        # Force magnitude: k_rep * (sigma/d)^12
        # Use (sigma/d)^6 for a softer but still repulsive force to avoid numerical issues
        s_over_d = sigma / d  # (n,n)
        rep_mag = 10.0 * (s_over_d ** 12)  # strong repulsion
        rep_mag = np.where(mask, rep_mag, 0.0)

        # Force vector on i from j: rep_mag[i,j] * (c[i]-c[j])/d[i,j]
        # diff[i,j] = c[i] - c[j], so force on i is +rep_mag * diff/d
        # force on j is -rep_mag * diff/d
        force_vec = rep_mag[:, :, np.newaxis] * diff / d[:, :, np.newaxis]  # (n,n,3)
        # Sum over j for each i
        forces += np.sum(force_vec, axis=1)

        # Spring forces toward anchors
        spring = -k_spring * (c - anchors)
        forces += spring

        # Wall forces: push inward if too close to walls
        for axis in range(3):
            # Left wall at -0.5
            dist_left = c[:, axis] + 0.5
            too_close_left = dist_left < 0.05
            forces[too_close_left, axis] += 50.0 * (0.05 - dist_left[too_close_left])
            # Right wall at +0.5
            dist_right = 0.5 - c[:, axis]
            too_close_right = dist_right < 0.05
            forces[too_close_right, axis] -= 50.0 * (0.05 - dist_right[too_close_right])

        return forces

    def _generate_bcc_anchors(n_points):
        """Generate BCC lattice anchor points centered in the cube."""
        m = 3  # 3x3x3 BCC cells = 54 points
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k + 0.5])
        pts = np.array(pts, dtype=float)
        # Center
        pts -= pts.mean(axis=0)
        # Scale to fit in [-0.4, 0.4]
        max_ext = np.max(np.abs(pts))
        pts *= 0.4 / max_ext
        # Select N points (take the most central ones)
        if len(pts) >= n_points:
            dists_from_origin = np.sum(pts * pts, axis=1)
            idx = np.argsort(dists_from_origin)[:n_points]
            pts = pts[idx]
        else:
            # Pad with random
            extra = np.random.uniform(-0.3, 0.3, (n_points - len(pts), 3))
            pts = np.vstack([pts, extra])
        return pts

    def _generate_fcc_anchors(n_points):
        """Generate FCC-like lattice anchor points."""
        m = 3
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    # FCC: corners + face centers
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k])
                    pts.append([i + 0.5, j, k + 0.5])
                    pts.append([i, j + 0.5, k + 0.5])
        pts = np.array(pts, dtype=float)
        pts -= pts.mean(axis=0)
        max_ext = np.max(np.abs(pts))
        pts *= 0.4 / max_ext
        if len(pts) >= n_points:
            dists_from_origin = np.sum(pts * pts, axis=1)
            idx = np.argsort(dists_from_origin)[:n_points]
            pts = pts[idx]
        else:
            extra = np.random.uniform(-0.3, 0.3, (n_points - len(pts), 3))
            pts = np.vstack([pts, extra])
        return pts

    def _generate_grid_anchors(n_points):
        """Simple cubic grid anchors."""
        m = int(np.ceil(n_points ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        if len(pts) >= n_points:
            dists = np.sum(pts * pts, axis=1)
            idx = np.argsort(dists)[:n_points]
            pts = pts[idx]
        else:
            extra = np.random.uniform(-0.3, 0.3, (n_points - len(pts), 3))
            pts = np.vstack([pts, extra])
        return pts

    def _generate_random_anchors(n_points):
        return np.random.uniform(-0.4, 0.4, (n_points, 3))

    # Generate diverse anchor sets
    anchor_sets = [
        _generate_bcc_anchors(N),
        _generate_fcc_anchors(N),
        _generate_grid_anchors(N),
        _generate_random_anchors(N),
        _generate_bcc_anchors(N) + np.random.normal(0, 0.05, (N, 3)),
    ]

    best_r = -1.0
    best_centers = None

    # Multi-phase annealing for each anchor set
    for anchor_idx, anchors in enumerate(anchor_sets):
        if _elapsed() > time_limit * 0.15:
            break

        c = anchors.copy()
        c = np.clip(c, -0.45, 0.45)

        # Phase 1: High temperature exploration
        T = 0.05
        sigma = 0.25
        k_spring = 2.0
        dt = 0.002
        n_iter_phase1 = 2000

        for it in range(n_iter_phase1):
            if _elapsed() > time_limit * 0.4:
                break

            F = _lj_forces(c, sigma, k_spring, anchors)
            # Velocity-Verlet-like update with damping
            v = F * dt
            c += v
            c = np.clip(c, -0.49, 0.49)

            # Cool
            T *= 0.999
            sigma *= 0.9995
            k_spring *= 1.0005

        # Phase 2: Medium temperature refinement
        sigma = 0.2
        k_spring = 5.0
        dt = 0.001
        n_iter_phase2 = 3000

        for it in range(n_iter_phase2):
            if _elapsed() > time_limit * 0.7:
                break

            F = _lj_forces(c, sigma, k_spring, anchors)
            v = F * dt
            c += v
            c = np.clip(c, -0.49, 0.49)

            sigma *= 0.9998
            k_spring *= 1.0003

        # Phase 3: Fine relaxation - minimize radius directly via gradient
        # Use subgradient ascent on min radius
        r = _compute_radius(c)
        step = 0.001
        for it in range(2000):
            if _elapsed() > time_limit * 0.9:
                break

            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(d, np.inf)
            dist_walls = np.minimum(c + 0.5, 0.5 - c)
            r_curr = min(np.min(d) / 2.0, np.min(dist_walls))

            G = np.zeros_like(c)
            eps = 1e-6
            # Active pair constraints
            active = d < 2 * r_curr + eps
            for i in range(N):
                for j in range(i + 1, N):
                    if active[i, j]:
                        vec = c[i] - c[j]
                        norm = d[i, j]
                        if norm > 1e-9:
                            G[i] += vec / norm
                            G[j] -= vec / norm
            # Active wall constraints
            for i in range(N):
                for k in range(3):
                    if dist_walls[i, k] < r_curr + eps:
                        if c[i, k] > 0:
                            G[i, k] -= 1.0
                        else:
                            G[i, k] += 1.0

            norm_G = np.linalg.norm(G)
            if norm_G < 1e-10:
                break
            G /= norm_G

            c_new = c + step * G
            c_new = np.clip(c_new, -0.49, 0.49)
            r_new = _compute_radius(c_new)

            if r_new > r_curr:
                c = c_new
                step = min(step * 1.05, 0.005)
            else:
                step *= 0.7
                if step < 1e-8:
                    break

        r_final = _compute_radius(c)
        if r_final > best_r:
            best_r = r_final
            best_centers = c.copy()

        # Try a perturbation restart from this solution
        if _elapsed() < time_limit * 0.85:
            c_pert = best_centers + np.random.normal(0, 0.02, (N, 3))
            c_pert = np.clip(c_pert, -0.45, 0.45)
            # Quick refinement
            sigma = 0.18
            k_spring = 3.0
            dt = 0.001
            for it in range(1500):
                if _elapsed() > time_limit * 0.95:
                    break
                F = _lj_forces(c_pert, sigma, k_spring, anchors)
                v = F * dt
                c_pert += v
                c_pert = np.clip(c_pert, -0.49, 0.49)
                sigma *= 0.9999
                k_spring *= 1.0002

            # Gradient polish
            r = _compute_radius(c_pert)
            step = 0.001
            for it in range(1000):
                if _elapsed() > time_limit * 0.98:
                    break
                diff = c_pert[:, np.newaxis, :] - c_pert[np.newaxis, :, :]
                d = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(d, np.inf)
                dist_walls = np.minimum(c_pert + 0.5, 0.5 - c_pert)
                r_curr = min(np.min(d) / 2.0, np.min(dist_walls))

                G = np.zeros_like(c_pert)
                eps = 1e-6
                active = d < 2 * r_curr + eps
                for i in range(N):
                    for j in range(i + 1, N):
                        if active[i, j]:
                            vec = c_pert[i] - c_pert[j]
                            norm = d[i, j]
                            if norm > 1e-9:
                                G[i] += vec / norm
                                G[j] -= vec / norm
                for i in range(N):
                    for k in range(3):
                        if dist_walls[i, k] < r_curr + eps:
                            if c_pert[i, k] > 0:
                                G[i, k] -= 1.0
                            else:
                                G[i, k] += 1.0

                norm_G = np.linalg.norm(G)
                if norm_G < 1e-10:
                    break
                G /= norm_G

                c_new = c_pert + step * G
                c_new = np.clip(c_new, -0.49, 0.49)
                r_new = _compute_radius(c_new)

                if r_new > r_curr:
                    c_pert = c_new
                    step = min(step * 1.05, 0.005)
                else:
                    step *= 0.7
                    if step < 1e-8:
                        break

            r_pert = _compute_radius(c_pert)
            if r_pert > best_r:
                best_r = r_pert
                best_centers = c_pert.copy()

    # Final verification and cleanup
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]

    # Ensure all values are finite and within bounds
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    if not np.all(np.isfinite(best_centers)):
        # Last resort fallback
        m = int(np.ceil(N ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]

    return best_centers
# EVOLVE-BLOCK-END
