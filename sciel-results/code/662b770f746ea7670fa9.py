# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Approach:
    - For a fixed radius r, minimize a smooth penalty energy:
        E = 0.5 * sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |c_i| - (1-r))^2
      using L-BFGS-B.
    - Try several radii near the known best (~0.1258) and multiple diverse initial
      configurations (structured ring layouts + random restarts).
    - Track the best configuration that achieves near-zero energy; return it.
    """
    N = 50

    def energy_and_grad(c, r):
        """
        Compute the penalty energy and its gradient.
        Energy = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |c_i| - (1-r))^2
        """
        c = c.reshape(N, 2)
        energy = 0.0
        grad = np.zeros_like(c)

        # Pairwise interactions
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)

        # Overlap amount: overlap_ij = max(0, 2r - d_ij)
        overlap = np.maximum(0, 2 * r - dists)

        # Energy contribution: sum(overlap^2) / 2
        pair_energy = 0.5 * np.sum(overlap ** 2)
        energy += pair_energy

        # Gradient calculation
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        np.fill_diagonal(overlap, 0)

        # weights[i,j] = -2 * overlap[i,j] / safe_dists[i,j]
        weights = -2.0 * overlap / safe_dists
        np.fill_diagonal(weights, 0)

        grad_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        grad += grad_pairs

        # Boundary interactions
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))

        # E_bound = sum (bound_violation)^2
        energy += np.sum(bound_violation ** 2)

        # Gradient
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        grad_bound = 2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        grad += grad_bound

        return energy, grad

    def solve_for_radius(r, c_init):
        """
        Minimize energy for a fixed radius r starting from c_init.
        Returns (c_opt, min_energy).
        """
        c_flat = c_init.flatten()

        def obj(x):
            e, _ = energy_and_grad(x, r)
            return e

        def jac(x):
            _, g = energy_and_grad(x, r)
            return g.flatten()

        try:
            res = minimize(obj, c_flat, jac=jac, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            return res.x.reshape(N, 2), res.fun
        except Exception:
            return c_init, np.inf

    def make_ring_config(n_rings, base_radius, ring_counts, offsets=None):
        """
        Build a configuration with concentric rings.
        ring_counts: list of how many points on each ring (first ring may be center).
        offsets: optional phase offsets per ring (radians).
        """
        points = []
        if offsets is None:
            offsets = [0.0] * len(ring_counts)

        for k, n_pts in enumerate(ring_counts):
            if n_pts == 1 and k == 0:
                # center point
                points.append([0.0, 0.0])
            else:
                rad = base_radius * (k + 1)
                for i in range(n_pts):
                    angle = 2 * np.pi * i / n_pts + offsets[k]
                    points.append([rad * np.cos(angle), rad * np.sin(angle)])

        # If we have more than N points, keep the innermost N
        points = np.array(points)
        if len(points) > N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            points = points[idx]
        elif len(points) < N:
            # fill remaining with random points near boundary
            while len(points) < N:
                ang = np.random.uniform(0, 2 * np.pi)
                rad = np.random.uniform(0.7, 0.95)
                points = np.vstack([points, [rad * np.cos(ang), rad * np.sin(ang)]])
        return points[:N]

    def generate_initial_configurations():
        """
        Generate a list of diverse initial configurations.
        """
        configs = []

        # 1) Structured ring configurations (mimicking known near-optimal layouts)
        # Known best ~0.1258; typical ring radii for r ~ 0.126:
        #   center, then rings at ~0.25, 0.5, 0.75, 0.9
        ring_counts = [1, 6, 12, 18, 13]  # total 50
        base_r = 0.25
        configs.append(make_ring_config(5, base_r, ring_counts, offsets=[0, 0, np.pi/6, 0, np.pi/12]))

        # Slight variations in ring radii and phases
        for base in [0.24, 0.26]:
            configs.append(make_ring_config(5, base, ring_counts, offsets=[0, np.pi/12, 0, np.pi/12, 0]))

        # 2) Hexagonal lattice-like start
        for scale in [0.24, 0.26, 0.28]:
            pts = []
            for j in range(-8, 9):
                for i in range(-8, 9):
                    x = i * scale + 0.5 * scale * (j % 2)
                    y = j * scale * np.sqrt(3) / 2
                    if x * x + y * y < 0.81:  # keep well inside
                        pts.append([x, y])
            pts = np.array(pts)
            if len(pts) >= N:
                d = np.linalg.norm(pts, axis=1)
                idx = np.argsort(d)[:N]
                configs.append(pts[idx])
            else:
                # pad with random
                while len(pts) < N:
                    ang = np.random.uniform(0, 2 * np.pi)
                    rad = np.random.uniform(0.6, 0.9)
                    pts = np.vstack([pts, [rad * np.cos(ang), rad * np.sin(ang)]])
                configs.append(pts[:N])

        # 3) Random restarts (a few)
        for _ in range(4):
            c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.9
            if np.any(mask):
                c_rand[mask] /= norms[mask, np.newaxis] * 0.9
            configs.append(c_rand)

        return configs

    # Deterministic seed
    np.random.seed(12345)

    # Candidate radii around known best (~0.1258). We'll try a small grid.
    # We'll evaluate each radius with several initial configs and keep the best feasible.
    radii_to_try = [0.124, 0.125, 0.1255, 0.1258, 0.126, 0.1262, 0.1264]

    configs = generate_initial_configurations()

    best_centers = None
    best_energy = np.inf

    for r in radii_to_try:
        for c_init in configs:
            c_opt, e_opt = solve_for_radius(r, c_init)

            # If this configuration has near-zero energy, it's a feasible packing for radius r.
            # We prefer larger r, so we track the best (smallest energy) per r, and overall best r.
            if e_opt < best_energy:
                best_energy = e_opt
                best_centers = c_opt

            # Early stop for this radius if we found a very good solution
            if e_opt < 1e-10:
                # We can still try other radii (larger) to see if we can do better.
                # But for this radius, no need to try more starts.
                break

    # If the best solution still has nonzero energy, do a final polish at a slightly smaller radius
    if best_centers is not None and best_energy > 1e-8:
        # Try to reduce radius a bit and re-optimize to get a clean feasible solution
        r_polish = 0.124
        c_polish, e_polish = solve_for_radius(r_polish, best_centers)
        if e_polish < best_energy:
            best_energy = e_polish
            best_centers = c_polish

    # Fallback: if something went wrong, return a simple ring configuration
    if best_centers is None:
        best_centers = make_ring_config(5, 0.25, [1, 6, 12, 18, 13], offsets=[0, 0, np.pi/6, 0, np.pi/12])

    return best_centers
# EVOLVE-BLOCK-END
