"""Physics-based molecular dynamics with repulsive forces, initialized from hex lattice,
then refined with local optimization. Structural change: MD simulation drives the system
into a local energy minimum where circles are equidistant from neighbors and boundary."""

import numpy as np
import time
from scipy.optimize import minimize

N = 300


def construct():
    t0 = time.time()
    DEADLINE = t0 + 135.0  # 15s safety margin under the 150s kill

    def elapsed():
        return time.time() - t0

    # ---------- Score (exact, double precision) ----------
    def score(points):
        n = points.shape[0]
        if n < 2:
            return 0.0
        # pair distances
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = dist.min()
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - radii.max()
        return min(0.5 * min_pair, min_boundary)

    # ---------- Hex lattice seed ----------
    def gen_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.6:
                    pts.append((x, y))
        return np.array(pts)

    spacing = 2.0 * np.sqrt(0.907 / N)
    while True:
        cand = gen_hex_lattice(spacing)
        if len(cand) >= N:
            break
        spacing *= 0.98
    radii = np.hypot(cand[:, 0], cand[:, 1])
    order = np.argsort(radii, kind="stable")
    seed = cand[order[:N]].copy()

    best_pts = seed.copy()
    best_s = score(best_pts)

    # ---------- Molecular Dynamics Simulation ----------
    def md_simulate(points, n_steps=5000, dt=0.001, cooling_factor=0.999):
        """
        Physics-based MD with repulsive forces (inverse power law) and confining potential.
        """
        pts = points.copy()
        n = pts.shape[0]
        
        # Estimate target spacing
        target_spacing = 2.0 * np.sqrt(0.9 / n)
        
        # Force parameters
        repulsion_power = 12  # Lennard-Jones-like
        attraction_power = 6
        epsilon = 1.0
        sigma = target_spacing
        
        # Confining potential strength
        confining_strength = 100.0
        
        # Velocity Verlet integration
        velocities = np.zeros_like(pts)
        
        for step in range(n_steps):
            if elapsed() > DEADLINE - 20:
                break
                
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsive/attractive forces
            diff = pts[:, None, :] - pts[None, :, :]
            dist_sq = (diff ** 2).sum(axis=2)
            np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
            
            dist = np.sqrt(dist_sq)
            
            # Lennard-Jones-like force: F = 12*eps*sigma^12/r^13 - 6*eps*sigma^6/r^7
            # Simplified: use inverse power repulsion for stability
            # F = k / r^3 (direction along diff)
            
            # Only compute forces for pairs closer than 3*sigma to save time
            mask = dist < 3.0 * sigma
            mask = np.where(mask, 1.0, 0.0)
            np.fill_diagonal(mask, 0.0)
            
            # Force magnitude: k / r^2 (for 1/r^3 force, F = k/r^3 * r_hat = k/r^2 * diff/r)
            # Actually, for potential U = k/r^12, F = -dU/dr * r_hat = 12k/r^13 * r_hat
            # Let's use a simpler repulsive force: F = k / r^4 * r_hat (for U = k/r^3)
            
            # Use: F_ij = k * (sigma/r)^12 * r_hat - 2*k * (sigma/r)^6 * r_hat
            # But this can be unstable. Let's use pure repulsion: F = k / r^3 * r_hat
            
            k_rep = 10.0
            inv_dist = 1.0 / dist
            inv_dist_cubed = inv_dist ** 3
            inv_dist_sq = inv_dist ** 2
            
            # Force = k_rep / r^3 * r_hat = k_rep * diff / r^4
            force_mag = k_rep * mask * inv_dist_sq  # This gives k/r^2, multiply by diff/r for r_hat
            # Actually: F = k * diff / r^4 = k * diff * inv_dist^4
            inv_dist_fourth = inv_dist_sq ** 2
            force_vec = k_rep * mask * inv_dist_fourth[:, :, None] * diff
            forces += force_vec.sum(axis=1)
            
            # Confining force: push points back if they go outside unit circle
            radii = np.linalg.norm(pts, axis=1)
            outside = radii > 1.0
            if np.any(outside):
                # Force proportional to how far outside
                overshoot = (radii[outside] - 1.0)
                # Direction is inward (toward origin)
                force_dir = -pts[outside] / radii[outside, None]
                forces[outside] += confining_strength * overshoot[:, None] * force_dir
            
            # Velocity Verlet
            velocities += forces * dt
            velocities *= 0.999  # Damping
            pts += velocities * dt
            
            # Keep points inside unit circle (hard constraint)
            radii = np.linalg.norm(pts, axis=1)
            m = radii > 1.0
            if np.any(m):
                pts[m] = pts[m] / radii[m, None]
        
        return pts

    # Run MD simulation
    if elapsed() < DEADLINE - 30:
        md_pts = md_simulate(seed, n_steps=3000, dt=0.0005)
        s_md = score(md_pts)
        if s_md > best_s:
            best_s = s_md
            best_pts = md_pts

    # ---------- Local optimization refinement ----------
    # Use L-BFGS-B on the exact objective with the MD result as starting point
    x0 = best_pts.flatten()
    
    def objective(x):
        pts = x.reshape(N, 2)
        # Soft penalty for going outside
        rr = np.linalg.norm(pts, axis=1)
        pen = np.maximum(0.0, rr - 1.0).sum() ** 2
        s = score(pts)
        return -s + 1e-3 * pen

    if elapsed() < DEADLINE - 15:
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20},
            )
            if res.x is not None:
                cand_pts = res.x.reshape(N, 2)
                if np.all(np.isfinite(cand_pts)):
                    s = score(cand_pts)
                    if s > best_s:
                        best_s = s
                        best_pts = cand_pts
        except Exception:
            pass

    # ---------- Local hill-climbing polish ----------
    cur_pts = best_pts.copy()
    cur_s = best_s
    
    if elapsed() < DEADLINE - 5:
        step = 0.01
        stuck = 0
        while elapsed() < DEADLINE - 5:
            improved = False
            for _ in range(40):
                if elapsed() > DEADLINE - 5:
                    break
                p2 = cur_pts + np.random.normal(0, step, cur_pts.shape)
                # Keep inside unit circle
                rr = np.linalg.norm(p2, axis=1)
                m = rr > 1.0
                if np.any(m):
                    p2[m] = p2[m] / rr[m, None]
                s2 = score(p2)
                if s2 > cur_s + 1e-12:
                    cur_pts = p2
                    cur_s = s2
                    improved = True
                    if s2 > best_s:
                        best_s = s2
                        best_pts = cur_pts.copy()
            if improved:
                stuck = 0
                step = min(0.02, step * 1.05)
            else:
                stuck += 1
                step *= 0.85
                if step < 1e-5 or stuck > 20:
                    break

    # ---------- Final validity guard ----------
    final = best_pts
    if not np.all(np.isfinite(final)):
        final = seed
    rr = np.linalg.norm(final, axis=1)
    m = rr > 1.0
    if np.any(m):
        final[m] = final[m] / rr[m, None]

    return final
