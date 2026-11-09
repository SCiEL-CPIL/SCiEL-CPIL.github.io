"""Improved packer: hex-lattice init + vectorized force refinement + PSO + L-BFGS-B polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # seconds; leaves 15s margin under 150s kill


def _score(pts):
    """Compute the packing radius r for centres pts (N,2)."""
    n = len(pts)
    if n == 0:
        return 0.0
    # distance to boundary
    d = 1.0 - np.linalg.norm(pts, axis=1)
    r_bound = np.min(d)
    # pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_bound, r_pair)


def _force_refine(pts, max_iter=20000, tol=1e-12):
    """
    Vectorized force-based local refinement.
    Points repel each other when closer than 2r and are pulled inward when too close to boundary.
    """
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    r = _score(pts)
    k = 5.0
    best_r = r
    best_pts = pts.copy()

    for it in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break

        forces = np.zeros_like(pts)

        # Vectorized pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, 1.0)  # avoid div by zero

        # Only apply force where dist < 2r
        mask = dist < 2.0 * r
        # Force magnitude: k * (2r - dist) / dist, applied along direction
        force_mag = np.zeros_like(dist)
        force_mag[mask] = k * (2.0 * r - dist[mask]) / dist[mask]
        # Sum forces: for each i, sum over j of force_mag[i,j] * diff[i,j]
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)

        # Boundary constraint: if |p| > 1 - r, push INWARD (toward origin)
        norms = np.linalg.norm(pts, axis=1)
        boundary_mask = norms > (1.0 - r)
        if np.any(boundary_mask):
            excess = norms[boundary_mask] - (1.0 - r)
            safe_norms = np.where(norms[boundary_mask] > 1e-10, norms[boundary_mask], 1.0)
            forces[boundary_mask] -= k * 2.0 * excess / safe_norms * pts[boundary_mask]

        # Adaptive step size with line search
        step = 0.02
        improved = False
        for _ in range(15):
            pts_new = pts + step * forces

            # Clip to ensure inside container
            norms_new = np.linalg.norm(pts_new, axis=1)
            max_norm = 1.0 - r * 0.9
            mask_out = norms_new > max_norm
            if np.any(mask_out):
                safe_norms_new = np.where(norms_new[mask_out] > 1e-10, norms_new[mask_out], 1.0)
                pts_new[mask_out] *= max_norm / safe_norms_new

            new_r = _score(pts_new)
            if new_r >= r + 1e-12:
                improved = True
                break
            elif new_r >= r - 1e-10:
                # Accept small decrease to escape plateaus
                improved = True
                break
            step *= 0.5

        if not improved:
            break

        if abs(new_r - r) < tol:
            break

        pts = pts_new
        r = new_r
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts


def _psO_optimize(pts, n_particles=20, max_iter=50):
    """Particle Swarm Optimization to escape local minima."""
    if time.time() > DEADLINE - 20.0:
        return pts
    
    pts = pts.copy().astype(np.float64)
    best_r = _score(pts)
    best_pts = pts.copy()
    
    # Initialize particles
    particles = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    pbests = np.zeros((n_particles, N, 2))
    pbest_scores = np.zeros(n_particles)
    
    for i in range(n_particles):
        particles[i] = pts + 0.01 * np.random.randn(N, 2)
        # Ensure inside container
        norms = np.linalg.norm(particles[i], axis=1)
        mask = norms > 0.99
        if np.any(mask):
            particles[i][mask] *= 0.99 / norms[mask]
        pbests[i] = particles[i].copy()
        pbest_scores[i] = _score(particles[i])
    
    gbest = best_pts.copy()
    gbest_score = best_r
    
    w = 0.9  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10.0:
            break
        
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.rand(N, 2)
            r2 = np.random.rand(N, 2)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbests[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            
            # Clamp velocity
            v_max = 0.05
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            # Update position
            particles[i] += velocities[i]
            
            # Ensure inside container
            norms = np.linalg.norm(particles[i], axis=1)
            mask = norms > 0.99
            if np.any(mask):
                particles[i][mask] *= 0.99 / norms[mask]
            
            # Update personal best
            score_i = _score(particles[i])
            if score_i > pbest_scores[i]:
                pbest_scores[i] = score_i
                pbests[i] = particles[i].copy()
                if score_i > gbest_score:
                    gbest_score = score_i
                    gbest = particles[i].copy()
        
        # Update global best
        if gbest_score > best_r:
            best_r = gbest_score
            best_pts = gbest.copy()
    
    return best_pts


def _lbfgs_polish(pts):
    """Use L-BFGS-B to maximize the minimum distance (packing radius)."""
    def neg_score(x):
        p = x.reshape(-1, 2)
        return -_score(p)

    # Constraints: each point must be inside unit circle (|p| <= 1)
    # Use tighter bounds based on current radius
    r = _score(pts)
    margin = 1.0 - r
    bounds = [(-margin, margin)] * (2 * N)

    best_pts = pts.copy()
    best_r = r

    # Multiple L-BFGS-B runs from slightly perturbed starts
    for attempt in range(3):
        if time.time() > DEADLINE - 5.0:
            break

        if attempt == 0:
            x0 = pts.flatten()
        else:
            x0 = (pts + 0.005 * np.random.randn(N, 2)).flatten()
            # Ensure within bounds
            norms = np.linalg.norm(x0.reshape(-1, 2), axis=1)
            mask = norms > margin
            if np.any(mask):
                safe = np.where(norms[mask] > 1e-10, norms[mask], 1.0)
                x0.reshape(-1, 2)[mask] *= margin / safe

        try:
            result = minimize(neg_score, x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
            new_pts = result.x.reshape(-1, 2)
            new_r = _score(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        except Exception:
            pass

    return best_pts


def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, pick best N."""
    best_pts = None
    best_r = 0.0
    for spacing in np.linspace(0.1, 0.25, 30):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            idx = np.argsort(np.hypot(points[:, 0], points[:, 1]), kind="stable")[:N]
            pts = points[idx]
            r = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    return best_pts, best_r


def construct():
    t0 = time.time()
    # Start with hex lattice
    pts, r = _hex_lattice_init()
    if pts is None:
        pts = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            pts[i] = [0.9 * np.cos(angle), 0.9 * np.sin(angle)]
        r = _score(pts)

    # Force-based refinement (vectorized, faster)
    if time.time() < DEADLINE - 40.0:
        pts = _force_refine(pts, max_iter=30000, tol=1e-13)

    # PSO to escape local minima
    if time.time() < DEADLINE - 30.0:
        pts_pso = _psO_optimize(pts, n_particles=15, max_iter=40)
        if _score(pts_pso) > _score(pts):
            pts = pts_pso

    # L-BFGS-B polish for finer convergence
    if time.time() < DEADLINE - 15.0:
        pts = _lbfgs_polish(pts)

    # Final force refinement pass
    if time.time() < DEADLINE - 5.0:
        pts = _force_refine(pts, max_iter=15000, tol=1e-14)

    # Final L-BFGS polish
    if time.time() < DEADLINE - 3.0:
        pts = _lbfgs_polish(pts)

    # Ensure validity: clip any point outside
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        safe_norms = np.where(norms[mask] > 1e-10, norms[mask], 1.0)
        pts[mask] /= safe_norms[:, np.newaxis] * 1.0001

    return pts
