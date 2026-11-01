"""Fifty equal circles in a circle – strong local-search / annealing approach."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # seconds; evaluator kills at 150 s


def _radius_from_points(pts):
    """Return the largest common radius r for the given centres."""
    if len(pts) < N:
        return 0.0
    # pairwise distances
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    # distance to boundary
    min_boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1)).max()
    r = min(min_pair, min_boundary)
    if not np.isfinite(r) or r <= 0:
        return 0.0
    return float(r)


def _energy(pts, beta=50.0):
    """
    Smoothed energy: we want to MAXIMIZE r.
    r = min( min_{i<j} d_ij/2, min_i (1 - |c_i|) )
    We minimize -log-sum-exp style or a soft-min.
    Use: E = -softmin( all pairwise half-distances, all boundary distances )
    softmin of a set of values v_k is approx: - (1/beta) * log(sum(exp(-beta * v_k)))
    So E = -softmin = (1/beta)*log(sum(exp(-beta*v_k)))
    We want to minimize E.
    """
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    pair_vals = d / 2.0
    boundary_vals = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    # combine
    vals = np.concatenate([pair_vals.ravel(), boundary_vals.ravel()])
    # filter out inf
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1e6
    # softmin
    m = vals.min()
    if m < -100:
        return 1e6
    logsum = np.log(np.sum(np.exp(-beta * (vals - m))))
    E = (m - np.log(np.sum(np.exp(-beta * (vals - m)))) ) / beta
    # Actually softmin(v) = - (1/beta) log sum exp(-beta v)
    # We want to minimize -softmin, i.e., maximize softmin.
    # Let's just return -softmin as the energy to minimize.
    softmin_val = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (vals - m))))
    return -softmin_val


def _grad_energy(pts, beta=50.0):
    """Numerical gradient of energy."""
    eps = 1e-5
    E0 = _energy(pts, beta)
    g = np.zeros_like(pts)
    for i in range(len(pts)):
        for j in range(2):
            pts[i, j] += eps
            Ep = _energy(pts, beta)
            pts[i, j] -= 2 * eps
            Em = _energy(pts, beta)
            pts[i, j] += eps
            g[i, j] = (Ep - Em) / (2 * eps)
    return E0, g


def _hex_init():
    """Hexagonal lattice initialization."""
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # pick N points closest to center
            idx = np.argsort(radius)
            return points[idx[:N]].copy()
        spacing *= 0.98


def _ring_init():
    """Ring structure initialization."""
    pts = np.zeros((N, 2))
    # Place in concentric rings
    # Ring 0: 1 point at center
    pts[0] = [0, 0]
    remaining = N - 1
    ring_radius = 0.18
    ring_index = 1
    count = 0
    while count < remaining:
        # number of points on this ring
        n_ring = min(remaining - count, 6 * ring_index)
        angles = 2 * np.pi * np.arange(n_ring) / n_ring
        for i in range(n_ring):
            idx = count + 1
            if idx >= N:
                break
            pts[idx] = [ring_radius * np.cos(angles[i]), ring_radius * np.sin(angles[i])]
            count += 1
        remaining -= n_ring
        ring_radius += 0.16
        ring_index += 1
    return pts


def _random_init():
    """Random initialization inside unit disk."""
    angles = 2 * np.pi * np.random.rand(N)
    radii = np.sqrt(np.random.rand(N)) * 0.95
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _force_directed(pts, max_iters=2000, step=0.01):
    """Simple force-directed relaxation to push overlapping circles apart."""
    pts = pts.copy()
    for it in range(max_iters):
        forces = np.zeros_like(pts)
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        # repulsive force between pairs
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * 0.12:
                    diff = pts[i] - pts[j]
                    norm = d[i, j]
                    if norm > 1e-10:
                        f = diff / norm * (2 * 0.12 - d[i, j])
                        forces[i] += f
                        forces[j] -= f
        # attractive force toward boundary (push outward)
        for i in range(N):
            r = np.sqrt(np.sum(pts[i] ** 2))
            if r < 0.95:
                if r > 1e-10:
                    forces[i] += pts[i] / r * 0.01 * (0.95 - r)
        # apply forces
        pts += step * forces
        # clip to unit disk
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        mask = radii > 0.99
        if np.any(mask):
            pts[mask] *= 0.99 / radii[mask]
    return pts


def _lbfgs_refine(pts, beta=50.0, maxiter=200):
    """L-BFGS-B refinement of the packing."""
    def objective(x):
        p = x.reshape(N, 2)
        return _energy(p, beta)
    
    res = minimize(objective, pts.ravel(), method='L-BFGS-B', maxiter=maxiter,
                   options={'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
    if res.success or res.fun < _energy(pts, beta):
        return res.x.reshape(N, 2)
    return pts


def construct():
    start_time = time.time()
    
    # Generate multiple initial candidates
    inits = []
    inits.append(_hex_init())
    inits.append(_ring_init())
    for _ in range(8):
        inits.append(_random_init())
    
    best_pts = None
    best_r = 0.0
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Force-directed relaxation
        pts_fd = _force_directed(init, max_iters=500, step=0.02)
        r_fd = _radius_from_points(pts_fd)
        
        # L-BFGS refinement with adaptive beta
        pts_opt = pts_fd.copy()
        for beta in [30, 50, 80, 120]:
            if time.time() - start_time > DEADLINE - 5:
                break
            pts_opt = _lbfgs_refine(pts_opt, beta=beta, maxiter=150)
            r_opt = _radius_from_points(pts_opt)
            if r_opt > r_fd:
                r_fd = r_opt
                pts_fd = pts_opt
        
        if r_fd > best_r:
            best_r = r_fd
            best_pts = pts_fd.copy()
    
    # Final fine-tuning with high beta
    if best_pts is not None and time.time() - start_time < DEADLINE - 3:
        for beta in [150, 200, 300]:
            if time.time() - start_time > DEADLINE - 2:
                break
            best_pts = _lbfgs_refine(best_pts, beta=beta, maxiter=100)
            r_new = _radius_from_points(best_pts)
            if r_new <= best_r:
                break
            best_r = r_new
    
    # Ensure validity: all points inside unit disk, finite values
    if best_pts is None:
        best_pts = _hex_init()
    
    # Final safety check
    radii = np.sqrt(np.sum(best_pts ** 2, axis=1))
    if np.any(radii > 1.0):
        best_pts *= (0.999 / radii.max())
    
    # Verify shape
    if best_pts.shape != (N, 2) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init()
    
    return best_pts
