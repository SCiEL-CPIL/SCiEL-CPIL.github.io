"""Improved packing: dynamics-based optimization with wall particles."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _score(pts):
    """Compute the packing score (minimum radius)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check all points are within unit circle
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_pair = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists_pair, np.inf)
    min_pair = np.min(dists_pair) / 2.0
    min_boundary = np.min(1.0 - dists)
    return min(min_pair, min_boundary)


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(N, 2)
    s = _score(pts)
    return -s


def _wall_particle_init(n_wall=200):
    """Initialize wall particles on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n_wall, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _dynamics_optimize(pts0, n_wall=200, steps=5000, dt=0.01, wall_strength=10.0,
                       circle_strength=5.0, cooling=0.999, deadline=None):
    """
    Rigid-body dynamics with wall particles.
    Circles repel each other and are repelled by wall particles.
    """
    pts = pts0.copy().astype(np.float64)
    walls = _wall_particle_init(n_wall)
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Velocity
    vel = np.zeros_like(pts)
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Circle-circle repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Repulsive force: proportional to 1/dist^2 when close
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 1.0:
                    # Strong repulsion
                    f = circle_strength / (d * d + 1e-6)
                    forces[i] += f * diff[i, j] / d
                    forces[j] -= f * diff[i, j] / d
        
        # Wall particle repulsion
        for i in range(N):
            for j in range(n_wall):
                diff_w = pts[i] - walls[j]
                d_w = np.sqrt(np.sum(diff_w ** 2) + 1e-12)
                if d_w < 0.15:
                    f_w = wall_strength / (d_w * d_w + 1e-6)
                    forces[i] += f_w * diff_w / d_w
        
        # Update velocity with damping
        vel = vel * 0.95 + forces * dt
        pts = pts + vel * dt
        
        # Cool down
        dt *= cooling
        
        # Evaluate score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    return best_pts, best_score


def _local_optimize(pts0, deadline=None):
    """Local optimization using Nelder-Mead on the score function."""
    x0 = pts0.flatten()
    
    def obj(x):
        if deadline is not None and time.time() > deadline:
            return 0.0
        return _neg_score(x)
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        if res.success or res.fun < _neg_score(x0):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return pts0


def _hex_lattice_init(n=N):
    """Generate a hexagonal lattice initial configuration."""
    spacing = 0.15
    points = []
    k = int(2.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 0.9:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fill in with random points
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.9)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    # Take the n points closest to center
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists)[:n]
    return points[idx]


def _ring_init(n=N):
    """Initialize points on concentric rings."""
    points = [[0.0, 0.0]]
    remaining = n - 1
    ring_radius = 0.25
    while remaining > 0:
        # How many points fit on this ring?
        circumference = 2 * np.pi * ring_radius
        n_ring = min(remaining, max(2, int(circumference / 0.25)))
        for k in range(n_ring):
            angle = 2 * np.pi * k / n_ring
            points.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
            remaining -= 1
            if remaining <= 0:
                break
        ring_radius += 0.2
    return np.array(points[:n])


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Try multiple initializations
    inits = [
        ('hex', _hex_lattice_init()),
        ('ring', _ring_init()),
    ]
    
    # Add a few random inits
    for _ in range(3):
        pts = np.random.uniform(-0.8, 0.8, (N, 2))
        # Project inside unit circle
        dists = np.hypot(pts[:, 0], pts[:, 1])
        mask = dists > 0.9
        pts[mask] = pts[mask] * 0.9 / dists[mask, np.newaxis]
        inits.append(('random', pts))
    
    best_pts = inits[0][1]
    best_score = _score(best_pts)
    
    # Phase 1: Dynamics optimization
    for name, init_pts in inits:
        if time.time() > deadline - 20:
            break
        pts, score = _dynamics_optimize(init_pts, n_wall=150, steps=3000,
                                        dt=0.02, cooling=0.9995, deadline=deadline)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Phase 2: Local optimization (Nelder-Mead)
    if time.time() < deadline - 10:
        pts = _local_optimize(best_pts, deadline=deadline)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Phase 3: Final polish with tighter dynamics
    if time.time() < deadline - 5:
        pts, score = _dynamics_optimize(best_pts, n_wall=300, steps=2000,
                                        dt=0.005, cooling=0.9999, deadline=deadline)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Final local polish
    if time.time() < deadline - 2:
        pts = _local_optimize(best_pts, deadline=deadline)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Ensure validity
    dists = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(dists > 1.0):
        # Scale down to fit
        max_d = np.max(dists)
        best_pts = best_pts * (0.999 / max_d)
    
    return best_pts
