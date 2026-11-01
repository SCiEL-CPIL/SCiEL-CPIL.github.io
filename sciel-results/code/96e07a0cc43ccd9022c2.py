"""Rewritten program: rigid-body dynamics (velocity-Verlet) with adaptive damping,
hexagonal-lattice initialization, strong short-range repulsion, and confining potential,
plus a final high-precision polish pass."""

import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _hex_lattice_init(n):
    """Generate n points on a hexagonal lattice inside the unit disk."""
    # Start with a fine hex lattice and pick the best n points
    spacing = 0.2
    while True:
        k = int(1.0 / spacing) + 3
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            # Sort by distance from center to prefer central points
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists, kind="stable")[:n]
            return points[idx].copy()
        spacing *= 0.95


def _ring_init(n):
    """Initialize points on concentric rings."""
    points = [[0.0, 0.0]]
    remaining = n - 1
    ring = 1
    while remaining > 0:
        # Number of points on this ring
        count = min(6 * ring, remaining)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        r = ring * 0.18  # approximate ring radius
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
        remaining -= count
        ring += 1
    return np.array(points[:n])


def _score(points):
    """Compute the packing score: min(min pair dist / 2, min(1 - |c_i|))."""
    n = len(points)
    if n < 2:
        return 0.0
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    # Distance to boundary
    min_boundary = np.inf
    for i in range(n):
        r = np.linalg.norm(points[i])
        if (1.0 - r) < min_boundary:
            min_boundary = 1.0 - r
    return min(min_pair / 2.0, min_boundary)


def _dynamics_step(points, velocities, dt, damping, repulsion_k, confine_k):
    """One velocity-Verlet step with adaptive damping."""
    n = len(points)
    forces = np.zeros_like(points)
    
    # Repulsive forces between pairs
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                # Random small perturbation
                diff = np.random.randn(2) * 1e-6
                d = np.linalg.norm(diff)
            # Strong short-range repulsion
            if d < 0.3:
                f = repulsion_k * (0.3 - d) / d
                forces[i] += f * diff
                forces[j] -= f * diff
    
    # Confining harmonic potential
    for i in range(n):
        r = np.linalg.norm(points[i])
        if r > 0.85:
            f = confine_k * (r - 0.85)
            forces[i] -= f * points[i] / r if r > 1e-10 else 0
    
    # Velocity Verlet
    new_velocities = velocities * damping + forces * dt
    new_points = points + new_velocities * dt
    
    # Clamp to unit disk
    for i in range(n):
        r = np.linalg.norm(new_points[i])
        if r > 0.999:
            new_points[i] = new_points[i] * 0.999 / r
    
    return new_points, new_velocities


def _polish(points):
    """Final high-precision polish using scipy minimize."""
    def objective(x):
        pts = x.reshape(-1, 2)
        # Smooth surrogate for min pair distance
        min_pair = np.inf
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_pair:
                    min_pair = d
        min_boundary = np.inf
        for i in range(len(pts)):
            r = np.linalg.norm(pts[i])
            if (1.0 - r) < min_boundary:
                min_boundary = 1.0 - r
        # Use a smooth min approximation
        eps = 1e-6
        return -min(min_pair / 2.0, min_boundary) + eps * np.sum(pts**2)
    
    x0 = points.flatten()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        return result.x.reshape(-1, 2)
    except Exception:
        return points


def construct():
    start_time = time.time()
    
    # Try multiple initializations and dynamics runs
    best_points = None
    best_score = -np.inf
    
    initializations = [_hex_lattice_init(N), _ring_init(N)]
    
    for init_idx, init_points in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        points = init_points.copy()
        velocities = np.zeros_like(points)
        
        # Adaptive parameters
        dt = 0.001
        damping = 0.95
        repulsion_k = 500.0
        confine_k = 100.0
        
        # Run dynamics
        for step in range(5000):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            # Adaptive damping: decrease over time
            damping = 0.95 * (0.999 ** step)
            
            points, velocities = _dynamics_step(points, velocities, dt, damping, repulsion_k, confine_k)
            
            # Check convergence
            if step % 500 == 0:
                current_score = _score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
        
        # Final score for this run
        current_score = _score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    # Polish the best result
    if best_points is not None and time.time() - start_time < DEADLINE - 10:
        polished = _polish(best_points)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points
