"""Rewritten program for circle packing in unit square (N=26).
Uses a physics-based relaxation with Delaunay-based constraint reduction,
multiple random restarts, and a deadline guard."""
import numpy as np
import time
from scipy.spatial import Delaunay

N = 26
DEADLINE = 135.0  # safety margin before 150s kill

def _elapsed(t0):
    return time.time() - t0

def _remaining(t0):
    return DEADLINE - _elapsed(t0)

def _validity(points):
    """Check if points form valid circles in [0,1]^2 with disjoint interiors."""
    n = len(points)
    for i in range(n):
        x, y, r = points[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = points[i]
            xj, yj, rj = points[j]
            dist = np.hypot(xi - xj, yi - yj)
            if dist < ri + rj - 1e-7:
                return False
    return True

def _score(points):
    """Return sum of radii if valid, else -inf."""
    if not _validity(points):
        return -np.inf
    return np.sum(points[:, 2])

def _delaunay_neighbors(points):
    """Get neighbor pairs from Delaunay triangulation of circle centers."""
    centers = points[:, :2]
    try:
        tri = Delaunay(centers)
        pairs = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    pairs.add((min(a, b), max(a, b)))
        return list(pairs)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _relax(points, t0, max_time=10.0):
    """Physics-based relaxation: push overlapping circles apart, grow radii."""
    points = points.copy()
    n = len(points)
    
    # Initial radii: use a reasonable starting value
    # For 26 circles in unit square, average radius ~ 1/sqrt(26*pi) ~ 0.11
    # But we want to maximize sum, so start small and grow
    min_r = 0.01
    max_r = 0.25
    
    # Set initial radii based on position (smaller near corners/edges)
    for i in range(n):
        x, y = points[i, 0], points[i, 1]
        # Distance to nearest wall
        d_wall = min(x, 1 - x, y, 1 - y)
        # Cap radius by wall distance
        points[i, 2] = min(max_r, max(min_r, d_wall * 0.9))
    
    # Precompute all pairs for fallback
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Simulation parameters
    dt = 0.01
    damping = 0.9
    force_scale = 1.0
    grow_rate = 0.001
    
    velocities = np.zeros((n, 2))
    best_score = -np.inf
    best_points = points.copy()
    
    max_iter = int(max_time / dt)
    for it in range(max_iter):
        if _elapsed(t0) > t0 + max_time:
            break
        
        # Get active constraints (Delaunay neighbors + boundary)
        active_pairs = _delaunay_neighbors(points)
        
        # Forces
        forces = np.zeros((n, 2))
        
        # Pairwise repulsion for active pairs
        for i, j in active_pairs:
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            dist = np.hypot(dx, dy)
            if dist < 1e-12:
                # Random small perturbation
                dx, dy = np.random.uniform(-0.01, 0.01, 2)
                dist = np.hypot(dx, dy)
            overlap = points[i, 2] + points[j, 2] - dist
            if overlap > 0:
                # Repulsive force proportional to overlap
                f = force_scale * overlap / dist
                fx = f * dx
                fy = f * dy
                forces[i, 0] -= fx
                forces[i, 1] -= fy
                forces[j, 0] += fx
                forces[j, 1] += fy
        
        # Boundary forces
        for i in range(n):
            x, y, r = points[i]
            # Left wall
            if x - r < 0:
                forces[i, 0] += force_scale * (r - x)
            # Right wall
            if x + r > 1:
                forces[i, 0] -= force_scale * (x + r - 1)
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += force_scale * (r - y)
            # Top wall
            if y + r > 1:
                forces[i, 1] -= force_scale * (y + r - 1)
        
        # Update velocities and positions
        velocities = damping * velocities + dt * forces
        points[:, 0] += dt * velocities[:, 0]
        points[:, 1] += dt * velocities[:, 1]
        
        # Clamp to unit square (with radius margin)
        for i in range(n):
            r = points[i, 2]
            points[i, 0] = np.clip(points[i, 0], r, 1 - r)
            points[i, 1] = np.clip(points[i, 1], r, 1 - r)
        
        # Grow radii slightly
        for i in range(n):
            # Max radius allowed by walls
            max_allowed = min(points[i, 0], 1 - points[i, 0], points[i, 1], 1 - points[i, 1])
            points[i, 2] = min(points[i, 2] + grow_rate, max_allowed)
        
        # Check score periodically
        if it % 100 == 0:
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
    
    # Final check
    s = _score(points)
    if s > best_score:
        best_score = s
        best_points = points.copy()
    
    return best_points, best_score

def _hex_initial():
    """Generate a hexagonal packing initial configuration."""
    points = []
    # Hexagonal lattice
    r = 0.08
    dx = 2 * r * np.sqrt(3) / 2  # horizontal spacing
    dy = 2 * r * 0.5  # vertical spacing between rows
    
    row = 0
    while len(points) < N:
        col = 0
        offset = 0 if row % 2 == 0 else dx / 2
        while len(points) < N:
            x = offset + col * dx
            y = row * 2 * r
            if x < 1 - r and y < 1 - r:
                points.append([x, y, r])
            col += 1
        row += 1
    
    return np.array(points[:N])

def _random_initial():
    """Generate a random initial configuration."""
    points = []
    r_base = 0.05
    for _ in range(N):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = r_base * np.random.uniform(0.5, 1.5)
        # Ensure within bounds
        r = min(r, x, 1 - x, y, 1 - y)
        points.append([x, y, max(r, 0.01)])
    return np.array(points)

def construct():
    t0 = time.time()
    
    best_points = None
    best_score = -np.inf
    
    # Try hexagonal initial
    if _remaining(t0) > 5:
        pts = _hex_initial()
        pts, s = _relax(pts, t0, max_time=min(8.0, _remaining(t0) - 2))
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Try multiple random restarts
    n_restarts = 20
    for restart in range(n_restarts):
        if _remaining(t0) < 5:
            break
        
        pts = _random_initial()
        # Jitter the initial positions slightly
        pts[:, 0] += np.random.uniform(-0.02, 0.02, N)
        pts[:, 1] += np.random.uniform(-0.02, 0.02, N)
        
        max_time = min(5.0, _remaining(t0) - 2)
        if max_time < 1.0:
            break
            
        pts, s = _relax(pts, t0, max_time=max_time)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # If no valid solution found, return a trivial valid one
    if best_points is None or not _validity(best_points):
        # Fallback: small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.4 / side
                cells.append([x, y, r])
        best_points = np.array(cells[:N])
    
    return best_points
