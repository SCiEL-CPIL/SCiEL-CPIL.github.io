"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0
    
    # Min distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    # Min pair distance
    min_pair = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    
    return min(min_pair / 2.0, min_boundary)

def _local_optimize(points, max_iter=500, lr=0.001):
    """Local optimization using gradient-free approach to maximize minimum distance."""
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Use a simple iterative repulsion approach
    for iteration in range(max_iter):
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i+1, n):
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(2) * 1e-10
                    dist = np.linalg.norm(d)
                # Repel if too close
                if dist < 0.15:
                    force = (0.15 - dist) / dist * d
                    forces[i] += force
                    forces[j] -= force
        
        # Attraction to center (to keep points inside)
        for i in range(n):
            r = np.linalg.norm(points[i])
            if r > 0.95:
                forces[i] -= 0.1 * points[i] / r
        
        # Update positions
        points = points + lr * forces
        
        # Ensure points stay inside unit circle
        for i in range(n):
            r = np.linalg.norm(points[i])
            if r > 1.0:
                points[i] = points[i] * (1.0 / r)
        
        # Compute score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Adaptive learning rate
        lr *= 0.999
    
    return best_points

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    # Try multiple hexagonal lattice configurations
    best_points = None
    best_score = -np.inf
    
    # Different lattice orientations and densities
    for angle_deg in [0, 30, 60, 90]:
        angle = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Try different spacings
        for spacing_factor in [0.95, 0.97, 0.99, 1.0, 1.02]:
            spacing = 2.0 * spacing_factor / (np.sqrt(N) + 1.0)
            
            k = int(1.0 / spacing) + 3
            rows = []
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    # Rotate
                    xr = x * cos_a - y * sin_a
                    yr = x * sin_a + y * cos_a
                    rows.append((xr, yr))
            
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            
            if len(points) < N:
                continue
            
            # Sort by distance from center and take N closest
            radius = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radius, kind="stable")[:N]
            points = points[idx]
            
            # Compute score
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Local optimization if we have time
    if best_points is not None and time.time() < deadline - 20:
        optimized = _local_optimize(best_points, max_iter=200, lr=0.0005)
        opt_score = _compute_score(optimized)
        if opt_score > best_score:
            best_points = optimized
            best_score = opt_score
    
    # Final validation
    if best_points is None:
        # Fallback: simple grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            best_points = points[np.argsort(np.hypot(points[:, 0], points[:, 1]), kind="stable")[:N]]
        else:
            # Emergency fallback
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = np.linspace(0, 0.9, N)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
