"""Optimized 50-circle packing using physics-inspired relaxation and grid-based neighbor lookup."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def _build_grid(points, cell_size):
    """Build a spatial grid for fast neighbor lookup."""
    grid = {}
    for i in range(len(points)):
        x, y = points[i]
        ix = int(np.floor(x / cell_size))
        iy = int(np.floor(y / cell_size))
        key = (ix, iy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid

def _relax(points, r, max_iters=500, tol=1e-6, seed=42):
    """
    Relax the configuration to satisfy packing constraints for radius r.
    Returns (points, max_violation, success).
    """
    rng = np.random.RandomState(seed)
    points = points.copy()
    velocities = np.zeros_like(points)
    
    # Grid cell size should be at least 2*r to catch all potential collisions
    cell_size = 2.0 * r + 0.05
    
    for it in range(max_iters):
        grid = _build_grid(points, cell_size)
        
        forces = np.zeros_like(points)
        max_violation = 0.0
        
        for i in range(N):
            x, y = points[i]
            ix = int(np.floor(x / cell_size))
            iy = int(np.floor(y / cell_size))
            
            # Boundary force
            r_i = np.hypot(x, y)
            margin = 1.0 - r
            if r_i > margin:
                violation = r_i - margin
                if violation > max_violation:
                    max_violation = violation
                # Push towards center
                if r_i > 1e-8:
                    forces[i, 0] -= (violation / r_i) * x
                    forces[i, 1] -= (violation / r_i) * y
                else:
                    angle = rng.uniform(0, 2*np.pi)
                    forces[i, 0] -= np.cos(angle) * violation
                    forces[i, 1] -= np.sin(angle) * violation
            
            # Neighbor forces
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = (ix + dx, iy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if j <= i:
                                continue
                            x_j, y_j = points[j]
                            dx_ij = x - x_j
                            dy_ij = y - y_j
                            d = np.hypot(dx_ij, dy_ij)
                            
                            min_dist = 2.0 * r
                            if d < min_dist:
                                violation = min_dist - d
                                if violation > max_violation:
                                    max_violation = violation
                                if d > 1e-8:
                                    fx = (violation / d) * dx_ij
                                    fy = (violation / d) * dy_ij
                                    forces[i, 0] += fx
                                    forces[i, 1] += fy
                                    forces[j, 0] -= fx
                                    forces[j, 1] -= fy
                                else:
                                    angle = rng.uniform(0, 2*np.pi)
                                    fx = np.cos(angle) * violation
                                    fy = np.sin(angle) * violation
                                    forces[i, 0] += fx
                                    forces[i, 1] += fy
                                    forces[j, 0] -= fx
                                    forces[j, 1] -= fy
        
        # Update velocities and positions
        dt = 0.05
        damping = 0.9
        velocities = (velocities + forces * dt) * damping
        points += velocities * dt
        
        # Clamp to container to prevent flying out
        radii = np.hypot(points[:, 0], points[:, 1])
        max_allowed = 1.0
        mask = radii > max_allowed
        if np.any(mask):
            points[mask] *= max_allowed / radii[mask]
        
        if max_violation < tol:
            return points, max_violation, True
            
    return points, max_violation, False

def _hexagonal_init(spacing=0.25):
    """Generate hexagonal lattice initialization."""
    points = []
    r_max = 1.0
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= r_max - spacing/2:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        points = points[idx]
    elif len(points) < N:
        rng = np.random.RandomState(42)
        while len(points) < N:
            r = rng.uniform(0, 0.9)
            a = rng.uniform(0, 2*np.pi)
            points = np.vstack([points, [r*np.cos(a), r*np.sin(a)]])
    return points[:N]

def construct():
    """
    Construct the optimal packing of 50 equal circles in the unit disk.
    """
    best_points = None
    best_radius = 0.0
    
    inits = [
        _hexagonal_init(0.25),
        _hexagonal_init(0.26),
        np.random.RandomState(1).uniform(-0.8, 0.8, (N, 2)),
        np.random.RandomState(2).uniform(-0.8, 0.8, (N, 2)),
    ]
    
    for init in inits:
        r = 0.05
        current_points = init.copy()
        
        while r < 0.13:
            pts, viol, success = _relax(current_points, r, max_iters=200, tol=1e-5)
            
            if success:
                current_points = pts
                r += 0.001
            else:
                r -= 0.0005
                if r < 0.05:
                    break
                pts, viol, success = _relax(current_points, r, max_iters=200, tol=1e-5)
                if success:
                    current_points = pts
                    r += 0.001
                else:
                    break
        
        # Calculate actual radius
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(current_points[i,0]-current_points[j,0], current_points[i,1]-current_points[j,1])
                if d < min_dist: min_dist = d
        min_bound = np.min(1.0 - np.hypot(current_points[:,0], current_points[:,1]))
        actual_r = min(min_dist/2.0, min_bound)
        
        if actual_r > best_radius:
            best_radius = actual_r
            best_points = current_points.copy()
            
    if best_points is None:
        # Fallback
        best_points = np.zeros((N, 2))
        
    return best_points
# EVOLVE-BLOCK-END
