"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_radius_full(points):
    """
    Calculate the maximum common radius for the given points.
    Returns min(min_pair_dist/2, min_boundary_dist).
    """
    n = len(points)
    if n == 0:
        return 0.0
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist) / 2.0
    
    # Boundary distances
    norms = np.sqrt(np.sum(points * points, axis=1))
    min_boundary_dist = np.min(1.0 - norms)
    
    return min(min_pair_dist, min_boundary_dist)

def _get_radius_fast(points, changed_idx=None):
    """
    Fast radius computation. If changed_idx is given, only check that point
    against all others (assuming other pairwise distances haven't changed).
    """
    n = len(points)
    if n == 0:
        return 0.0
    
    if changed_idx is not None:
        # Only check distances from changed_idx to all others
        other = points[np.arange(n) != changed_idx]
        diffs = other - points[changed_idx]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        min_pair_dist = np.min(dists) / 2.0 if len(dists) > 0 else 1.0
        
        # Also need min over all pairs - but we can't get that from just one point
        # So we still need full computation for accuracy. 
        # However, for SA acceptance, we can use a lower bound.
        # Actually, let's just do full computation but optimized.
        pass
    
    # Optimized full computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist) / 2.0
    
    norms = np.sqrt(np.sum(points * points, axis=1))
    min_boundary_dist = np.min(1.0 - norms)
    
    return min(min_pair_dist, min_boundary_dist)

def _radius_with_cache(points):
    """Compute radius using vectorized operations."""
    n = len(points)
    if n < 2:
        norms = np.sqrt(np.sum(points * points, axis=1))
        return float(np.min(1.0 - norms))
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    norms = np.sqrt(np.sum(points * points, axis=1))
    min_bound = np.min(1.0 - norms)
    
    return float(min(min_pair, min_bound))

def _init_ring():
    """
    Initialize with concentric rings.
    """
    # Try different ring configurations
    best_pts = None
    best_r = -1
    
    # Configuration: 1 center + rings
    # For 50 points: 1 + 6 + 12 + 18 + 13 = 50 or similar
    ring_configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 19, 12],
        [1, 6, 13, 18, 12],
        [1, 7, 12, 18, 12],
        [1, 6, 12, 17, 14],
        [1, 6, 13, 17, 13],
        [2, 6, 12, 18, 12],
        [1, 6, 12, 18, 12, 1],
    ]
    
    for config in ring_configs:
        pts = []
        # Center
        if config[0] >= 1:
            pts.append([0.0, 0.0])
            config[0] -= 1
        
        radius = 0.15
        for ring_size in config:
            if ring_size <= 0:
                continue
            for i in range(ring_size):
                angle = 2.0 * np.pi * i / ring_size
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pts.append([x, y])
            radius += 0.15
        
        if len(pts) >= N:
            pts = pts[:N]
            pts = np.array(pts)
            r = _radius_with_cache(pts)
            if r > best_r:
                best_r = r
                best_pts = pts
        elif len(pts) > 0:
            # Pad with random points if needed
            while len(pts) < N:
                x = np.random.uniform(-0.9, 0.9)
                y = np.random.uniform(-0.9, 0.9)
                if x*x + y*y <= 0.81:
                    pts.append([x, y])
            pts = np.array(pts[:N])
            r = _radius_with_cache(pts)
            if r > best_r:
                best_r = r
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: hexagonal lattice
    return _init_lattice()

def _init_lattice():
    """
    Generate an initial configuration using a hexagonal lattice.
    """
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
            return points[np.argsort(radius[mask], kind="stable")[:N]].copy()
        spacing *= 0.95

def _init_random():
    """
    Generate a random initial configuration.
    """
    pts = []
    while len(pts) < N:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x*x + y*y <= 0.81:
            pts.append([x, y])
    return np.array(pts)

def _sa_optimize(points, max_iters=5000, t_start=1.0, t_end=1e-6, time_budget=30.0):
    """
    Simulated Annealing to maximize the minimum radius.
    """
    current_points = points.copy()
    current_r = _radius_with_cache(current_points)
    
    best_points = current_points.copy()
    best_r = current_r
    
    ratio = t_end / t_start
    start_time = time.time()
    
    for i in range(max_iters):
        # Time check
        if time.time() - start_time > time_budget:
            break
        
        t = t_start * (ratio ** (i / max_iters))
        
        idx = np.random.randint(N)
        
        # Adaptive step size
        step_size = max(0.02 * current_r, 0.005) * (0.5 + t)
        
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_points = current_points.copy()
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
        
        new_r = _radius_with_cache(new_points)
        
        if new_r > current_r:
            current_points = new_points
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_points = current_points.copy()
        else:
            if t > 1e-10:
                delta = (new_r - current_r) / t
                if delta > -50:  # Avoid underflow
                    if np.random.random() < np.exp(delta):
                        current_points = new_points
                        current_r = new_r
    
    return best_points, best_r

def _local_search(points, max_iters=3000, time_budget=15.0):
    """
    Hill climbing with adaptive step size.
    """
    current_points = points.copy()
    current_r = _radius_with_cache(current_points)
    
    best_points = current_points.copy()
    best_r = current_r
    
    step_size = 0.03
    no_improve_count = 0
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        for _ in range(8):
            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            
            new_points = current_points.copy()
            new_points[idx, 0] += dx
            new_points[idx, 1] += dy
            
            new_r = _radius_with_cache(new_points)
            
            if new_r > current_r:
                current_points = new_points
                current_r = new_r
                improved = True
                no_improve_count = 0
                
                if current_r > best_r:
                    best_r = current_r
                    best_points = current_points.copy()
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.85
            if step_size < 1e-6 or no_improve_count > 50:
                break
    
    return best_points, best_r

def _coordinate_descent(points, max_iters=2000, time_budget=20.0):
    """
    Coordinate-wise optimization for final polish.
    """
    current_points = points.copy()
    current_r = _radius_with_cache(current_points)
    best_points = current_points.copy()
    best_r = current_r
    
    start_time = time.time()
    step = 0.01
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        for idx in range(N):
            # Try moving in 8 directions
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = current_points.copy()
                    new_points[idx, 0] += dx
                    new_points[idx, 1] += dy
                    
                    new_r = _radius_with_cache(new_points)
                    if new_r > current_r:
                        current_points = new_points
                        current_r = new_r
                        improved = True
                        if current_r > best_r:
                            best_r = current_r
                            best_points = current_points.copy()
        
        if not improved:
            step *= 0.7
            if step < 1e-7:
                break
    
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_r = -1.0
    
    # 1. Start with ring initialization
    pts = _init_ring()
    best_pts, best_r_curr = _sa_optimize(pts, max_iters=4000, t_start=0.8, t_end=1e-6, time_budget=25.0)
    if best_r_curr > best_r:
        best_r = best_r_curr
        best_points = best_pts
    
    # Refine
    if time.time() < deadline - 30.0:
        refined_pts, refined_r = _local_search(best_points, max_iters=2000, time_budget=15.0)
        if refined_r > best_r:
            best_r = refined_r
            best_points = refined_pts
    
    # 2. Try multiple restarts with different initializations
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 40.0:
            break
        
        # Alternate between ring and random
        if i % 2 == 0:
            pts = _init_ring()
        else:
            pts = _init_random()
        
        # Decreasing time budget for later restarts
        remaining = deadline - time.time() - 30.0
        if remaining < 10.0:
            break
        tb = min(20.0, remaining / (num_restarts - i))
        
        best_pts, best_r_curr = _sa_optimize(pts, max_iters=3000, t_start=0.5, t_end=1e-6, time_budget=tb)
        
        if best_r_curr > best_r:
            best_r = best_r_curr
            best_points = best_pts
            
            # Quick local search on new best
            if time.time() < deadline - 20.0:
                refined_pts, refined_r = _local_search(best_points, max_iters=1000, time_budget=8.0)
                if refined_r > best_r:
                    best_r = refined_r
                    best_points = refined_pts

    # Final polish with coordinate descent
    if best_points is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 2.0
        final_pts, final_r = _coordinate_descent(best_points, max_iters=3000, time_budget=max(5.0, remaining))
        if final_r > best_r:
            best_r = final_r
            best_points = final_pts
    
    # One more local search pass if time permits
    if best_points is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 0.5
        final_pts, final_r = _local_search(best_points, max_iters=1000, time_budget=max(2.0, remaining))
        if final_r > best_r:
            best_r = final_r
            best_points = final_pts

    if best_points is None:
        best_points = _init_lattice()
        
    return best_points

# EVOLVE-BLOCK-END
