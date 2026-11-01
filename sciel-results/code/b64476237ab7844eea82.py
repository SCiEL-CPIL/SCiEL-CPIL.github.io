"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the packing radius for given centers."""
    if len(centers) != N:
        return 0.0
    # Check finiteness
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside unit circle
    dists_center = np.linalg.norm(centers, axis=1)
    if np.any(dists_center > 1.0):
        return 0.0
    # Min distance to boundary
    min_boundary = 1.0 - np.max(dists_center)
    if min_boundary <= 0:
        return 0.0
    # Min pairwise distance
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    min_pair_half = min_pair / 2.0
    if min_pair_half <= 0:
        return 0.0
    return min(min_pair_half, min_boundary)


def _force_directed_step(centers, r_target, step_size):
    """One step of force-directed layout to maximize packing radius."""
    n = len(centers)
    forces = np.zeros_like(centers)
    
    # Repulsive forces between pairs
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                # Random small perturbation
                diff = np.array([np.random.randn(), np.random.randn()]) * 1e-6
                d = np.linalg.norm(diff)
            # Force magnitude: stronger when closer
            f_mag = 1.0 / (d * d)
            f = f_mag * diff / d
            forces[i] += f
            forces[j] -= f
    
    # Force pushing circles away from boundary (if too close)
    for i in range(n):
        d = np.linalg.norm(centers[i])
        if d > 0.001:
            # If circle center is close to boundary, push inward
            margin = 1.0 - d
            if margin < r_target:
                f_mag = (r_target - margin) * 10.0
                f = -f_mag * centers[i] / d
                forces[i] += f
    
    # Update positions
    new_centers = centers + step_size * forces
    # Project back inside unit circle
    for i in range(n):
        d = np.linalg.norm(new_centers[i])
        if d > 1.0:
            new_centers[i] = new_centers[i] / d
    
    return new_centers


def _optimize(centers, max_iter=5000, deadline=None):
    """Optimize packing using force-directed method."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Estimate current radius
    r_current = best_score
    
    # Adaptive step size
    step_size = 0.01
    base_step = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        new_centers = _force_directed_step(centers, r_current, step_size)
        new_score = _score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            r_current = new_score
            # Increase step size slightly on improvement
            step_size = min(step_size * 1.05, base_step * 2.0)
        else:
            # Decrease step size
            step_size = max(step_size * 0.95, base_step * 0.1)
        
        centers = new_centers
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    best_centers = None
    best_score = 0.0
    
    # Multiple restarts with different initial configurations
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Different initializations
        if restart == 0:
            # Hexagonal grid
            spacing = 0.2
            points = []
            for j in range(-10, 11):
                for i in range(-10, 11):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) <= 0.95:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= N:
                # Select N points closest to center for stability
                dists = np.linalg.norm(points, axis=1)
                idx = np.argsort(dists)[:N]
                centers = points[idx].copy()
            else:
                centers = np.random.randn(N, 2) * 0.8
        elif restart == 1:
            # Random uniform in disk
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 0.8, N))
            centers = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif restart == 2:
            # Ring-based initialization
            centers = np.zeros((N, 2))
            # Center circle
            centers[0] = [0, 0]
            # First ring: 6 circles
            for i in range(1, 7):
                angle = 2 * np.pi * i / 6
                centers[i] = [0.2 * np.cos(angle), 0.2 * np.sin(angle)]
            # Second ring: 12 circles
            for i in range(7, 19):
                angle = 2 * np.pi * (i - 7) / 12
                centers[i] = [0.4 * np.cos(angle), 0.4 * np.sin(angle)]
            # Third ring: 18 circles
            for i in range(19, 37):
                angle = 2 * np.pi * (i - 19) / 18
                centers[i] = [0.6 * np.cos(angle), 0.6 * np.sin(angle)]
            # Fourth ring: 13 circles
            for i in range(37, 50):
                angle = 2 * np.pi * (i - 37) / 13
                centers[i] = [0.8 * np.cos(angle), 0.8 * np.sin(angle)]
        else:
            # Random with varying density
            scale = 0.3 + 0.5 * (restart % 10) / 10.0
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, scale, N))
            centers = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Ensure all inside unit circle
        for i in range(N):
            d = np.linalg.norm(centers[i])
            if d > 0.95:
                centers[i] = centers[i] * 0.95 / d
        
        # Optimize
        centers_opt, score = _optimize(centers, max_iter=3000, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_centers = centers_opt.copy()
    
    # Final refinement with more iterations if time allows
    if best_centers is not None and time.time() < deadline - 15.0:
        best_centers, best_score = _optimize(best_centers, max_iter=5000, deadline=deadline)
    
    if best_centers is None:
        # Fallback: simple hexagonal
        spacing = 0.2
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 0.95:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            best_centers = points[idx]
        else:
            best_centers = np.random.randn(N, 2) * 0.5
    
    return best_centers
# EVOLVE-BLOCK-END
