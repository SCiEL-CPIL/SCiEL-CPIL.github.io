"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START

def _compute_score_fast(points):
    """Compute packing score using vectorized operations."""
    n = len(points)
    if n == 0:
        return 0.0
    
    # Min distance from any center to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    if min_boundary <= 0:
        return 0.0
    
    # Min pairwise distance using cdist (much faster than nested loops)
    dist_matrix = cdist(points, points)
    # Set diagonal to inf to exclude self-distances
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix)
    
    return min(min_pair / 2.0, min_boundary)


def _ring_initial(n=50):
    """Create initial ring-based structure with optimized parameters."""
    # Try multiple ring configurations
    best_points = None
    best_score = -1
    
    # Ring sizes that sum to 50: 1 + 6 + 12 + 12 + 19 = 50
    ring_configs = [
        [1, 6, 12, 12, 19],
        [1, 6, 10, 14, 19],
        [1, 8, 12, 14, 15],
        [1, 6, 12, 15, 16],
    ]
    
    for ring_sizes in ring_configs:
        if sum(ring_sizes) != n:
            continue
        
        # Optimize ring radii and angular offsets
        # Initial guess: evenly spaced rings
        n_rings = len(ring_sizes) - 1  # exclude center
        base_radii = np.linspace(0.2, 0.85, n_rings)
        
        for trial in range(5):
            # Perturb radii
            rng = np.random.default_rng(trial * 100 + 42)
            radii = base_radii + (rng.random(n_rings) - 0.5) * 0.1
            radii = np.clip(radii, 0.1, 0.9)
            
            # Angular offsets
            offsets = rng.random(n_rings) * 2 * np.pi
            
            # Build points
            points = np.zeros((n, 2))
            idx = 0
            # Center
            if ring_sizes[0] == 1:
                points[0] = [0.0, 0.0]
                idx = 1
            
            for ring_i, count in enumerate(ring_sizes[1:], 1):
                r = radii[ring_i - 1]
                theta0 = offsets[ring_i - 1]
                for k in range(count):
                    theta = theta0 + 2 * np.pi * k / count
                    points[idx] = [r * np.cos(theta), r * np.sin(theta)]
                    idx += 1
            
            # Check validity
            if np.any(np.linalg.norm(points, axis=1) > 1.0):
                continue
            
            score = _compute_score_fast(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    if best_points is None:
        # Fallback: hexagonal-like
        best_points = np.zeros((n, 2))
        rng = np.random.default_rng(42)
        for i in range(n):
            for attempt in range(1000):
                angle = rng.uniform(0, 2 * np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                pt = np.array([r * np.cos(angle), r * np.sin(angle)])
                if i == 0 or np.all(np.linalg.norm(pt - best_points[:i], axis=1) > 0.12):
                    best_points[i] = pt
                    break
    
    return best_points


def _local_polish(points, time_budget=5.0):
    """Final high-precision polish using coordinate descent with small steps."""
    n = len(points)
    points = points.copy()
    current_score = _compute_score_fast(points)
    best_points = points.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    rng = np.random.default_rng(999)
    
    step_size = 0.001
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            x, y = points[i]
            
            # Try small moves in each direction
            for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                           (step_size * 0.707, step_size * 0.707), (-step_size * 0.707, step_size * 0.707),
                           (step_size * 0.707, -step_size * 0.707), (-step_size * 0.707, -step_size * 0.707)]:
                new_x, new_y = x + dx, y + dy
                if new_x**2 + new_y**2 > 1.0:
                    continue
                
                new_points = points.copy()
                new_points[i] = [new_x, new_y]
                new_score = _compute_score_fast(new_points)
                
                if new_score > current_score:
                    points = new_points
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Gradually reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    return best_points, best_score


def _simulated_annealing(points, time_budget=80.0):
    """Run simulated annealing to improve the packing."""
    n = len(points)
    points = points.copy()
    best_points = points.copy()
    best_score = _compute_score_fast(points)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Temperature schedule
    T = 0.005
    T_min = 1e-8
    cooling_rate = 0.9998
    
    rng = np.random.default_rng(2024)
    
    while time.time() < deadline and T > T_min:
        # Propose a random move
        idx = rng.integers(0, n)
        move_type = rng.random()
        
        if move_type < 0.6:
            # Random displacement with adaptive scale
            scale = 0.02 * (1.0 + T * 10)
            delta = (rng.random(2) - 0.5) * 2 * scale
            new_points = points.copy()
            new_points[idx] = points[idx] + delta
        elif move_type < 0.8:
            # Nudge: move toward/away from center
            r = np.linalg.norm(points[idx])
            if r < 1e-6:
                continue
            direction = points[idx] / r
            delta = direction * (rng.random() - 0.5) * 0.03
            new_points = points.copy()
            new_points[idx] = points[idx] + delta
        elif move_type < 0.9:
            # Swap two points
            jdx = rng.integers(0, n)
            if jdx == idx:
                continue
            new_points = points.copy()
            new_points[idx], new_points[jdx] = points[jdx].copy(), points[idx].copy()
        else:
            # Rotate a point around origin
            angle = (rng.random() - 0.5) * 0.2
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            new_points = points.copy()
            new_points[idx] = rot @ points[idx]
        
        # Check validity
        if np.any(np.linalg.norm(new_points, axis=1) > 1.0):
            continue
        
        new_score = _compute_score_fast(new_points)
        
        # Accept or reject
        if new_score >= best_score or rng.random() < np.exp((new_score - best_score) / T):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        T *= cooling_rate
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate initial guess using ring structure
    points = _ring_initial(N)
    current_score = _compute_score_fast(points)
    
    # Run simulated annealing with most of the time budget
    remaining = deadline - (time.time() - start_time) - 10.0
    if remaining > 10:
        points, current_score = _simulated_annealing(points, time_budget=remaining)
    
    # Final polish pass
    remaining = deadline - (time.time() - start_time) - 2.0
    if remaining > 1:
        points, current_score = _local_polish(points, time_budget=min(5.0, remaining))
    
    # Verify final result is valid
    final_score = _compute_score_fast(points)
    if final_score <= 0:
        # Fallback to a simple valid configuration
        points = _ring_initial(N)
    
    return points

# EVOLVE-BLOCK-END
