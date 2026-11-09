"""Pack 50 equal circles in a unit circle using binary search + simulated annealing."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 50
KNOWN_BEST = 0.1258254895304


def _compute_score(points):
    """Compute the maximum radius for the given centres."""
    # Boundary constraint
    r_boundary = 1.0 - np.max(np.hypot(points[:, 0], points[:, 1]))
    # Pairwise constraint
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    r_pairs = np.min(d) / 2.0
    return min(r_boundary, r_pairs)


def _hex_lattice_init(n, seed=42):
    """Generate a hexagonal lattice initial configuration."""
    rng = np.random.RandomState(seed)
    # Try different spacings to get at least n points
    for spacing in np.linspace(0.2, 0.05, 50):
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                    rows.append((x, y))
        points = np.array(rows)
        if len(points) >= n:
            # Shuffle and pick n points
            idx = rng.choice(len(points), n, replace=False)
            return points[idx]
    # Fallback: random
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0, 0.8, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _simulated_annealing(points, target_r, max_iter=20000, T_start=0.05, T_end=0.001, seed=0):
    """Simulated annealing to find a valid packing with radius >= target_r."""
    rng = np.random.RandomState(seed)
    points = points.copy()
    n = len(points)
    
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    for it in range(max_iter):
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Pick a random point and perturb it
        idx = rng.randint(n)
        delta = rng.normal(0, T * 0.5, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        # Clamp to unit circle
        dist = np.hypot(new_points[idx, 0], new_points[idx, 1])
        if dist > 1.0 - target_r:
            scale = (1.0 - target_r) / dist
            new_points[idx] *= scale
        
        new_score = _compute_score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            delta_E = current_score - new_score
            if delta_E < 0 or rng.random() < np.exp(-delta_E / (T + 1e-10)):
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
    
    return best_points, best_score


def _polish(points, target_r, max_iter=5000, seed=0):
    """Final polish with smaller steps."""
    rng = np.random.RandomState(seed)
    points = points.copy()
    n = len(points)
    
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    for it in range(max_iter):
        step_size = 0.01 * (1.0 - it / max_iter) + 0.001
        idx = rng.randint(n)
        delta = rng.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        dist = np.hypot(new_points[idx, 0], new_points[idx, 1])
        if dist > 1.0 - target_r:
            scale = (1.0 - target_r) / dist
            new_points[idx] *= scale
        
        new_score = _compute_score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            delta_E = current_score - new_score
            if delta_E < 0 or rng.random() < np.exp(-delta_E / 0.01):
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = _hex_lattice_init(N, seed=42)
    best_score = _compute_score(best_points)
    
    # Binary search on radius
    r_low = 0.1
    r_high = 0.135
    
    # First, find a good upper bound by trying to pack
    for seed in range(20):
        if time.time() > deadline - 30.0:
            break
        points = _hex_lattice_init(N, seed=seed + 100)
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Binary search iterations
    for iteration in range(15):
        if time.time() > deadline - 20.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a valid packing with radius r_mid
        found = False
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
            points = _hex_lattice_init(N, seed=seed * 1000 + iteration * 100)
            polished, score = _simulated_annealing(points, r_mid, max_iter=5000, T_start=0.03, T_end=0.001, seed=seed * 7)
            if score >= r_mid:
                found = True
                if score > best_score:
                    best_score = score
                    best_points = polished.copy()
                break
        
        if found:
            r_low = r_mid
        else:
            r_high = r_mid
    
    # Final polish
    if time.time() < deadline - 5.0:
        polished, score = _polish(best_points, best_score, max_iter=3000, seed=12345)
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    # Verify validity
    d = cdist(best_points, best_points)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    max_radius = np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
    final_score = min(min_pair_dist / 2.0, 1.0 - max_radius)
    
    # Ensure all points are inside the unit circle
    dists = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = dists <= 1.0
    if not np.all(mask):
        # Fallback to a valid configuration
        best_points = _hex_lattice_init(N, seed=42)
    
    return best_points
