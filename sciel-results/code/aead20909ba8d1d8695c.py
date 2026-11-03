"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 200
HALF = 0.5


def _score(centers):
    """Compute the radius for a given set of centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside square
    if np.any(np.abs(centers) > HALF + 1e-12):
        return 0.0
    # Wall distance
    wall_min = np.min(HALF - np.abs(centers))
    # Pairwise distance
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.0)  # only need pairs closer than 2 (all are)
    if len(pairs) == 0:
        pair_min = np.inf
    else:
        d = np.zeros(len(pairs))
        for i, (a, b) in enumerate(pairs):
            d[i] = np.linalg.norm(centers[a] - centers[b])
        pair_min = np.min(d)
    r = min(wall_min, pair_min / 2.0)
    return r


def _hex_grid_init():
    """Initialize with a hexagonal close-packed pattern."""
    # For N=200, a hex grid with appropriate spacing
    # Hex grid: rows offset by half spacing
    # Let's use a grid that approximates hex packing
    m = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    # Actually, let's just use a simple approach:
    # Place points in a hex pattern
    points = []
    # Hex spacing: dx = s, dy = s * sqrt(3)/2
    # For a unit square, we want to fit as many as possible
    # Let's try different densities
    best_points = None
    best_score = 0.0
    
    for s in np.linspace(0.03, 0.15, 50):
        dx = s
        dy = s * np.sqrt(3) / 2
        row = 0
        while len(points) < N:
            offset = 0 if row % 2 == 0 else dx / 2
            x = -HALF + dx / 2 + offset
            while x <= HALF - dx / 2 + 1e-9:
                if len(points) < N:
                    points.append([x, -HALF + dy / 2 + row * dy])
                x += dx
            row += 1
            if row * dy > 1.0:
                break
        if len(points) >= N:
            pts = np.array(points[:N])
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts
        points = []
    
    if best_points is not None:
        return best_points
    
    # Fallback: square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _push_apart(centers, r_target, iters=2000, lr=0.01):
    """Iteratively push apart circles that overlap and keep them inside the square."""
    centers = centers.copy()
    for it in range(iters):
        # Compute all pairwise distances
        tree = cKDTree(centers)
        pairs = tree.query_pairs(r_target * 2 + 0.01)
        moved = False
        for a, b in pairs:
            d = centers[a] - centers[b]
            dist = np.linalg.norm(d)
            if dist < r_target * 2 - 1e-12:
                if dist < 1e-12:
                    d = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(d)
                # Push apart
                overlap = r_target * 2 - dist
                d = d / dist
                centers[a] += lr * overlap * d
                centers[b] -= lr * overlap * d
                moved = True
        
        # Push inside square
        for i in range(N):
            for k in range(2):
                if centers[i, k] > HALF - r_target:
                    centers[i, k] = HALF - r_target
                    moved = True
                elif centers[i, k] < -HALF + r_target:
                    centers[i, k] = -HALF + r_target
                    moved = True
        
        if not moved:
            break
    
    return centers


def _simulated_annealing(centers, time_budget=100.0):
    """Simulated annealing to maximize the minimum distance."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial temperature
    T = 0.01
    T_min = 1e-8
    cooling_rate = 0.9995
    
    np.random.seed(42)
    
    while T > T_min and time.time() < deadline - 5.0:
        # Random perturbation
        delta = np.random.randn(N, 2) * T
        new_centers = centers + delta
        
        # Clamp to square
        new_centers = np.clip(new_centers, -HALF, HALF)
        
        new_score = _score(new_centers)
        
        if new_score >= best_score or np.random.rand() < np.exp((new_score - best_score) / T):
            centers = new_centers
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        
        T *= cooling_rate
    
    return best_centers, best_score


def _coordinate_ascent(centers, iters=50000, time_budget=80.0):
    """Coordinate ascent: move each point to maximize the minimum constraint."""
    centers = centers.copy()
    best_score = _score(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    step = 0.005
    np.random.seed(123)
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # Pick a random point (or cycle through all)
        i = it % N
        
        # Try moving point i in random directions
        best_local = centers[i].copy()
        best_local_score = best_score
        
        for trial in range(20):
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
            for s in [step, step * 0.5, step * 2]:
                new_pos = centers[i] + direction * s
                # Check if inside square
                if np.any(np.abs(new_pos) > HALF):
                    continue
                old_pos = centers[i]
                centers[i] = new_pos
                sc = _score(centers)
                if sc > best_local_score:
                    best_local_score = sc
                    best_local = new_pos.copy()
                centers[i] = old_pos
        
        if np.any(best_local != centers[i]):
            centers[i] = best_local
            if best_local_score > best_score:
                best_score = best_local_score
                best_centers = centers.copy()
        
        # Decay step size
        if it % 1000 == 0:
            step *= 0.95
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    total_budget = 140.0  # Leave 10s margin before 150s kill
    
    # Step 1: Hex grid initialization
    centers = _hex_grid_init()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    # Step 2: Coordinate ascent with targeted bottleneck elimination
    if remaining > 20:
        time_budget_ca = min(remaining - 10, 60)
        best_centers, best_score = _coordinate_ascent(
            best_centers, iters=200000, time_budget=time_budget_ca
        )
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    # Step 3: Simulated annealing for further refinement
    if remaining > 10:
        time_budget_sa = min(remaining - 5, 50)
        best_centers, sa_score = _simulated_annealing(
            best_centers, time_budget=time_budget_sa
        )
        if sa_score > best_score:
            best_score = sa_score
    
    # Final validation
    final_score = _score(best_centers)
    if final_score <= 0:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
