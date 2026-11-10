import time
import random
import numpy as np
from collections import Counter

_N = 30
_TRIPLE_IDX = None

def _get_triple_indices():
    global _TRIPLE_IDX
    if _TRIPLE_IDX is None:
        triples = []
        for i in range(_N):
            for j in range(i + 1, _N):
                for k in range(j + 1, _N):
                    triples.append((i, j, k))
        _TRIPLE_IDX = np.array(triples, dtype=np.int32)
    return _TRIPLE_IDX

def _score_fast(points):
    P = np.array(points, dtype=np.int64)
    triples = _get_triple_indices()
    
    i_idx = triples[:, 0]
    j_idx = triples[:, 1]
    k_idx = triples[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    dets = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    dets = dets[dets > 0]
    
    if len(dets) == 0:
        return 0
    
    # Use bincount for fast histogram
    max_d = int(np.max(dets))
    if max_d > 100000:
        # Fallback for very large areas
        counts = Counter(dets.tolist())
        return max(counts.values())
        
    hist_size = max_d + 1
    counts = np.bincount(dets, minlength=hist_size)
    
    return int(np.max(counts))

def _delta_score(points, idx, old_pt, new_pt):
    """
    Compute the change in score when moving point idx from old_pt to new_pt.
    This is an approximation: it calculates the new score and compares.
    For true speedup, we would need to track the histogram, but for now
    we just use the fast full score to ensure correctness, relying on
    a better initial state and more efficient search moves.
    """
    points[idx] = new_pt
    new_score = _score_fast(points)
    points[idx] = old_pt
    return new_score

def _generate_hexagonal_grid(n):
    """Generate a hexagonal-like grid of n points."""
    points = []
    # Hex grid layout
    # Row 0: 0
    # Row 1: 1, 2
    # Row 2: 3, 4, 5
    # ...
    # We want 30 points.
    # Let's try a diamond shape or a hex shape.
    # Hex numbers: 1, 7, 19, 37...
    # 30 is close to 37.
    # Let's generate a hex grid of radius 2 (19 points) and add some more.
    # Or just use a rectangular grid with hex offset.
    
    # Alternative: Simple grid 5x6
    # Let's try to construct a set of points that maximizes collinearity or specific area multiplicity.
    # The best known configurations often involve points on a lattice.
    
    # Let's try a 5x6 grid with spacing 1, centered.
    # x in [-2, 3], y in [-3, 2] -> 5*6=30
    pts = []
    for i in range(5):
        for j in range(6):
            pts.append((i - 2, j - 3))
    return pts

def _generate_lattice_points(n, spacing=1):
    """Generate a grid of points."""
    # Find W, H such that W*H >= n
    # Try to make it square-ish
    W = int(np.ceil(np.sqrt(n)))
    H = int(np.ceil(n / W))
    
    pts = []
    for i in range(W):
        for j in range(H):
            if len(pts) < n:
                pts.append((i * spacing, j * spacing))
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_score = 0
    best_points = None
    
    # 1. Try various structured initial configurations
    
    # A. Rectangular Grids
    factors = [(1, 30), (2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2), (30, 1)]
    for W, H in factors:
        for sx in [1, 2, 3]:
            for sy in [1, 2, 3]:
                grid_width = (W - 1) * sx
                grid_height = (H - 1) * sy
                if grid_width <= 60 and grid_height <= 60:
                    points = [(x * sx, y * sy) for x in range(W) for y in range(H)]
                    # Center it
                    cx = grid_width // 2
                    cy = grid_height // 2
                    points = [(x - cx, y - cy) for x, y in points]
                    
                    score = _score_fast(points)
                    if score > best_score:
                        best_score = score
                        best_points = points[:]
    
    # B. Hexagonal-like / Offset Grids
    # Try a few manual configurations
    # 1. All points on a line (score 0, bad)
    # 2. Points on two parallel lines
    # 3. Points on three parallel lines
    
    # Let's try a specific "good" configuration found in literature or by intuition.
    # For 30 points, a 5x6 grid is standard.
    # Let's also try 10x3, 15x2.
    
    if best_points is None:
        best_points = _generate_lattice_points(30)
        best_score = _score_fast(best_points)
    
    # 2. Local Search
    # To make local search effective, we need to be able to evaluate moves quickly.
    # _score_fast is O(N^3) but with small constant due to NumPy.
    # N=30, triples=4060.
    # One score call takes ~1-2 ms.
    # 85s / 2ms = 42500 moves.
    # This is plenty for a good local search if we are smart about it.
    
    current_points = best_points[:]
    current_score = best_score
    
    # Precompute neighbors for moves
    # We will try moving each point to a nearby empty spot.
    # To avoid O(N^2) collision checks per move, we maintain a set.
    
    point_set = set(current_points)
    
    # Define search neighborhood
    # Radius 2 gives 24 neighbors.
    # Radius 3 gives 48 neighbors.
    # Let's use Radius 2 first, then expand if stuck.
    
    def get_neighbors(x, y, radius):
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    neighbors.append((nx, ny))
        return neighbors
    
    # Hill Climbing
    # We'll do multiple passes.
    # In each pass, we try to improve by moving one point.
    # To speed up, we can randomize the order of points.
    
    max_rounds = 50
    for r in range(max_rounds):
        if time.time() > deadline - 5:
            break
            
        improved = False
        # Shuffle point indices to avoid bias
        indices = list(range(_N))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 3:
                break
                
            ox, oy = current_points[idx]
            neighbors = get_neighbors(ox, oy, 2)
            random.shuffle(neighbors)
            
            for nx, ny in neighbors:
                if (nx, ny) in point_set:
                    continue
                
                # Try move
                current_points[idx] = (nx, ny)
                point_set.remove((ox, oy))
                point_set.add((nx, ny))
                
                new_score = _score_fast(current_points)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points[:]
                else:
                    # Revert
                    current_points[idx] = (ox, oy)
                    point_set.remove((nx, ny))
                    point_set.add((ox, oy))
                    
            if not improved:
                # If no improvement in this point, continue to next point
                pass
                
        if not improved:
            # Local optimum reached for radius 2
            # Try radius 3?
            # Or restart with different initial config?
            # Let's try a few random restarts if we have time.
            
            # Check if we should try a different initial config
            # For now, just break and try final polish.
            break
    
    # 3. Final Polish with Radius 1 (exhaustive)
    if time.time() < deadline - 2:
        current_points = best_points[:]
        current_score = best_score
        point_set = set(current_points)
        
        for _ in range(10):
            if time.time() > deadline - 1:
                break
            improved = False
            for idx in range(_N):
                if time.time() > deadline - 1:
                    break
                ox, oy = current_points[idx]
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = ox + dx, oy + dy
                        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                            continue
                        if (nx, ny) in point_set:
                            continue
                        
                        current_points[idx] = (nx, ny)
                        point_set.remove((ox, oy))
                        point_set.add((nx, ny))
                        
                        new_score = _score_fast(current_points)
                        
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = current_points[:]
                        else:
                            current_points[idx] = (ox, oy)
                            point_set.remove((nx, ny))
                            point_set.add((ox, oy))
            if not improved:
                break
    
    return best_points
