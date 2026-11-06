"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import Counter
import itertools

# EVOLVE-BLOCK-START
import numpy as np

def _compute_score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _inbounds(points, limit=30):
    for x, y in points:
        if x < -limit or x > limit or y < -limit or y > limit:
            return False
    return True

def _build_grid(w, h, x0=0, y0=0):
    return [(x0 + x, y0 + y) for x in range(w) for y in range(h)]

def _vectorized_determinants(points):
    """Compute all C(n,3) determinants using NumPy vectorization.
    Returns array of shape (n, n, n) where det[i,j,k] = abs(det) for i<j<k."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    # Create index arrays
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    # Only consider i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    ii = i_idx[mask]
    jj = j_idx[mask]
    kk = k_idx[mask]
    # Compute determinants: |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    x1, y1 = pts[ii, 0], pts[ii, 1]
    x2, y2 = pts[jj, 0], pts[jj, 1]
    x3, y3 = pts[kk, 0], pts[kk, 1]
    dets = np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    return dets, ii, jj, kk

def _find_best_area_and_triples(points):
    """Find the area value with highest multiplicity and return (area, list_of_triples)."""
    dets, ii, jj, kk = _vectorized_determinants(points)
    # Filter out zero-area
    positive_mask = dets > 0
    if not np.any(positive_mask):
        return 0, []
    pos_dets = dets[positive_mask]
    pos_ii = ii[positive_mask]
    pos_jj = jj[positive_mask]
    pos_kk = kk[positive_mask]
    
    # Count frequencies
    unique_areas, counts = np.unique(pos_dets, return_counts=True)
    max_count = np.max(counts)
    best_area = unique_areas[np.argmax(counts)]
    
    # Get all triples with this area
    mask = pos_dets == best_area
    triples = list(zip(pos_ii[mask], pos_jj[mask], pos_kk[mask]))
    return int(best_area), triples

def _greedy_set_cover_for_area(points, target_area, max_iters=5000, timeout=30):
    """Greedy heuristic: iteratively move points to maximize count of triples with target_area.
    Uses the hypergraph model where we want to maximize edges (triples) with area = target_area."""
    start_time = time.time()
    n = len(points)
    best_points = list(points)
    
    # Precompute current score for target_area
    def _count_target_area(pts):
        dets, ii, jj, kk = _vectorized_determinants(pts)
        return np.sum(dets == target_area)
    
    best_count = _count_target_area(best_points)
    no_improve = 0
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_point = best_points[idx]
        
        # Try several candidate positions
        best_new_point = None
        best_new_count = best_count
        
        # Candidate positions: nearby grid points + random
        candidates = set()
        ox, oy = old_point
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = ox + dx, oy + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    if (nx, ny) not in best_points:
                        candidates.add((nx, ny))
        # Add some random candidates
        for _ in range(5):
            rx = random.randint(-30, 30)
            ry = random.randint(-30, 30)
            if (rx, ry) not in best_points:
                candidates.add((rx, ry))
        
        for cand in candidates:
            best_points[idx] = cand
            new_count = _count_target_area(best_points)
            if new_count > best_new_count:
                best_new_count = new_count
                best_new_point = cand
        
        if best_new_point is not None:
            best_points[idx] = best_new_point
            best_count = best_new_count
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 100:
                break
    
    return best_points, best_count

def _local_improve(points, max_iters=2000, timeout=45):
    """Local improvement via point moves, targeting the best area."""
    start_time = time.time()
    best_points = list(points)
    best_score = _compute_score(best_points)
    
    # Find the current best area
    best_area, _ = _find_best_area_and_triples(best_points)
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Try moving a random point to a random location
        idx = random.randint(0, len(best_points) - 1)
        old_point = best_points[idx]
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_point = (new_x, new_y)
        
        # Check if point already exists
        if new_point in best_points:
            continue
        
        best_points[idx] = new_point
        new_score = _compute_score(best_points)
        
        if new_score > best_score:
            best_score = new_score
            # Update target area if score improved
            best_area, _ = _find_best_area_and_triples(best_points)
        else:
            best_points[idx] = old_point
    
    return best_points, best_score

def _generate_candidate(w, h, x0=None, y0=None):
    """Generate a grid candidate."""
    if x0 is None:
        x0 = random.randint(-30, 30 - w + 1)
    if y0 is None:
        y0 = random.randint(-30, 30 - h + 1)
    return _build_grid(w, h, x0, y0)

def _construct_via_area_focus(deadline):
    """Construct points by focusing on maximizing a specific area value.
    Uses vectorized determinant computation and greedy set-cover."""
    start_time = time.time()
    best_points = None
    best_score = 0
    
    # Start with a grid baseline
    grid_configs = [(6, 5), (5, 6), (10, 3), (3, 10)]
    for w, h in grid_configs:
        if time.time() > deadline - 5:
            break
        points = _generate_candidate(w, h)
        if not _inbounds(points):
            continue
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = list(points)
    
    if best_points is None:
        best_points = _build_grid(6, 5)
        best_score = _compute_score(best_points)
    
    # Find the best area for this configuration
    target_area, triples = _find_best_area_and_triples(best_points)
    
    # Apply greedy set-cover to maximize this area
    if time.time() < deadline - 10:
        improved, improved_count = _greedy_set_cover_for_area(
            best_points, target_area, max_iters=3000, timeout=25
        )
        improved_score = _compute_score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved
    
    # Try local improvement
    if time.time() < deadline - 5:
        improved_points, improved_score = _local_improve(
            best_points, max_iters=1000, timeout=15
        )
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s safety margin
    
    # Strategy 1: Area-focused construction with vectorized determinants
    best_points, best_score = _construct_via_area_focus(deadline)
    
    # Strategy 2: Try a few random starts with local improvement if time permits
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            points = []
            used = set()
            while len(points) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in used:
                    points.append((x, y))
                    used.add((x, y))
            
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = list(points)
            
            if time.time() < deadline - 5:
                improved_points, improved_score = _local_improve(
                    points, max_iters=300, timeout=10
                )
                if improved_score > best_score:
                    best_score = improved_score
                    best_points = improved_points
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
