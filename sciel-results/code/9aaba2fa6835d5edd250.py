"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import Counter

# EVOLVE-BLOCK-START
import numpy as np

def _compute_all_areas(pts):
    """Compute all positive doubled areas for 30 points. Returns list of areas."""
    n = len(pts)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3s = pts[j+1:, 0]
            y3s = pts[j+1:, 1]
            dets = np.abs(x1 * (y2 - y3s) + x2 * (y3s - y1) + x3s * (y1 - y2))
            mask = dets > 0
            if np.any(mask):
                areas.extend(dets[mask].tolist())
    return areas

def _compute_score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    areas = _compute_all_areas(pts)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _compute_score_with_areas(points):
    """Compute score and return (score, Counter of areas)."""
    n = len(points)
    if n < 3:
        return 0, Counter()
    pts = np.array(points, dtype=np.int64)
    areas = _compute_all_areas(pts)
    if not areas:
        return 0, Counter()
    counts = Counter(areas)
    return max(counts.values()), counts

def _incremental_score(points, idx, old_pt, new_pt, pts_arr, area_counts):
    """
    Compute new score after moving point idx from old_pt to new_pt.
    Uses incremental update: remove areas involving old_pt, add areas involving new_pt.
    Returns (new_score, updated_counts).
    """
    n = len(points)
    counts = dict(area_counts)
    
    # Remove all areas that involve the old point at index idx
    # Areas involving point idx: (idx, j, k) for all j<k, j,k != idx
    # We need to recompute which areas change
    
    # Actually, let's just recompute - it's simpler and still fast enough
    # with numpy vectorization for 30 points
    pts_new = pts_arr.copy()
    pts_new[idx] = new_pt
    areas = _compute_all_areas(pts_new)
    if not areas:
        return 0, Counter()
    new_counts = Counter(areas)
    return max(new_counts.values()), new_counts

def _inbounds(points, limit=30):
    for x, y in points:
        if x < -limit or x > limit or y < -limit or y > limit:
            return False
    return True

def _build_grid(w, h, x0=0, y0=0):
    return [(x0 + x, y0 + y) for x in range(w) for y in range(h)]

def _build_lattice(w, h, x0=0, y0=0, dx=1, dy=1, skew=0):
    """Build a lattice/parallelogram pattern."""
    return [(x0 + x * dx + y * skew, y0 + y * dy) for x in range(w) for y in range(h)]

def _sa_search(points, max_iters=3000, timeout=40, start_temp=10.0, end_temp=0.1):
    """
    Simulated annealing search with vectorized score computation.
    Uses acceptance criterion to escape local optima.
    """
    start_time = time.time()
    best_points = list(points)
    best_score = _compute_score(best_points)
    current_score = best_score
    n = len(best_points)
    
    current_points = list(best_points)
    current_pts = np.array(current_points, dtype=np.int64)
    
    total_iters = max_iters
    for it in range(total_iters):
        if time.time() - start_time > timeout:
            break
        
        # Temperature schedule
        progress = it / max(1, total_iters - 1)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Choose move type
        if random.random() < 0.8:
            # Local move: small perturbation
            idx = random.randint(0, n - 1)
            ox, oy = current_points[idx]
            new_x = ox + random.randint(-4, 4)
            new_y = oy + random.randint(-4, 4)
        else:
            # Larger jump
            idx = random.randint(0, n - 1)
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        
        new_point = (new_x, new_y)
        
        # Check bounds and uniqueness
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
        if new_point in current_points:
            continue
        
        # Compute new score
        new_pts = current_pts.copy()
        new_pts[idx] = new_point
        areas = _compute_all_areas(new_pts)
        if not areas:
            continue
        new_counts = Counter(areas)
        new_score = max(new_counts.values())
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or (temp > 0.01 and random.random() < np.exp(delta / temp)):
            current_points[idx] = new_point
            current_pts = new_pts
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
        else:
            # Reject
            pass
    
    return best_points, best_score

def _targeted_polish(points, timeout=15):
    """
    Targeted polish: for each point, try all nearby positions and keep the best.
    Multiple passes for convergence.
    """
    start_time = time.time()
    best_points = list(points)
    best_score = _compute_score(best_points)
    n = len(best_points)
    
    for _pass in range(5):
        if time.time() - start_time > timeout:
            break
        improved = False
        for idx in range(n):
            if time.time() - start_time > timeout:
                break
            ox, oy = best_points[idx]
            best_local_score = best_score
            best_local_pt = (ox, oy)
            
            # Try all positions in a 7x7 neighborhood
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    new_pt = (nx, ny)
                    if new_pt in best_points:
                        continue
                    
                    new_pts = np.array(best_points, dtype=np.int64)
                    new_pts[idx] = new_pt
                    areas = _compute_all_areas(new_pts)
                    if not areas:
                        continue
                    new_counts = Counter(areas)
                    new_score = max(new_counts.values())
                    
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_local_pt = new_pt
            
            if best_local_pt != (ox, oy):
                best_points[idx] = best_local_pt
                best_score = best_local_score
                improved = True
        
        if not improved:
            break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (evaluator kills at 90s)
    
    best_points = None
    best_score = 0
    
    def try_config(points):
        nonlocal best_points, best_score
        if not _inbounds(points):
            return
        if len(set(points)) != 30:
            return
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = list(points)
    
    # Phase 1: Try structured configurations
    # Grids
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30), (5, 5), (4, 8), (8, 4), (3, 10), (10, 3)]:
        if time.time() > deadline - 20:
            break
        for x0 in range(-2, 3):
            for y0 in range(-2, 3):
                points = _build_grid(w, h, x0, y0)
                try_config(points)
    
    # Lattice patterns with various parameters
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (4, 8), (8, 4)]:
        if time.time() > deadline - 20:
            break
        for dx in [1, 2, 3]:
            for dy in [1, 2, 3]:
                for skew in [-2, -1, 0, 1, 2]:
                    if time.time() > deadline - 20:
                        break
                    x0 = random.randint(-20, 20)
                    y0 = random.randint(-20, 20)
                    points = _build_lattice(w, h, x0, y0, dx, dy, skew)
                    try_config(points)
    
    # Phase 2: SA search from best structured config
    if best_points is not None and time.time() < deadline - 30:
        improved_points, improved_score = _sa_search(
            best_points, max_iters=2000, timeout=25, start_temp=5.0, end_temp=0.05
        )
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    # Phase 3: Targeted polish
    if best_points is not None and time.time() < deadline - 10:
        improved_points, improved_score = _targeted_polish(
            best_points, timeout=min(15, deadline - time.time() - 5)
        )
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    # Phase 4: Additional SA restarts if time permits
    if time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            # Create a perturbed version of best
            perturbed = list(best_points)
            for _ in range(5):
                idx = random.randint(0, 29)
                ox, oy = perturbed[idx]
                perturbed[idx] = (ox + random.randint(-5, 5), oy + random.randint(-5, 5))
            # Clean up: ensure in bounds and unique
            clean = []
            used = set()
            for p in perturbed:
                x, y = p
                if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in used:
                    clean.append((x, y))
                    used.add((x, y))
            while len(clean) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in used:
                    clean.append((x, y))
                    used.add((x, y))
            
            improved_points, improved_score = _sa_search(
                clean, max_iters=500, timeout=min(8, deadline - time.time() - 5),
                start_temp=3.0, end_temp=0.05
            )
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points

def _generate_candidate(w, h, x0=None, y0=None):
    """Generate a grid candidate."""
    if x0 is None:
        x0 = random.randint(-30, 30 - w + 1)
    if y0 is None:
        y0 = random.randint(-30, 30 - h + 1)
    return _build_grid(w, h, x0, y0)
# EVOLVE-BLOCK-END
