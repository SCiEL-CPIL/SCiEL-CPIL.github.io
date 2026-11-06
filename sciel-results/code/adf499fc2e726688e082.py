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
    """Compute all C(n,3) determinants using NumPy vectorization."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    ii = i_idx[mask]
    jj = j_idx[mask]
    kk = k_idx[mask]
    x1, y1 = pts[ii, 0], pts[ii, 1]
    x2, y2 = pts[jj, 0], pts[jj, 1]
    x3, y3 = pts[kk, 0], pts[kk, 1]
    dets = np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    return dets, ii, jj, kk

def _find_best_area_and_triples(points):
    """Find the area value with highest multiplicity."""
    dets, ii, jj, kk = _vectorized_determinants(points)
    positive_mask = dets > 0
    if not np.any(positive_mask):
        return 0, []
    pos_dets = dets[positive_mask]
    pos_ii = ii[positive_mask]
    pos_jj = jj[positive_mask]
    pos_kk = kk[positive_mask]
    
    unique_areas, counts = np.unique(pos_dets, return_counts=True)
    best_idx = np.argmax(counts)
    best_area = unique_areas[best_idx]
    
    mask = pos_dets == best_area
    triples = list(zip(pos_ii[mask], pos_jj[mask], pos_kk[mask]))
    return int(best_area), triples

def _count_target_area(points, target_area):
    """Count triples with exactly target_area using vectorization."""
    dets, _, _, _ = _vectorized_determinants(points)
    return int(np.sum(dets == target_area))

def _greedy_hypergraph_optimize(points, target_area, max_iters=3000, timeout=30):
    """Optimize points to maximize count of triples with target_area using greedy moves."""
    start_time = time.time()
    n = len(points)
    best_points = list(points)
    best_count = _count_target_area(best_points, target_area)
    no_improve = 0
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        idx = random.randint(0, n - 1)
        old_point = best_points[idx]
        ox, oy = old_point
        
        best_new_point = None
        best_new_count = best_count
        
        # Generate diverse candidates
        candidates = set()
        # Nearby positions
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                nx, ny = ox + dx, oy + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    if (nx, ny) not in best_points:
                        candidates.add((nx, ny))
        # Random positions
        for _ in range(8):
            rx = random.randint(-30, 30)
            ry = random.randint(-30, 30)
            if (rx, ry) not in best_points:
                candidates.add((rx, ry))
        
        for cand in candidates:
            best_points[idx] = cand
            new_count = _count_target_area(best_points, target_area)
            if new_count > best_new_count:
                best_new_count = new_count
                best_new_point = cand
        
        if best_new_point is not None:
            best_points[idx] = best_new_point
            best_count = best_new_count
            no_improve = 0
        else:
            best_points[idx] = old_point
            no_improve += 1
            if no_improve > 150:
                break
    
    return best_points, best_count

def _local_improve(points, max_iters=2000, timeout=45):
    """Local improvement targeting the best area."""
    start_time = time.time()
    best_points = list(points)
    best_score = _compute_score(best_points)
    best_area, _ = _find_best_area_and_triples(best_points)
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        idx = random.randint(0, len(best_points) - 1)
        old_point = best_points[idx]
        new_point = (random.randint(-30, 30), random.randint(-30, 30))
        
        if new_point in best_points:
            continue
        
        best_points[idx] = new_point
        new_score = _compute_score(best_points)
        
        if new_score > best_score:
            best_score = new_score
            best_area, _ = _find_best_area_and_triples(best_points)
        else:
            best_points[idx] = old_point
    
    return best_points, best_score

def _generate_grid_candidate(w, h):
    """Generate a grid candidate at random position."""
    x0 = random.randint(-30, 30 - w + 1)
    y0 = random.randint(-30, 30 - h + 1)
    return _build_grid(w, h, x0, y0)

def _generate_lattice_candidate(deadline):
    """Generate lattice-based candidates with various structures."""
    candidates = []
    
    # Standard grids
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]:
        if time.time() > deadline - 3:
            break
        pts = _generate_grid_candidate(w, h)
        if _inbounds(pts):
            candidates.append(pts)
    
    # Parallelogram lattices
    for _ in range(3):
        if time.time() > deadline - 3:
            break
        # Pick two basis vectors
        ax = random.randint(-10, 10)
        ay = random.randint(-10, 10)
        bx = random.randint(-10, 10)
        by = random.randint(-10, 10)
        if ax * by - ay * bx == 0:
            continue
        x0 = random.randint(-20, 20)
        y0 = random.randint(-20, 20)
        pts = []
        used = set()
        for i in range(6):
            for j in range(5):
                px = x0 + i * ax + j * bx
                py = y0 + i * ay + j * by
                if -30 <= px <= 30 and -30 <= py <= 30 and (px, py) not in used:
                    pts.append((px, py))
                    used.add((px, py))
        if len(pts) == 30:
            candidates.append(pts)
    
    return candidates

def _construct_via_area_focus(deadline):
    """Construct points focusing on maximizing a specific area value."""
    start_time = time.time()
    best_points = None
    best_score = 0
    
    # Generate diverse initial candidates
    all_candidates = _generate_lattice_candidate(deadline)
    
    # Also add some random configurations
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        pts = []
        used = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        all_candidates.append(pts)
    
    # Evaluate all candidates
    for pts in all_candidates:
        if time.time() > deadline - 10:
            break
        if not _inbounds(pts):
            continue
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
    
    if best_points is None:
        best_points = _build_grid(6, 5)
        best_score = _compute_score(best_points)
    
    # Find the best area and optimize for it
    target_area, _ = _find_best_area_and_triples(best_points)
    
    # Apply greedy hypergraph optimization
    if time.time() < deadline - 15:
        improved, _ = _greedy_hypergraph_optimize(
            best_points, target_area, max_iters=2000, timeout=20
        )
        improved_score = _compute_score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved
    
    # Final local improvement
    if time.time() < deadline - 5:
        improved_points, improved_score = _local_improve(
            best_points, max_iters=800, timeout=12
        )
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Primary strategy: area-focused construction
    best_points, best_score = _construct_via_area_focus(deadline)
    
    # Secondary: additional random restarts if time permits
    if time.time() < deadline - 15:
        for _ in range(2):
            if time.time() > deadline - 8:
                break
            pts = []
            used = set()
            while len(pts) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in used:
                    pts.append((x, y))
                    used.add((x, y))
            
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = list(pts)
            
            if time.time() < deadline - 5:
                improved_points, improved_score = _local_improve(
                    pts, max_iters=200, timeout=8
                )
                if improved_score > best_score:
                    best_score = improved_score
                    best_points = improved_points
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
