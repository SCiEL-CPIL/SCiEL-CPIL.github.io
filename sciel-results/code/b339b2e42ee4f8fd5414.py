"""CSP-inspired approach: target specific doubled areas, use lattice-based constructions, and incremental local search."""
import random
import time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples."""
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    if n < 3:
        return 0
    
    # Precompute all triples
    triples = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    if not triples:
        return 0
    
    triples = np.array(triples)
    Pi = pts[triples[:, 0]]
    Pj = pts[triples[:, 1]]
    Pk = pts[triples[:, 2]]
    
    dx_ij = Pj[:, 0] - Pi[:, 0]
    dy_ij = Pj[:, 1] - Pi[:, 1]
    dx_ik = Pk[:, 0] - Pi[:, 0]
    dy_ik = Pk[:, 1] - Pi[:, 1]
    
    dets = dx_ij * dy_ik - dy_ij * dx_ik
    non_zero = dets[dets != 0]
    
    if len(non_zero) == 0:
        return 0
    
    counts = Counter(non_zero.tolist())
    return max(counts.values()) if counts else 0


def _compute_all_areas(points):
    """Return dict: area -> count of triples with that positive doubled area."""
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    triples = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    if not triples:
        return Counter()
    
    triples = np.array(triples)
    Pi = pts[triples[:, 0]]
    Pj = pts[triples[:, 1]]
    Pk = pts[triples[:, 2]]
    
    dx_ij = Pj[:, 0] - Pi[:, 0]
    dy_ij = Pj[:, 1] - Pi[:, 1]
    dx_ik = Pk[:, 0] - Pi[:, 0]
    dy_ik = Pk[:, 1] - Pi[:, 1]
    
    dets = dx_ij * dy_ik - dy_ij * dx_ik
    non_zero = dets[dets != 0]
    
    if len(non_zero) == 0:
        return Counter()
    
    return Counter(non_zero.tolist())


def _lattice_construction(origin, size_x, size_y, shift=None):
    """Create a grid of points. Returns list of (x,y) tuples."""
    pts = []
    for dx in range(size_x):
        for dy in range(size_y):
            x = origin[0] + dx
            y = origin[1] + dy
            if shift is not None:
                x += shift[0]
                y += shift[1]
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts


def _build_initial_candidates(rng):
    """Generate diverse initial configurations."""
    candidates = []
    
    # Candidate 1: Dense 6x5 grid at origin
    pts = _lattice_construction((0, 0), 6, 5)
    if len(pts) == 30:
        candidates.append(pts[:])
    
    # Candidate 2: Two 3x5 grids offset
    lat1 = _lattice_construction((0, 0), 3, 5)
    lat2 = _lattice_construction((8, 0), 3, 5)
    pts = lat1 + lat2
    if len(pts) == 30:
        seen = set()
        valid = []
        for p in pts:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                valid.append(p)
        if len(valid) == 30:
            candidates.append(valid)
    
    # Candidate 3: Triangular arrangement
    pts = []
    row = 0
    while len(pts) < 30:
        for col in range(row + 1):
            x = col
            y = row
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
            if len(pts) >= 30:
                break
        row += 1
    if len(pts) == 30:
        candidates.append(pts[:30])
    
    # Candidate 4: Random points
    seen = set()
    pts = []
    while len(pts) < 30:
        p = (rng.randint(-15, 15), rng.randint(-15, 15))
        if p not in seen:
            seen.add(p)
            pts.append(p)
    candidates.append(pts)
    
    # Candidate 5: Concentric rings
    pts = []
    for radius in range(0, 6):
        if radius == 0:
            pts.append((0, 0))
        else:
            for angle in np.linspace(0, 2 * np.pi, max(1, int(2 * np.pi * radius // 1.5)), endpoint=False):
                x = round(radius * np.cos(angle))
                y = round(radius * np.sin(angle))
                p = (x, y)
                if p not in pts and -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append(p)
    # Pad if needed
    if len(pts) < 30:
        seen = set(pts)
        while len(pts) < 30:
            p = (rng.randint(-10, 10), rng.randint(-10, 10))
            if p not in seen:
                seen.add(p)
                pts.append(p)
    candidates.append(pts[:30])
    
    # Candidate 6: Arithmetic progression on a line + perturbations
    pts = []
    for i in range(15):
        pts.append((i * 2, 0))
    for i in range(15):
        pts.append((i * 2, 1))
    # Check validity
    seen = set()
    valid = []
    for p in pts:
        if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
            seen.add(p)
            valid.append(p)
    if len(valid) == 30:
        candidates.append(valid)
    
    return candidates


def _local_search(points, rng, deadline, target_area=None):
    """Local search with incremental area tracking."""
    pts = points[:]
    n = len(pts)
    
    # Precompute triples
    triples = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    triples = np.array(triples)
    
    # Build triple index: for each point, which triples it appears in
    point_to_triples = defaultdict(list)
    for idx, (i, j, k) in enumerate(triples):
        point_to_triples[i].append(idx)
        point_to_triples[j].append(idx)
        point_to_triples[k].append(idx)
    
    # Compute initial areas
    Pi = pts[triples[:, 0]]
    Pj = pts[triples[:, 1]]
    Pk = pts[triples[:, 2]]
    
    dx_ij = Pj[:, 0] - Pi[:, 0]
    dy_ij = Pj[:, 1] - Pi[:, 1]
    dx_ik = Pk[:, 0] - Pi[:, 0]
    dy_ik = Pk[:, 1] - Pi[:, 1]
    
    dets = dx_ij * dy_ik - dy_ij * dx_ik
    non_zero_mask = dets != 0
    areas = np.abs(dets[non_zero_mask])
    
    area_counts = Counter(areas.tolist())
    best_score = max(area_counts.values()) if area_counts else 0
    best_points = pts[:]
    
    # Track which triples are non-zero and their areas
    triple_areas = {}  # idx -> area (0 if degenerate)
    for idx in range(len(triples)):
        triple_areas[idx] = abs(int(dets[idx])) if dets[idx] != 0 else 0
    
    # Area count tracking
    current_area_counts = Counter()
    for idx, area in triple_areas.items():
        if area > 0:
            current_area_counts[area] += 1
    
    temp = 20.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick random point to move
        idx = rng.randint(0, n - 1)
        old_pt = pts[idx]
        
        # Generate new position
        if rng.random() < 0.6:
            # Small perturbation
            dx = rng.randint(-2, 2)
            dy = rng.randint(-2, 2)
            new_pt = (old_pt[0] + dx, old_pt[1] + dy)
        else:
            # Larger move
            new_pt = (rng.randint(-30, 30), rng.randint(-30, 30))
        
        # Check validity
        if new_pt[0] < -30 or new_pt[0] > 30 or new_pt[1] < -30 or new_pt[1] > 30:
            continue
        if new_pt in pts:
            continue
        
        # Try the move
        pts[idx] = new_pt
        
        # Update affected triples
        affected = point_to_triples[idx]
        old_areas = {}
        new_areas = {}
        
        for tidx in affected:
            old_areas[tidx] = triple_areas[tidx]
            i, j, k = triples[tidx]
            Pi_pt = pts[i]
            Pj_pt = pts[j]
            Pk_pt = pts[k]
            det = (Pj_pt[0] - Pi_pt[0]) * (Pk_pt[1] - Pi_pt[1]) - (Pj_pt[1] - Pi_pt[1]) * (Pk_pt[0] - Pi_pt[0])
            new_area = abs(det) if det != 0 else 0
            new_areas[tidx] = new_area
        
        # Check if this move is beneficial
        # For simplicity, just check if we can improve
        if target_area is not None:
            # Count how many affected triples now have target area
            gain = sum(1 for tidx in affected if new_areas[tidx] == target_area and old_areas[tidx] != target_area)
            loss = sum(1 for tidx in affected if old_areas[tidx] == target_area and new_areas[tidx] != target_area)
            if gain > loss:
                # Accept
                for tidx in affected:
                    old_a = old_areas[tidx]
                    new_a = new_areas[tidx]
                    if old_a > 0:
                        current_area_counts[old_a] -= 1
                        if current_area_counts[old_a] == 0:
                            del current_area_counts[old_a]
                    if new_a > 0:
                        current_area_counts[new_a] += 1
                    triple_areas[tidx] = new_a
            else:
                pts[idx] = old_pt
                continue
        else:
            # General case: compute new max
            for tidx in affected:
                old_a = old_areas[tidx]
                new_a = new_areas[tidx]
                if old_a > 0:
                    current_area_counts[old_a] -= 1
                    if current_area_counts[old_a] == 0:
                        del current_area_counts[old_a]
                if new_a > 0:
                    current_area_counts[new_a] += 1
                triple_areas[tidx] = new_a
            
            new_score = max(current_area_counts.values()) if current_area_counts else 0
            if new_score >= best_score or (new_score >= best_score - 1 and rng.random() < temp):
                if new_score > best_score:
                    best_score = new_score
                    best_points = pts[:]
            else:
                # Revert
                pts[idx] = old_pt
                for tidx in affected:
                    old_a = old_areas[tidx]
                    new_a = new_areas[tidx]
                    if new_a > 0:
                        current_area_counts[new_a] -= 1
                        if current_area_counts[new_a] == 0:
                            del current_area_counts[new_a]
                    if old_a > 0:
                        current_area_counts[old_a] += 1
                    triple_areas[tidx] = old_a
        
        temp *= cooling
        if temp < 0.5:
            temp = 15.0
    
    return best_points, best_score


def construct():
    deadline = time.time() + 85  # Safety margin
    
    rng = random.Random(42)
    
    # Generate initial candidates
    candidates = _build_initial_candidates(rng)
    
    # Score all candidates
    best_score = 0
    best_points = None
    for cand in candidates:
        if len(cand) != 30:
            continue
        seen = set()
        valid = []
        for p in cand:
            if p not in seen:
                seen.add(p)
                valid.append(p)
        if len(valid) == 30:
            s = _score(valid)
            if s > best_score:
                best_score = s
                best_points = valid[:]
    
    if best_points is None:
        # Fallback
        best_points = [(i % 6, i // 6) for i in range(30)]
        best_score = _score(best_points)
    
    # Try local search from best
    if time.time() < deadline - 10:
        improved_pts, improved_score = _local_search(best_points, rng, deadline - 5)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts
    
    # Try a few more random restarts if time permits
    attempts = 0
    while time.time() < deadline - 15 and attempts < 5:
        # Generate new random candidate
        seen = set()
        pts = []
        while len(pts) < 30:
            p = (rng.randint(-20, 20), rng.randint(-20, 20))
            if p not in seen:
                seen.add(p)
                pts.append(p)
        
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]
        
        # Local search from this candidate
        if time.time() < deadline - 15:
            improved_pts, improved_score = _local_search(pts, rng, deadline - 10)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_pts
        
        attempts += 1
    
    return best_points


if __name__ == "__main__":
    result = construct()
    print(f"Score: {_score(result)}")
    print(f"Points: {result}")
