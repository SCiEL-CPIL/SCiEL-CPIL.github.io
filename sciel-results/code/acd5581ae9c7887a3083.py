"""Optimized construction of 16 integer points maximizing exact-4-point lines."""
import time
from itertools import combinations
from fractions import Fraction

def _line_key(p1, p2):
    """Return a canonical key for the line through two points (dx, dy, c)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Normalize direction
    g = abs(__import__('math').gcd(dx, dy))
    dx //= g
    dy //= g
    # Ensure canonical direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Line equation: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_exact4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    # For each line, the number of point-pairs is C(k,2) where k is the number of points on that line
    # We need to recover k from pair count
    exact4 = 0
    for key, pair_count in line_counts.items():
        # Find k such that C(k,2) == pair_count
        # k*(k-1)/2 = pair_count
        k = int((1 + (1 + 8 * pair_count) ** 0.5) / 2)
        if k * (k - 1) // 2 == pair_count and k == 4:
            exact4 += 1
    return exact4

def _has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    if n < 5:
        return False
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    for key, pair_count in line_counts.items():
        k = int((1 + (1 + 8 * pair_count) ** 0.5) / 2)
        if k * (k - 1) // 2 == pair_count and k >= 5:
            return True
    return False

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Start with the 4x4 grid (score 1.0 baseline: 10 exact-4 lines)
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _count_exact4(best_points)
    
    # Local search: try moving each point to a nearby position
    import random
    random.seed(42)
    
    # Candidate positions to try for each point
    def get_candidates(pt, current_points):
        """Generate candidate positions for a point."""
        x, y = pt
        candidates = []
        # Try small perturbations
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    # Check not already occupied
                    if [nx, ny] not in current_points:
                        candidates.append([nx, ny])
        return candidates
    
    # Local search
    current_points = [list(p) for p in best_points]
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        # Shuffle point order for diversity
        order = list(range(16))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            pt = current_points[idx]
            candidates = get_candidates(pt, current_points)
            
            best_cand = None
            best_cand_score = current_score - 1  # Must improve
            
            for cand in candidates:
                # Create new configuration
                new_points = [list(p) for p in current_points]
                new_points[idx] = cand
                
                # Check validity: no duplicates, no five collinear
                if len(set(tuple(p) for p in new_points)) != 16:
                    continue
                if _has_five_collinear(new_points):
                    continue
                
                score = _count_exact4(new_points)
                if score > best_cand_score:
                    best_cand_score = score
                    best_cand = cand
            
            if best_cand is not None:
                current_points[idx] = best_cand
                current_score = best_cand_score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 10  # Penalize full pass without improvement
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_points = [list(p) for p in current_points]
    
    # Final verification
    assert len(set(tuple(p) for p in best_points)) == 16
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in best_points)
    assert not _has_five_collinear(best_points)
    
    return [[int(p[0]), int(p[1])] for p in best_points]
