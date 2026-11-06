"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points. O(n^2)."""
    n = len(points)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            if A == 0 and B == 0:
                continue
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) if C != 0 else math.gcd(abs(A), abs(B))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key in line_map:
                line_map[key].add(j)
            else:
                line_map[key] = {i, j}
    ordinary = sum(1 for pts in line_map.values() if len(pts) == 2)
    return ordinary

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, len(points)):
        x3, y3 = points[i]
        if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
            return True
    return False

def _try_configuration(pts):
    """Evaluate a configuration and return (points, score) if valid."""
    if len(pts) != 12:
        return None
    if len(set(tuple(p) for p in pts)) != 12:
        return None
    if not _not_all_collinear(pts):
        return None
    for p in pts:
        if not (-10**9 <= p[0] <= 10**9 and -10**9 <= p[1] <= 10**9):
            return None
    score = _count_ordinary_lines(pts)
    return pts, score

def _local_search(points, max_iter=30000, timeout=30):
    """Local search: move one point at a time."""
    start_time = time.time()
    best = [p[:] for p in points]
    best_score = _count_ordinary_lines(best)
    current = [p[:] for p in points]
    current_score = best_score
    n = len(current)
    
    for it in range(max_iter):
        if time.time() - start_time > timeout:
            break
        i = random.randint(0, n - 1)
        old_x, old_y = current[i]
        
        # Generate candidate positions
        candidates = set()
        # Nearby positions
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                nx, ny = old_x + dx, old_y + dy
                if -10**9 <= nx <= 10**9 and -10**9 <= ny <= 10**9:
                    candidates.add((nx, ny))
        # Positions sharing x or y with other points
        for j in range(n):
            if j != i:
                candidates.add((current[j][0], old_y))
                candidates.add((old_x, current[j][1]))
        
        best_candidate = None
        best_candidate_score = current_score
        
        for (nx, ny) in candidates:
            if (nx, ny) == (old_x, old_y):
                continue
            if any(p[0] == nx and p[1] == ny for p in current if p is not current[i]):
                continue
            new_points = [p[:] for p in current]
            new_points[i] = [nx, ny]
            if not _not_all_collinear(new_points):
                continue
            score = _count_ordinary_lines(new_points)
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate = (nx, ny)
        
        if best_candidate is not None:
            current[i] = list(best_candidate)
            current_score = best_candidate_score
            if current_score < best_score:
                best_score = current_score
                best = [p[:] for p in current]
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 140
    
    random.seed(12345)
    
    best_points = None
    best_score = float('inf')
    
    # Strategy 1: 3x4 grid baseline
    base1 = [[x, y] for x in range(3) for y in range(4)]
    pts, score = _try_configuration(base1)
    if pts:
        best_points, best_score = pts, score
        pts, score = _local_search(base1, max_iter=10000, timeout=20)
        if score < best_score:
            best_score = score
            best_points = pts
    
    # Strategy 2: Try structured configurations
    # 4 points on 3 lines (12 points total, 4 per line)
    # Lines: y=0, y=1, y=2 with 4 points each
    # But need to avoid all collinear and try to maximize collinear triples
    
    # Try: 4 points on x-axis at x=0,1,2,3; 4 on y=1 at x=0,1,2,3; 4 on y=2 at x=0,1,2,3
    # This is 4x3 grid - same as 3x4 essentially
    
    # Try asymmetric: shift the rows
    configs = [
        # 4x3 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 3x4 grid  
        [[x, y] for x in range(3) for y in range(4)],
        # Shifted rows
        [[x, 0] for x in range(4)] + [[x+1, 1] for x in range(4)] + [[x+2, 2] for x in range(4)],
        # Diagonal arrangement
        [[i, i] for i in range(4)] + [[i+4, i] for i in range(4)] + [[i, i+4] for i in range(4)],
        # 2x6 grid
        [[x, y] for x in range(2) for y in range(6)],
        # 6x2 grid
        [[x, y] for x in range(6) for y in range(2)],
    ]
    
    for cfg in configs:
        if time.time() - start_time > deadline - 30:
            break
        result = _try_configuration(cfg)
        if result:
            pts, score = result
            if score < best_score:
                best_score = score
                best_points = pts
            # Local search from this config
            pts2, score2 = _local_search(cfg, max_iter=8000, timeout=10)
            if score2 < best_score:
                best_score = score2
                best_points = pts2
    
    # Strategy 3: Random restarts
    for restart in range(20):
        if time.time() - start_time > deadline - 15:
            break
        pts = []
        used = set()
        for _ in range(12):
            for _attempt in range(50):
                x = random.randint(-6, 6)
                y = random.randint(-6, 6)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append([x, y])
                    break
            else:
                pts.append([random.randint(-10, 10), random.randint(-10, 10)])
        
        if not _not_all_collinear(pts):
            continue
        
        pts2, score2 = _local_search(pts, max_iter=5000, timeout=5)
        if score2 < best_score:
            best_score = score2
            best_points = pts2
    
    # Final validation
    if best_points is None:
        best_points = [[x, y] for x in range(3) for y in range(4)]
    
    assert len(best_points) == 12
    assert len(set(tuple(p) for p in best_points)) == 12
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
