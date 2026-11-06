"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _line_key(i, pts):
    """Compute line keys for point i against all other points."""
    x1, y1 = pts[i]
    keys = []
    for j in range(len(pts)):
        if j == i:
            continue
        x2, y2 = pts[j]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g > 0:
            a //= g
            b //= g
            c //= g
        if a < 0 or (a == 0 and b < 0):
            a, b, c = -a, -b, -c
        keys.append((j, (a, b, c)))
    return keys

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j, key in _line_key(i, pts):
            if j > i:
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def construct():
    deadline = time.time() + 145
    
    # Initial configurations
    configs = [
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]],
        [[x, y] for x in range(4) for y in range(3)],
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]], # 3x4
        [[x, y] for x in range(2) for y in range(6)], # 2x6
        # Triangle + midpoints scaled and duplicated?
        [[0,0],[2,0],[0,2],[1,0],[0,1],[1,1], [4,0],[6,0],[4,2],[5,0],[4,1],[5,1]],
    ]
    
    best_pts = None
    best_score = float('inf')
    
    for config in configs:
        if not _is_valid(config):
            continue
        score = _count_ordinary(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
            
    if best_score <= 6:
        return [list(p) for p in best_pts]
        
    # Fast local search
    # We will maintain line_map and counts to speed up evaluation
    # But implementing full incremental update is complex.
    # Instead, we use a faster _count_ordinary by pre-computing line keys for static points?
    # No, points move.
    
    # Let's just run a more aggressive random restart local search
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Start from best or random
        if restart == 0:
            current_pts = [list(p) for p in best_pts]
        else:
            # Random cluster
            cx = random.randint(-50, 50)
            cy = random.randint(-50, 50)
            current_pts = [[cx + random.randint(-10, 10), cy + random.randint(-10, 10)] for _ in range(12)]
            # Ensure distinct
            valid = False
            for _ in range(100):
                if _is_valid(current_pts):
                    valid = True
                    break
                current_pts = [[cx + random.randint(-10, 10), cy + random.randint(-10, 10)] for _ in range(12)]
            if not valid:
                continue
                
        current_score = _count_ordinary(current_pts)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(12):
                if time.time() > deadline:
                    break
                orig_x, orig_y = current_pts[idx]
                for dx in range(-10, 11):
                    if time.time() > deadline:
                        break
                    for dy in range(-10, 11):
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = orig_x + dx, orig_y + dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        new_pts = [list(p) for p in current_pts]
                        new_pts[idx] = [new_x, new_y]
                        if not _is_valid(new_pts):
                            continue
                        new_score = _count_ordinary(new_pts)
                        if new_score < current_score:
                            current_pts = new_pts
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        if current_score < best_score:
            best_score = current_score
            best_pts = [list(p) for p in current_pts]
            if best_score <= 6:
                return [list(p) for p in best_pts]
                
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
