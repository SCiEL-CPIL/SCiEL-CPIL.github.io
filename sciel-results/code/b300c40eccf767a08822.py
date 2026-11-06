"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _line_key(i, pts):
    """Compute line keys for point i against all other points. Returns list of (j, key)."""
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

def _count_ordinary_fast(pts):
    """Optimized count using dict of sets, built in one pass."""
    n = len(pts)
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
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
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = [i, j]
            else:
                line_map[key].append(j)  # i is already there from earlier pair
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
    # Check not all collinear
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def _try_move(pts, idx, new_x, new_y):
    """Try moving point idx to (new_x, new_y), return new pts if valid."""
    if new_x == pts[idx][0] and new_y == pts[idx][1]:
        return None
    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
        return None
    # Check duplicate
    for i, p in enumerate(pts):
        if i != idx and p[0] == new_x and p[1] == new_y:
            return None
    new_pts = [list(p) for p in pts]
    new_pts[idx] = [new_x, new_y]
    return new_pts

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try multiple initial configurations and pick the best
    initial_configs = []
    
    # Config 1: 3x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Config 2: 2x6 grid
    initial_configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # Config 3: Points on 2 parallel lines with 6 each
    initial_configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # Config 4: Regular hexagon vertices + center + 3 more
    # Use points that create many collinear triples
    # 6 points on a line, 6 on another line, with intersections
    initial_configs.append(
        [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],
         [0,1],[2,1],[4,1],[1,2],[3,2],[5,2]]
    )
    
    # Config 5: 3 lines of 4 points each (triangle with midpoints)
    initial_configs.append(
        [[0,0],[1,0],[2,0],[3,0],
         [0,0],[0,1],[0,2],[0,3],
         [3,3],[2,2],[1,1],[0,0]]  # has duplicates, will be filtered
    )
    
    # Config 6: 4 points on each of 3 concurrent lines through origin
    # Line 1: y=0: (1,0),(2,0),(3,0),(4,0)
    # Line 2: y=x: (1,1),(2,2),(3,3),(4,4)
    # Line 3: y=2x: (1,2),(2,4),(3,6),(4,8)
    initial_configs.append(
        [[1,0],[2,0],[3,0],[4,0],
         [1,1],[2,2],[3,3],[4,4],
         [1,2],[2,4],[3,6],[4,8]]
    )
    
    # Config 7: 12 points on a circle (no 3 collinear generally, but let's check)
    # Use rational approximations of cos/sin
    initial_configs.append(
        [[10,0],[7,7],[0,10],[-7,7],[-10,0],[-7,-7],[0,-10],[7,-7],
         [10,3],[3,10],[-10,3],[-3,10]]
    )
    
    # Config 8: 4x3 grid variant
    initial_configs.append(
        [[0,0],[1,0],[2,0],[3,0],
         [0,1],[1,1],[2,1],[3,1],
         [0,2],[1,2],[2,2],[3,2]]
    )
    
    best_pts = None
    best_score = float('inf')
    
    for config in initial_configs:
        if not _is_valid(config):
            continue
        score = _count_ordinary_fast(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
    
    # If no valid config found, use fallback
    if best_pts is None:
        best_pts = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
        best_score = _count_ordinary_fast(best_pts)
    
    # If we already have the theoretical minimum, return immediately
    if best_score <= 6:
        return [list(p) for p in best_pts]
    
    # Local search with incremental evaluation
    random.seed(42)
    
    # Precompute line membership for fast delta evaluation
    def _build_line_membership(pts):
        """Build mapping: point_index -> set of (line_key, other_points_on_line)"""
        n = len(pts)
        line_map = {}
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
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
                key = (a, b, c)
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map
    
    def _delta_score(pts, idx, new_x, new_y):
        """Compute change in ordinary line count when moving point idx to (new_x, new_y)."""
        n = len(pts)
        line_map = _build_line_membership(pts)
        
        # Remove all lines involving point idx
        old_lines_with_idx = {}
        for key, members in line_map.items():
            if idx in members:
                old_lines_with_idx[key] = members.copy()
        
        # Build new line map without point idx
        new_line_map = {}
        for key, members in line_map.items():
            if idx in members:
                new_members = members - {idx}
                if len(new_members) >= 2:
                    new_line_map[key] = new_members
            else:
                new_line_map[key] = members
        
        # Add new lines from point idx to all other points
        for j in range(n):
            if j == idx:
                continue
            x1, y1 = new_x, new_y
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
            key = (a, b, c)
            if key not in new_line_map:
                new_line_map[key] = set()
            new_line_map[key].add(idx)
            new_line_map[key].add(j)
        
        # Count ordinary lines in old and new
        old_ordinary = sum(1 for v in line_map.values() if len(v) == 2)
        new_ordinary = sum(1 for v in new_line_map.values() if len(v) == 2)
        
        return new_ordinary - old_ordinary
    
    # Phase 1: Random moves with delta evaluation
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 11)
        orig_x, orig_y = best_pts[idx]
        
        for _ in range(3):
            step = random.randint(1, 20)
            new_x = orig_x + random.randint(-step, step)
            new_y = orig_y + random.randint(-step, step)
            
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            if new_x == orig_x and new_y == orig_y:
                continue
            # Check duplicate
            dup = False
            for i, p in enumerate(best_pts):
                if i != idx and p[0] == new_x and p[1] == new_y:
                    dup = True
                    break
            if dup:
                continue
            
            delta = _delta_score(best_pts, idx, new_x, new_y)
            if delta < 0:
                best_pts[idx] = [new_x, new_y]
                best_score += delta
                break
    
    # Phase 2: Systematic hill climbing with delta evaluation
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(12):
            if time.time() > deadline:
                break
            orig_x, orig_y = best_pts[idx]
            for dx in range(-5, 6):
                if time.time() > deadline:
                    break
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = orig_x + dx, orig_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    dup = False
                    for i, p in enumerate(best_pts):
                        if i != idx and p[0] == new_x and p[1] == new_y:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    delta = _delta_score(best_pts, idx, new_x, new_y)
                    if delta < 0:
                        best_pts[idx] = [new_x, new_y]
                        best_score += delta
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
