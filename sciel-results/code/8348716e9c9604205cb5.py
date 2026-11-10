"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from math import gcd

# EVOLVE-BLOCK-START

def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Map each line (defined by a pair) to the set of point indices on it.
    # A line is ordinary if it contains exactly 2 points.
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(abs(A), abs(B))
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    ordinary = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary


def _count_ordinary_fast(points):
    """Faster count using sorted line keys and point-count arrays."""
    n = len(points)
    if n < 2:
        return 0
    # Collect all line keys with their point sets
    line_map = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(abs(A), abs(B))
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    ordinary = 0
    for pts in line_map.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary


def _local_search(points, max_iters=5000, timeout=10.0):
    """Improve a configuration by moving individual points to nearby integer positions
    or to positions on existing lines (to create new collinearities)."""
    start = time.time()
    best = [list(p) for p in points]
    best_score = _count_ordinary(best)
    n = len(best)
    no_improve_count = 0
    max_no_improve = 200
    
    while True:
        if time.time() - start > timeout:
            break
        if no_improve_count > max_no_improve:
            break
            
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = best[idx]
        
        # Build candidate moves: nearby positions + points on existing lines
        candidate_moves = set()
        
        # Nearby positions (wider range)
        radius = 5
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                candidate_moves.add((old_x + dx, old_y + dy))
        
        # Points on existing lines (to create new collinearities)
        # For each pair of other points, find integer points on that line
        for i in range(n):
            if i == idx:
                continue
            for j in range(i + 1, n):
                if j == idx:
                    continue
                x1, y1 = best[i]
                x2, y2 = best[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                step_x = dx // g
                step_y = dy // g
                # Points on the line segment and slightly beyond
                for t in range(-2, g + 3):
                    px = x1 + t * step_x
                    py = y1 + t * step_y
                    if abs(px) <= 10**9 and abs(py) <= 10**9:
                        if (px, py) != (old_x, old_y):
                            candidate_moves.add((px, py))
        
        # Shuffle and try candidates
        moves_list = list(candidate_moves)
        random.shuffle(moves_list)
        
        improved = False
        for new_x, new_y in moves_list:
            if time.time() - start > timeout:
                break
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            # Check distinctness
            dup = False
            for j, (px, py) in enumerate(best):
                if j != idx and px == new_x and py == new_y:
                    dup = True
                    break
            if dup:
                continue
            # Quick check: not all collinear
            candidate = [list(p) for p in best]
            candidate[idx] = [new_x, new_y]
            if _all_collinear(candidate):
                continue
            score = _count_ordinary(candidate)
            if score < best_score:
                best = candidate
                best_score = score
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
    return best, best_score


def _all_collinear(points):
    if len(points) < 3:
        return False
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, len(points)):
        x2, y2 = points[i]
        if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
            return False
    return True


def construct():
    """Return 12 integer points minimizing ordinary lines."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s hard kill
    
    # Structured candidate configurations
    candidates = []
    
    # 3x4 grid (baseline)
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 6x2 grid
    candidates.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # 2x6 grid
    candidates.append([[0, y] for y in range(6)] + [[1, y] for y in range(6)])
    
    # Diagonal pencil: 3 points on each of 4 parallel diagonals
    candidates.append([
        [0, 0], [1, 1], [2, 2],
        [0, 1], [1, 2], [2, 3],
        [0, -1], [1, 0], [2, 1],
        [0, 2], [1, 3], [2, 4],
    ])
    
    # 3x3 grid + 3 extra
    candidates.append([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [3, 0], [3, 1], [3, 2],
    ])
    
    # Hexagonal arrangement
    candidates.append([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [1, 2], [2, 2], [3, 2], [4, 2],
    ])
    
    # More random structured starts
    for _ in range(30):
        pts = []
        for _ in range(12):
            pts.append([random.randint(-6, 6), random.randint(-6, 6)])
        seen = set()
        unique_pts = []
        for p in pts:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique_pts.append(p)
        if len(unique_pts) == 12 and not _all_collinear(unique_pts):
            candidates.append(unique_pts)
    
    # Evaluate all candidates
    best_points = None
    best_score = float('inf')
    scored_candidates = []
    
    for cand in candidates:
        if time.time() > deadline:
            break
        seen = set()
        unique = []
        for p in cand:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        if len(unique) != 12:
            continue
        if _all_collinear(unique):
            continue
        score = _count_ordinary(unique)
        scored_candidates.append((score, unique))
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in unique]
    
    # Sort candidates by initial score and run local search on the best ones
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run local search on top candidates with time budget
    remaining = deadline - time.time() - 5
    if remaining > 10:
        # Allocate time to top candidates
        top_n = min(5, len(scored_candidates))
        time_per_candidate = max(5, remaining / top_n)
        
        for score_init, cand in scored_candidates[:top_n]:
            if time.time() > deadline - 5:
                break
            remaining_for_this = min(time_per_candidate, deadline - time.time() - 5)
            if remaining_for_this < 3:
                break
            improved, sc = _local_search(cand, max_iters=10000, timeout=remaining_for_this)
            if sc < best_score:
                best_score = sc
                best_points = improved
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points

# EVOLVE-BLOCK-END
