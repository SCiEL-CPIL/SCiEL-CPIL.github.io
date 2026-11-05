import time
import random
from math import gcd
from itertools import combinations, product

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)

def _collinear_check(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True

def _valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)

def _generate_line_pool():
    """
    Generate a pool of lines that contain at least 3 points from a small grid.
    We use a grid from -10 to 10 to capture diverse slopes and intercepts.
    """
    grid_points = [(x, y) for x in range(-10, 11) for y in range(-10, 11)]
    lines = {}
    
    # Map from line key to list of points on that line
    for i in range(len(grid_points)):
        x1, y1 = grid_points[i]
        for j in range(i + 1, len(grid_points)):
            x2, y2 = grid_points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = []
            lines[key].append((x1, y1))
            
    # Filter to lines with at least 3 points
    # Note: We store the set of points for each line
    rich_lines = []
    for key, pts in lines.items():
        if len(pts) >= 3:
            rich_lines.append((key, set(pts)))
            
    return rich_lines

def _search_configuration(rich_lines, time_limit=10.0):
    """
    Search for a 12-point configuration by combining points from rich lines.
    Strategy: 
    1. Select a subset of rich lines.
    2. Try to pick 12 distinct points from the union of these lines.
    3. Minimize ordinary lines.
    
    To be efficient, we prioritize lines with more points.
    We will try combinations of 2, 3, 4, 5 rich lines.
    """
    best_score = float('inf')
    best_pts = None
    
    # Sort lines by number of points (descending) to find dense clusters quickly
    rich_lines.sort(key=lambda x: len(x[1]), reverse=True)
    
    # We only need to consider a subset of the most promising lines
    # Lines with very many points are good, but we need to ensure we can pick 12 distinct points.
    # Let's take the top 100 lines.
    candidate_lines = rich_lines[:100]
    
    # Try combinations of lines
    # A configuration with 6 ordinary lines likely involves a few dense lines.
    # Max number of lines needed: If we have 12 points, and min 3 per line, max 4 lines.
    # But intersections might allow more. Let's try up to 6 lines.
    
    for k in range(2, 7):
        if time.time() > time_limit:
            break
        for combo in combinations(range(len(candidate_lines)), k):
            if time.time() > time_limit:
                break
            union_pts = set()
            for idx in combo:
                union_pts.update(candidate_lines[idx][1])
            
            if len(union_pts) < 12:
                continue
                
            # We have a candidate set of points. We need to pick 12.
            # To minimize ordinary lines, we should pick points that lie on the selected lines.
            # Actually, any 12 points from the union will have the property that any line containing 3+ of them is one of the selected lines (or a sub-line, but we selected lines with 3+ pts in the grid, so sub-lines are covered).
            # Wait, if we pick points from the union, a line passing through 3 of them might NOT be one of the selected lines if the 3 points are not collinear in the original grid sense? No, if they are collinear, they lie on a unique line. If that line has 3 points in the grid, it's in our pool. If we didn't select it, it's not "protected".
            
            # So, simply picking any 12 points from the union is not enough. We must ensure that we don't create "new" ordinary lines.
            # However, if we pick points that are "dense" on the selected lines, the number of ordinary lines is determined by the pairs that don't form 3-collinear sets.
            
            # Heuristic: Pick points that are shared by multiple selected lines first (intersections).
            # Then fill up with points from the lines.
            
            # For simplicity and speed, let's just try a few random subsets of 12 from the union.
            # Or better: greedy selection.
            
            # Let's just evaluate the best 12-point subset of this union.
            # This is still hard. 
            # Alternative: If the union has exactly 12 points, evaluate it.
            if len(union_pts) == 12:
                pts = list(union_pts)
                score = _count_ordinary(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts
            else:
                # Try to find a 12-point subset with low score.
                # We can use a quick local search on this small set.
                # Or just try a few random samples.
                for _ in range(10):
                    if time.time() > time_limit:
                        break
                    sample = random.sample(list(union_pts), 12)
                    score = _count_ordinary(sample)
                    if score < best_score:
                        best_score = score
                        best_pts = sample
                        
    return best_pts, best_score

def construct():
    best_pts = None
    best_score = float('inf')
    
    # 1. Try known good configurations (Grids, etc.)
    # 4x3 grid
    cfg1 = [[x, y] for x in range(4) for y in range(3)]
    if _valid(cfg1):
        s = _count_ordinary(cfg1)
        if s < best_score:
            best_score = s
            best_pts = cfg1
            
    # 3x4 grid
    cfg2 = [[x, y] for x in range(3) for y in range(4)]
    if _valid(cfg2):
        s = _count_ordinary(cfg2)
        if s < best_score:
            best_score = s
            best_pts = cfg2
            
    # 2. Structural Search using Line Pool
    if time.time() < _DEADLINE - 20:
        rich_lines = _generate_line_pool()
        pts, score = _search_configuration(rich_lines, time_limit=_DEADLINE - 25)
        if pts and score < best_score:
            best_pts = pts
            best_score = score
            
    # 3. Local Search Polish
    if best_pts and time.time() < _DEADLINE - 5:
        # Simple hill climbing on the best found configuration
        cur = [p[:] for p in best_pts]
        cur_score = best_score
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            for i in range(12):
                # Try moving point i to a nearby point
                x, y = cur[i]
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = x + dx, y + dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        # Check distinctness
                        if any(p == [new_x, new_y] for j, p in enumerate(cur) if j != i):
                            continue
                        new_pts = [p[:] for p in cur]
                        new_pts[i] = [new_x, new_y]
                        if not _valid(new_pts):
                            continue
                        new_score = _count_ordinary(new_pts)
                        if new_score < cur_score:
                            cur = new_pts
                            cur_score = new_score
                            improved = True
                            if cur_score < best_score:
                                best_score = cur_score
                                best_pts = [p[:] for p in cur]
                            break
                    if improved:
                        break
                if improved:
                    break

    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))

