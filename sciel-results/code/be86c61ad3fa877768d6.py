"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_four_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    four_count = 0
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # canonical direction
            g = 0
            a, b = dx, dy
            while a != 0 or b != 0:
                if a % 2 == 0 and b % 2 == 0:
                    a //= 2
                    b //= 2
                else:
                    break
            # use gcd
            from math import gcd
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            dxn = dx // g
            dyn = dy // g
            # canonical sign
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
            key = (x1, y1, dxn, dyn)
            if key in seen:
                continue
            seen.add(key)
            # count points on this line
            cnt = 0
            for k in range(n):
                px, py = points[k]
                # check if (px,py) is on line through (x1,y1) with direction (dxn,dyn)
                # cross product: (px-x1)*dyn - (py-y1)*dxn == 0
                if (px - x1) * dyn - (py - y1) * dxn == 0:
                    cnt += 1
            if cnt == 4:
                four_count += 1
    return four_count

def _has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    from math import gcd
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            dxn = dx // g
            dyn = dy // g
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
            key = (x1, y1, dxn, dyn)
            if key in seen:
                continue
            seen.add(key)
            cnt = 0
            for k in range(n):
                px, py = points[k]
                if (px - x1) * dyn - (py - y1) * dxn == 0:
                    cnt += 1
            if cnt >= 5:
                return True
    return False

def _is_valid(points):
    """Check: 16 distinct points, |x|,|y|<=1000, no five collinear."""
    if len(points) != 16:
        return False
    seen_pts = set()
    for p in points:
        if p in seen_pts:
            return False
        seen_pts.add(tuple(p))
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return False
    return not _has_five_collinear(points)

def _line_count_fast(points):
    """Count 4-point lines using line signature grouping."""
    n = len(points)
    from math import gcd
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            dxn = dx // g
            dyn = dy // g
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
            # line key: (dxn, dyn, c) where c = dyn*x1 - dxn*y1 (normal form)
            c = dyn * x1 - dxn * y1
            key = (dxn, dyn, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    four_count = 0
    for key, pts in lines.items():
        if len(pts) == 4:
            four_count += 1
    return four_count

def _try_move(points, idx, new_pt, line_cache):
    """Try moving point idx to new_pt. Returns (new_score, new_points) or (None, None)."""
    # Check distinctness
    for k, p in enumerate(points):
        if k != idx and (p[0], p[1]) == (new_pt[0], new_pt[1]):
            return None, None
    if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
        return None, None
    
    new_points = [list(p) for p in points]
    new_points[idx] = list(new_pt)
    
    # Check no five collinear
    if _has_five_collinear(new_points):
        return None, None
    
    score = _line_count_fast(new_points)
    return score, new_points

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Start from 4x4 grid (known 20 four-point lines)
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _line_count_fast(best_points)
    
    # Local search with simulated annealing-like acceptance
    rng = random.Random(42)
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    # Precompute candidate positions for each point
    # For each point, try nearby positions
    def get_candidates(pt, rng):
        x, y = pt[0], pt[1]
        candidates = []
        # Random offset within reasonable range
        for _ in range(30):
            nx = x + rng.randint(-3, 3)
            ny = y + rng.randint(-3, 3)
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                candidates.append([nx, ny])
        # Also try some larger jumps
        for _ in range(10):
            nx = x + rng.randint(-10, 10)
            ny = y + rng.randint(-10, 10)
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                candidates.append([nx, ny])
        return candidates
    
    temp = 5.0  # temperature for accepting worse moves
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = rng.randint(0, 15)
        candidates = get_candidates(current_points[idx], rng)
        rng.shuffle(candidates)
        
        improved = False
        for new_pt in candidates:
            if time.time() > deadline - 5:
                break
            score, new_pts = _try_move(current_points, idx, new_pt, None)
            if score is None:
                continue
            delta = score - current_score
            if delta >= 0:
                # Accept
                current_points = new_pts
                current_score = score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
                break
            else:
                # Accept with probability exp(delta/temp)
                import math
                if temp > 0.01 and rng.random() < math.exp(delta / temp):
                    current_points = new_pts
                    current_score = score
                    improved = True
                    break
        
        # Occasionally do a larger perturbation to escape local optima
        if not improved and rng.random() < 0.15:
            # Move 2-3 random points to new random positions
            for _ in range(rng.randint(2, 3)):
                idx2 = rng.randint(0, 15)
                nx = rng.randint(0, 6)
                ny = rng.randint(0, 6)
                new_pt = [nx, ny]
                score, new_pts = _try_move(current_points, idx2, new_pt, None)
                if score is not None:
                    current_points = new_pts
                    current_score = score
        
        temp *= cooling
    
    # Final polish: try to improve best_points with more focused search
    if time.time() < deadline - 10:
        polish_points = [list(p) for p in best_points]
        polish_score = best_score
        polish_deadline = min(deadline, time.time() + 10)
        
        for _ in range(1000):
            if time.time() > polish_deadline:
                break
            idx = rng.randint(0, 15)
            x, y = polish_points[idx][0], polish_points[idx][1]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [x + dx, y + dy]
                    score, new_pts = _try_move(polish_points, idx, new_pt, None)
                    if score is not None and score > polish_score:
                        polish_points = new_pts
                        polish_score = score
                        break
        
        if polish_score > best_score:
            best_score = polish_score
            best_points = polish_points
    
    # Validate final result
    if not _is_valid(best_points):
        # Fallback to grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
