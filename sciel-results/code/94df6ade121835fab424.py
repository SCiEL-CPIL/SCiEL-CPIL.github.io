"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _count_ordinary(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            a = dy
            b = -dx
            c = -a * x1 - b * y1
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
    
    ordinary = 0
    for key, cnt in lines.items():
        if cnt == 1:  # exactly 2 points on this line
            ordinary += 1
    return ordinary

def _count_ordinary_fast(points):
    """Faster count using set-based line identification."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Canonical form: ensure dx >= 0, or dx==0 and dy > 0
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line key: (normalized direction, point that is "first" in some sense)
            # Use the point with smaller x (or smaller y if x equal) as anchor
            if x1 < x2 or (x1 == x2 and y1 < y2):
                ax, ay = x1, y1
            else:
                ax, ay = x2, y2
            key = (dx, dy, ax, ay)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for key, cnt in line_counts.items():
        if cnt == 1:
            ordinary += 1
    return ordinary

def _valid(points):
    """Check if points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p, list) or len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    
    # Check not all collinear
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    # Cross product to check collinearity
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if cross == 0:
        # Check if all other points are also collinear
        for i in range(3, 12):
            xi, yi = points[i]
            cross_i = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
            if cross_i != 0:
                return True
        return False
    return True

def _generate_strategic_candidates(points, idx, existing_candidates):
    """Generate candidate positions that are likely to create 3+ point lines."""
    candidates = set(existing_candidates)
    x, y = points[idx]
    
    # For each pair of other points (j, k), find integer points on line jk
    # that are near (x, y) or at reasonable distances
    for j in range(12):
        if j == idx:
            continue
        xj, yj = points[j]
        for k in range(j + 1, 12):
            if k == idx:
                continue
            xk, yk = points[k]
            dx = xk - xj
            dy = yk - yj
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dxn = dx // g
            dyn = dy // g
            
            # Points on line jk: (xj + t*dxn, yj + t*dyn) for integer t
            # We want points that are "close" to where point idx currently is
            # or at small integer offsets
            # Find t values that give points near (x, y)
            # Solve approximately: xj + t*dxn ≈ x, yj + t*dyn ≈ y
            # t ≈ (x - xj) / dxn if dxn != 0, else (y - yj) / dyn if dyn != 0
            
            t_candidates = set()
            if dxn != 0:
                t_approx = (x - xj) / dxn
                for t in [int(t_approx) - 2, int(t_approx) - 1, int(t_approx), int(t_approx) + 1, int(t_approx) + 2]:
                    t_candidates.add(t)
            if dyn != 0:
                t_approx = (y - yj) / dyn
                for t in [int(t_approx) - 2, int(t_approx) - 1, int(t_approx), int(t_approx) + 1, int(t_approx) + 2]:
                    t_candidates.add(t)
            
            # Also try small t values for nearby points
            for t in range(-3, 4):
                t_candidates.add(t)
            
            for t in t_candidates:
                nx = xj + t * dxn
                ny = yj + t * dyn
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    candidates.add((nx, ny))
    
    return candidates

def _try_move(points, idx, new_x, new_y):
    """Try moving point idx to (new_x, new_y). Returns (new_points, count) if valid, else None."""
    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
        return None
    
    old_x, old_y = points[idx]
    if old_x == new_x and old_y == new_y:
        return None
    
    # Check if new position collides with another point
    for j in range(12):
        if j == idx:
            continue
        if points[j][0] == new_x and points[j][1] == new_y:
            return None
    
    new_points = [p[:] for p in points]
    new_points[idx] = [new_x, new_y]
    
    if not _valid(new_points):
        return None
    
    # Count ordinary lines
    count = _count_ordinary(new_points)
    return new_points, count

def _delta_count_ordinary(points, idx, new_x, new_y):
    """Compute the change in ordinary line count when moving point idx to (new_x, new_y).
    Returns the new count if valid, else None."""
    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
        return None
    
    old_x, old_y = points[idx]
    if old_x == new_x and old_y == new_y:
        return None
    
    # Check collision
    for j in range(12):
        if j == idx:
            continue
        if points[j][0] == new_x and points[j][1] == new_y:
            return None
    
    new_points = [p[:] for p in points]
    new_points[idx] = [new_x, new_y]
    
    if not _valid(new_points):
        return None
    
    count = _count_ordinary(new_points)
    return count

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Uses local search starting from a 3x4 grid and a line-based configuration.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Initialize with a 3x4 grid (known to have 24 ordinary lines)
    # Also try a line-based configuration (11 on a line, 1 off) which has 11 ordinary lines
    # The line-based one is better (11 < 24), so we start there.
    
    # Configuration A: 11 points on x-axis, 1 off
    points_a = [[i, 0] for i in range(11)]
    points_a.append([0, 1])
    
    # Configuration B: 3x4 grid
    points_b = [[x, y] for x in range(4) for y in range(3)]
    
    # Configuration C: 4x3 grid (same as B but rotated, same count)
    # Configuration D: 2x6 grid?
    points_d = [[x, y] for x in range(6) for y in range(2)]
    
    # Configuration E: 12 points in a 2x2x3 box? No, 2D.
    # Let's try a 3x3 grid + 3 points extending one row
    # (0,0)..(2,2) plus (3,0), (3,1), (3,2) -> This is 4x3 grid (12 points)
    
    # Let's evaluate the initial candidates
    candidates = [points_a, points_b, points_d]
    best_points = None
    best_count = float('inf')
    
    for pts in candidates:
        if not _valid(pts):
            continue
        c = _count_ordinary(pts)
        if c < best_count:
            best_count = c
            best_points = [p[:] for p in pts]
    
    # Local search
    current_points = best_points
    current_count = best_count
    
    # We will try to improve by moving points
    # Candidate positions for a point:
    # 1. Nearby integer coordinates (small range)
    # 2. Coordinates that create collinearity with existing points
    
    # To be efficient, we can generate candidate moves based on existing point structure
    # For each point i, and each other point j, consider lines through i and j.
    # Placing i on a line with j and k (where k is another point) might reduce ordinary lines.
    
    # Simpler approach: Random local search with hill climbing
    # Since n=12 is small, we can afford some exploration.
    
    random.seed(42)
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point
        for i in range(12):
            if time.time() > deadline:
                break
                
            x, y = current_points[i]
            
            # Generate candidate moves
            # 1. Small perturbations
            candidates = set()
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    candidates.add((x + dx, y + dy))
            
            # 2. Align with other points
            # For each other point j, consider moving i such that it lies on a line
            # with j and some other point k, or creates a new collinearity.
            # A good heuristic: try to place i at a position that is collinear with at least 2 other points.
            # For each pair (j, k) where j, k != i:
            # The line through j and k has equation. Can we place i on this line?
            # Yes, but we need integer coordinates.
            # Actually, we can just try positions that are "interesting".
            
            # Let's try specific strategic moves:
            # If we have many points on a line, try to add more to that line.
            # Or if we have a grid, try to extend it.
            
            # For now, just use the perturbation candidates.
            # Also try swapping with another point's position? No, that's just moving.
            
            # Add some random distant points?
            # Maybe not, as we want to keep coordinates small to avoid overflow issues (though 10^9 is fine).
            
            best_move = None
            best_move_count = current_count
            
            for nx, ny in candidates:
                if time.time() > deadline:
                    break
                    
                res = _try_move(current_points, i, nx, ny)
                if res is None:
                    continue
                new_pts, cnt = res
                if cnt < best_move_count:
                    best_move_count = cnt
                    best_move = (new_pts, cnt)
            
            if best_move:
                current_points = best_move[0]
                current_count = best_move[1]
                improved = True
                # If we found a significant improvement, break inner loop to restart from new state?
                # Or continue checking other points?
                # Hill climbing usually accepts the first improvement or the best.
                # Let's accept the best for this point and continue to next point.
                
                if current_count <= 6:
                    # We reached the theoretical minimum
                    return current_points
        
        # If no improvement in a full pass, try a random restart or larger perturbation
        if not improved:
            # Try a larger perturbation
            for i in range(12):
                if time.time() > deadline:
                    break
                x, y = current_points[i]
                for _ in range(50):
                    nx = x + random.randint(-50, 50)
                    ny = y + random.randint(-50, 50)
                    res = _try_move(current_points, i, nx, ny)
                    if res:
                        new_pts, cnt = res
                        if cnt < current_count:
                            current_points = new_pts
                            current_count = cnt
                            improved = True
                            break
            
            if not improved:
                # Stuck. Try a different starting configuration or random restart
                # We can try to randomly generate a new configuration and see if it's better
                # Or just break if we've been stuck for a while?
                # Let's try one random restart
                # Generate a random configuration
                new_pts = []
                for _ in range(12):
                    new_pts.append([random.randint(-10, 10), random.randint(-10, 10)])
                if _valid(new_pts):
                    cnt = _count_ordinary(new_pts)
                    if cnt < current_count:
                        current_points = new_pts
                        current_count = cnt
                        improved = True
                if not improved:
                    # If we are stuck, we might just return the best we have
                    # But let's try a few more random restarts
                    # To avoid infinite loop, we can limit restarts
                    # For now, just break if no improvement after restart
                    # Actually, let's just break the while loop if we've tried a few restarts
                    # We don't have a restart counter, so let's just return if we've been stuck for a while
                    # A simple heuristic: if we've been stuck, return
                    # But we don't know how long we've been stuck.
                    # Let's just break to return the current best.
                    break
    
    return current_points

# EVOLVE-BLOCK-END
