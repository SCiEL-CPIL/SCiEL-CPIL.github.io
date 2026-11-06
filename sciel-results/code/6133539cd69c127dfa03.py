"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _dist_sq(a, b, c, d):
    """Squared distance between (a,b) and (c,d) on triangular lattice."""
    da = a - c
    db = b - d
    return da*da + da*db + db*db

def _count_unit_edges(points):
    """Count pairs at distance exactly 1 (squared distance = 1)."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if _dist_sq(points[i][0], points[i][1], points[j][0], points[j][1]) == 1:
                count += 1
    return count

def _min_dist_sq(points):
    """Find minimum squared distance between any pair."""
    n = len(points)
    min_d = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = _dist_sq(points[i][0], points[i][1], points[j][0], points[j][1])
            if d < min_d:
                min_d = d
    return min_d

def _is_valid(points):
    """Check all points are distinct and at distance >= 1."""
    n = len(points)
    seen = set()
    for p in points:
        if tuple(p) in seen:
            return False
        seen.add(tuple(p))
    return _min_dist_sq(points) >= 1

def _hex_spiral(n):
    """Generate n points in a hexagonal spiral pattern on triangular lattice."""
    points = [(0, 0)]
    if n <= 1:
        return points
    
    # Hexagonal spiral: expand outward in rings
    # Ring k has 6k points
    ring = 1
    while len(points) < n:
        # Generate ring k points
        # Ring k consists of 6 segments of length k
        for side in range(6):
            # Direction vectors for each side of the hexagon
            dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
            dx, dy = dirs[side]
            # Start point for this side
            # Corners of ring k hexagon
            corners = [
                (ring, 0), (0, ring), (-ring, ring),
                (-ring, 0), (0, -ring), (ring, -ring)
            ]
            sx, sy = corners[side]
            for step in range(ring):
                px = sx + step * dx
                py = sy + step * dy
                if (px, py) not in points:
                    points.append((px, py))
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        ring += 1
    return points[:n]

def _local_search(points, deadline):
    """Local search to maximize unit edges while maintaining validity."""
    n = len(points)
    best_points = [list(p) for p in points]
    best_score = _count_unit_edges(best_points)
    
    # Neighbor moves: try moving a point to a nearby lattice position
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1),
             (-2,0),(2,0),(0,-2),(0,2),(-2,2),(2,-2),
             (-1,-1),(1,1),(2,-1),(-2,1),(1,-2),(-1,2)]
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            for dx, dy in moves:
                new_point = [best_points[i][0] + dx, best_points[i][1] + dy]
                if abs(new_point[0]) > 64 or abs(new_point[1]) > 64:
                    continue
                # Check if this position is already occupied
                if new_point in best_points:
                    continue
                # Try the move
                old = best_points[i]
                best_points[i] = new_point
                if _is_valid(best_points):
                    score = _count_unit_edges(best_points)
                    if score > best_score:
                        best_score = score
                        improved = True
                    else:
                        best_points[i] = old
                else:
                    best_points[i] = old
        if not improved:
            # Try random perturbations
            for _ in range(100):
                i = random.randint(0, n-1)
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                new_point = [best_points[i][0] + dx, best_points[i][1] + dy]
                if abs(new_point[0]) > 64 or abs(new_point[1]) > 64:
                    continue
                if new_point in best_points:
                    continue
                old = best_points[i]
                best_points[i] = new_point
                if _is_valid(best_points):
                    score = _count_unit_edges(best_points)
                    if score > best_score:
                        best_score = score
                        improved = True
                    else:
                        best_points[i] = old
                else:
                    best_points[i] = old
            if not improved:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    n = 32
    
    # Try multiple initial configurations and local searches
    best_result = None
    best_score = -1
    
    # Initial config 1: hexagonal spiral
    initial1 = _hex_spiral(n)
    result1, score1 = _local_search(initial1, deadline)
    if score1 > best_score:
        best_score = score1
        best_result = result1
    
    if time.time() >= deadline:
        return best_result
    
    # Initial config 2: compact rectangular-ish shape
    initial2 = []
    # Try a 6x6 block minus corners to get 32
    for a in range(-3, 3):
        for b in range(-3, 3):
            initial2.append([a, b])
            if len(initial2) >= n:
                break
        if len(initial2) >= n:
            break
    initial2 = initial2[:n]
    result2, score2 = _local_search(initial2, deadline)
    if score2 > best_score:
        best_score = score2
        best_result = result2
    
    if time.time() >= deadline:
        return best_result
    
    # Initial config 3: another spiral variant
    random.seed(42)
    initial3 = []
    used = set()
    # Random walk to build compact cluster
    initial3.append([0, 0])
    used.add((0, 0))
    while len(initial3) < n:
        # Pick a random existing point and try to add a neighbor
        base = random.choice(initial3)
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
            new = (base[0]+dx, base[1]+dy)
            if new not in used:
                initial3.append(list(new))
                used.add(new)
                break
        else:
            # Try farther
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    new = (base[0]+dx, base[1]+dy)
                    if new not in used:
                        initial3.append(list(new))
                        used.add(new)
                        break
                else:
                    continue
                break
    
    result3, score3 = _local_search(initial3, deadline)
    if score3 > best_score:
        best_score = score3
        best_result = result3
    
    return best_result
# EVOLVE-BLOCK-END
