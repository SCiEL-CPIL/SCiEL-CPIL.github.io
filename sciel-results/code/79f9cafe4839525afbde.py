"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Normalize: ensure dx > 0, or dx == 0 and dy > 0
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line key: (dx, dy, c) where c = y*dx - x*dy (normalized)
            # Use a consistent representation
            c = y1 * dx - x1 * dy
            key = (dx, dy, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
    return ordinary

def _hexagon_points(scale):
    """Generate 6 points of a regular hexagon scaled by integer factor."""
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Using integer coordinates: (s, 0), (s/2, s*sqrt(3)/2) - not integer
    # Use a different approach: hexagonal lattice points
    # For integer coords, use points that form nice collinearities
    # Try: center at origin, 6 points at distance r
    # (r, 0), (0, r), (-r, 0), (0, -r), (r, r), (-r, -r) - not regular but has symmetries
    # Better: use a grid-based hexagon approximation
    # Points: (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1) scaled
    base = [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1]
    ]
    return [[x * scale, y * scale] for x, y in base]

def _add_points_for_12(base_points, scale):
    """Add 6 more points to make 12, trying to create more collinearities."""
    points = list(base_points)
    # Try adding points that create 3-point lines
    # Add points along existing lines
    candidates = []
    # Extend lines through pairs of existing points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Try extending in both directions
            for factor in [2, 3, -1, -2]:
                nx = x1 + factor * dx
                ny = y1 + factor * dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    # Check if already exists
                    exists = False
                    for p in points:
                        if p[0] == nx and p[1] == ny:
                            exists = True
                            break
                    if not exists:
                        candidates.append([nx, ny])
    # Also try some standard positions
    extra_positions = [
        [2, 0], [0, 2], [-2, 0], [0, -2], [2, 2], [-1, 1], [1, -1],
        [-2, 2], [2, -2], [-1, -1], [1, 1], [3, 0], [0, 3]
    ]
    for pos in extra_positions:
        scaled = [pos[0] * scale, pos[1] * scale]
        if abs(scaled[0]) <= 10**9 and abs(scaled[1]) <= 10**9:
            exists = False
            for p in points:
                if p[0] == scaled[0] and p[1] == scaled[1]:
                    exists = True
                    break
            if not exists:
                candidates.append(scaled)
    
    # Greedily add 6 points that minimize ordinary lines
    while len(points) < 12 and candidates:
        best_candidate = None
        best_count = float('inf')
        for cand in candidates:
            test_points = points + [cand]
            count = _count_ordinary_lines(test_points)
            if count < best_count:
                best_count = count
                best_candidate = cand
        if best_candidate is not None:
            points.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            break
    
    # If we still don't have 12, add arbitrary distinct points
    while len(points) < 12:
        for i in range(1, 100):
            for j in range(1, 100):
                p = [i * scale, j * scale]
                if abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9:
                    exists = False
                    for q in points:
                        if q[0] == p[0] and q[1] == p[1]:
                            exists = True
                            break
                    if not exists:
                        points.append(p)
                        break
            if len(points) >= 12:
                break
    
    return points[:12]

def _greedy_improve(points, max_iterations=5000, time_limit=10.0):
    """Greedy local search: swap points to reduce ordinary lines."""
    start_time = time.time()
    current = [p[:] for p in points]
    current_count = _count_ordinary_lines(current)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try moving each point to a nearby position
        for i in range(len(current)):
            x, y = current[i]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    # Check distinctness
                    valid = True
                    for j in range(len(current)):
                        if j != i and current[j][0] == new_x and current[j][1] == new_y:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    old_point = current[i][:]
                    current[i] = [new_x, new_y]
                    new_count = _count_ordinary_lines(current)
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                    else:
                        current[i] = old_point
        
        if not improved:
            break
    
    return current, current_count

def _try_configurations(deadline):
    """Try multiple configurations and return the best one."""
    best_points = None
    best_count = float('inf')
    
    # Configuration 1: Hexagonal lattice base
    for scale in [1, 2, 3, 5, 10]:
        if time.time() > deadline:
            break
        base = _hexagon_points(scale)
        points = _add_points_for_12(base, scale)
        if len(points) == 12:
            points, count = _greedy_improve(points, max_iterations=2000, time_limit=5.0)
            if count < best_count:
                best_count = count
                best_points = points
    
    # Configuration 2: Grid-based
    for scale in [1, 2, 3]:
        if time.time() > deadline:
            break
        base = [
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2],
            [-1, 1], [3, 1], [1, 3]
        ]
        points = [[x * scale, y * scale] for x, y in base]
        points, count = _greedy_improve(points, max_iterations=2000, time_limit=5.0)
        if count < best_count:
            best_count = count
            best_points = points
    
    # Configuration 3: Star pattern
    for scale in [1, 2]:
        if time.time() > deadline:
            break
        base = [
            [0, 0],
            [scale, 0], [-scale, 0],
            [0, scale], [0, -scale],
            [scale, scale], [-scale, -scale],
            [scale, -scale], [-scale, scale],
            [2*scale, 0], [0, 2*scale]
        ]
        points = base[:12]
        points, count = _greedy_improve(points, max_iterations=2000, time_limit=5.0)
        if count < best_count:
            best_count = count
            best_points = points
    
    # Configuration 4: Two parallel lines
    for scale in [1, 2]:
        if time.time() > deadline:
            break
        base = []
        # 6 points on y=0
        for i in range(6):
            base.append([i * scale, 0])
        # 6 points on y=scale
        for i in range(6):
            base.append([i * scale, scale])
        points = base[:12]
        points, count = _greedy_improve(points, max_iterations=2000, time_limit=5.0)
        if count < best_count:
            best_count = count
            best_points = points
    
    # Configuration 5: 3x4 grid
    for scale in [1, 2]:
        if time.time() > deadline:
            break
        base = []
        for i in range(3):
            for j in range(4):
                base.append([i * scale, j * scale])
        points = base[:12]
        points, count = _greedy_improve(points, max_iterations=2000, time_limit=5.0)
        if count < best_count:
            best_count = count
            best_points = points
    
    # Configuration 6: 4x3 grid
    for scale in [1, 2]:
        if time.time() > deadline:
            break
        base = []
        for i in range(4):
            for j in range(3):
                base.append([i * scale, j * scale])
        points = base[:12]
        points, count = _greedy_improve(points, max_iterations=2000, time_limit=5.0)
        if count < best_count:
            best_count = count
            best_points = points
    
    # Configuration 7: Hexagon with center
    for scale in [1, 2, 3]:
        if time.time() > deadline:
            break
        # 6 outer + 6 inner or center + others
        base = [
            [0, 0],  # center
            [scale, 0], [-scale, 0],
            [0, scale], [0, -scale],
            [scale, scale], [-scale, -scale],
            [scale, -scale], [-scale, scale],
            [2*scale, 0], [0, 2*scale],
            [-2*scale, 0], [0, -2*scale]
        ]
        points = base[:12]
        points, count = _greedy_improve(points, max_iterations=2000, time_limit=5.0)
        if count < best_count:
            best_count = count
            best_points = points
    
    # Fallback: original configuration
    if best_points is None:
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        best_count = _count_ordinary_lines(best_points)
    
    return best_points, best_count

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Uses combinatorial design with hexagonal lattice and greedy local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points, best_count = _try_configurations(deadline)
    
    # Final check
    if time.time() > deadline:
        # Return best found so far
        pass
    
    # Ensure exactly 12 distinct points
    seen = set()
    unique_points = []
    for p in best_points:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    # If we don't have 12, add more
    while len(unique_points) < 12:
        for i in range(1, 1000):
            for j in range(1, 1000):
                p = [i, j]
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    unique_points.append(p)
                    break
            if len(unique_points) >= 12:
                break
    
    return unique_points[:12]
# EVOLVE-BLOCK-END
