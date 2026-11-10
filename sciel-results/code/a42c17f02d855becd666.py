import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Helper to count ordinary lines
    def count_ordinary_lines(points):
        n = len(points)
        if n < 2:
            return 0
        line_points = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    key = (0, 0, x1)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx //= g
                    dy //= g
                    if dx < 0:
                        dx = -dx
                        dy = -dy
                    c = dy * x1 - dx * y1
                    key = (dx, dy, c)
                if key not in line_points:
                    line_points[key] = []
                line_points[key].append((i, j))
        
        # Map line keys to sets of point indices
        line_indices = {}
        for key, pairs in line_points.items():
            idxs = set()
            for i, j in pairs:
                idxs.add(i)
                idxs.add(j)
            line_indices[key] = idxs
        
        ordinary_count = 0
        for key, idxs in line_indices.items():
            if len(idxs) == 2:
                ordinary_count += 1
        return ordinary_count

    def is_valid(points):
        if len(points) != 12:
            return False
        if len(set(points)) != 12:
            return False
        # Check bounds
        for p in points:
            if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
                return False
        # Check not all collinear
        if len(points) >= 3:
            p1, p2, p3 = points[0], points[1], points[2]
            (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
            if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                # Check if all are collinear
                for i in range(3, 12):
                    xi, yi = points[i]
                    if (x2-x1)*(yi-y1) != (y2-y1)*(xi-x1):
                        break
                else:
                    return False
        return True

    def local_search(points, max_iters=5000):
        current = [p[:] for p in points]
        current_score = count_ordinary_lines(current)
        best = current
        best_score = current_score
        best_time = time.time()
        
        while time.time() < deadline - 1.0:
            improved = False
            # Try moving each point
            for i in range(12):
                if time.time() > deadline - 1.0:
                    break
                x, y = current[i]
                # Try small perturbations
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = x + dx, y + dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        # Check distinctness
                        new_point = [new_x, new_y]
                        if any(new_point == p for p in current[:i] + current[i+1:]):
                            continue
                        
                        old_p = current[i]
                        current[i] = new_point
                        score = count_ordinary_lines(current)
                        if score < current_score:
                            current_score = score
                            improved = True
                            if score < best_score:
                                best_score = score
                                best = [p[:] for p in current]
                                best_time = time.time()
                        else:
                            current[i] = old_p
                            # If no improvement in this step, revert
                            # Actually, standard local search reverts if worse
                            # We only keep if strictly better
                            
            if not improved:
                # Stuck, try random restart or larger jumps
                # Randomly move a point to a new random location within bounds
                i = random.randint(0, 11)
                new_x = random.randint(-1000, 1000)
                new_y = random.randint(-1000, 1000)
                if any([new_x, new_y] == p for p in current[:i] + current[i+1:]):
                    continue
                current[i] = [new_x, new_y]
                score = count_ordinary_lines(current)
                if score < current_score:
                    current_score = score
                    if score < best_score:
                        best_score = score
                        best = [p[:] for p in current]
                        best_time = time.time()
                else:
                    # Revert
                    current[i] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
                    # This is a bit messy, let's just break and restart from best
                    break
        
        return best, best_score

    # Initial candidate: Böröczky-like configuration
    # The Böröczky configuration for 12 points typically consists of:
    # 6 points on a conic (e.g., circle or hyperbola) and 6 points at infinity (in projective plane).
    # In affine integer coordinates, "points at infinity" are tricky. 
    # However, we can use a configuration where many points are collinear or on a conic.
    # A known construction for 12 points with 6 ordinary lines:
    # Take 3 lines, each containing 4 points. If these 3 lines are not concurrent and no other 3 points are collinear, 
    # let's count.
    # Lines L1, L2, L3.
    # Points on L1: A1, A2, A3, A4
    # Points on L2: B1, B2, B3, B4
    # Points on L3: C1, C2, C3, C4
    # Total 12 points.
    # Ordinary lines are lines containing exactly 2 points.
    # The lines L1, L2, L3 are not ordinary (they have 4).
    # Any line connecting a point from L1 and a point from L2 is ordinary unless it passes through a point on L3.
    # To minimize ordinary lines, we want many such connecting lines to be non-ordinary.
    # This is related to the number of "accidental" collinearities.
    
    # Let's try a specific grid-like or structured set.
    # The 3x4 grid has many ordinary lines.
    # Let's try the vertices of a regular dodecagon? No, integer coordinates.
    
    # Let's use the local search starting from a few diverse initial configurations.
    
    # Config 1: 3x4 Grid
    pts1 = [[x, y] for x in range(4) for y in range(3)]
    
    # Config 2: 4x3 Grid
    pts2 = [[x, y] for x in range(3) for y in range(4)]
    
    # Config 3: Two parallel lines with 6 points each?
    # (0,0)..(5,0) and (0,1)..(5,1)
    pts3 = [[x, 0] for x in range(6)] + [[x, 1] for x in range(6)]
    
    # Config 4: Three lines forming a triangle, 4 points on each side?
    # This is hard to construct simply.
    
    # Config 5: Random
    pts5 = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(12)]
    
    best_overall = None
    best_score_overall = float('inf')
    
    configs = [pts1, pts2, pts3, pts5]
    
    for cfg in configs:
        if time.time() > deadline - 5.0:
            break
        # Ensure validity
        if not is_valid(cfg):
            continue
        best, score = local_search(cfg)
        if score < best_score_overall:
            best_score_overall = score
            best_overall = best
            
        # If we hit the theoretical minimum, stop
        if best_score_overall == 6:
            break
            
    if best_overall is None:
        # Fallback
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_overall
