"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary_fast(points):
    """Count ordinary lines using a dict of line->count."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _generate_structured_candidates():
    """Generate a diverse set of structured configurations."""
    configs = []
    
    # Configuration 1: 4+4+4 on three parallel lines
    for offset in range(-2, 3):
        c = [[0,0],[1,0],[2,0],[3,0],
             [offset+1,1],[offset+2,1],[offset+3,1],[offset+4,1],
             [offset+2,2],[offset+3,2],[offset+4,2],[offset+5,2]]
        if _is_valid(c):
            configs.append(c)
    
    # Configuration 2: 4+4+4 on three lines meeting at a point
    for m in range(-2, 3):
        c = [[1,0],[2,0],[3,0],[4,0],
             [0,1],[0,2],[0,3],[0,4],
             [1,m],[2,2*m],[3,3*m],[4,4*m]]
        # Remove duplicates
        seen = set()
        unique = []
        for p in c:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        if len(unique) == 12 and _is_valid(unique):
            configs.append(unique)
    
    # Configuration 3: Grid-like 4x3
    for dx in range(1, 4):
        for dy in range(1, 4):
            c = []
            for i in range(4):
                for j in range(3):
                    c.append([i*dx, j*dy])
            if _is_valid(c):
                configs.append(c)
    
    # Configuration 4: Two parallel lines with 6 points each
    for spacing in range(1, 5):
        c = [[i*spacing, 0] for i in range(6)] + [[i*spacing, 1] for i in range(6)]
        if _is_valid(c):
            configs.append(c)
    
    # Configuration 5: 3+3+3+3 on four lines
    c5 = [[0,0],[1,0],[2,0],
          [0,1],[1,1],[2,1],
          [0,2],[1,2],[2,2],
          [3,0],[3,1],[3,2]]
    if _is_valid(c5):
        configs.append(c5)
    
    # Configuration 6: Hesse-like configuration (9 inflection points + 3 more)
    # The Hesse configuration has 9 points with 12 lines, each line has 3 points
    # For 12 points, we can try to extend this
    c6 = [[0,0],[1,0],[2,0],[3,0],
          [0,1],[1,1],[2,1],[3,1],
          [0,2],[1,2],[2,2],[3,2]]
    if _is_valid(c6):
        configs.append(c6)
    
    # Configuration 7: Points on a conic-like structure
    for a in range(1, 4):
        for b in range(1, 4):
            c = []
            for t in range(12):
                x = t
                y = (a * t * t + b * t) % 13
                c.append([x, y])
            if len(set(tuple(p) for p in c)) == 12 and _is_valid(c):
                configs.append(c)
    
    # Configuration 8: Random structured with specific patterns
    random.seed(123)
    for _ in range(20):
        # 4 points on y=0
        x1 = random.sample(range(-5, 5), 4)
        # 4 points on y=x
        x2 = random.sample(range(-5, 5), 4)
        # 4 points on y=-x
        x3 = random.sample(range(-5, 5), 4)
        
        c = [[x, 0] for x in x1] + [[x, x] for x in x2] + [[x, -x] for x in x3]
        seen = set()
        unique = []
        for p in c:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        if len(unique) == 12 and _is_valid(unique):
            configs.append(unique)
    
    return configs

def _local_search(points, deadline):
    """Try single-point replacement moves to reduce ordinary line count."""
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Generate candidate pool - larger and more structured
    coords = set()
    for p in current:
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                coords.add((p[0] + dx, p[1] + dy))
    
    # Add structured candidates in a larger range
    for x in range(-10, 20):
        for y in range(-10, 20):
            coords.add((x, y))
    
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    max_no_improve = 30
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = current[i]
            for cand in candidate_list:
                if tuple(cand) == tuple(old_pt):
                    continue
                if any(tuple(cand) == tuple(current[j]) for j in range(12) if j != i):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary_fast(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    break
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            no_improve += 1
    
    return best, best_score

def _two_point_swap_search(points, deadline):
    """Try swapping two points to reduce ordinary line count."""
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Generate candidate pool
    coords = set()
    for p in current:
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                coords.add((p[0] + dx, p[1] + dy))
    for x in range(-8, 16):
        for y in range(-8, 16):
            coords.add((x, y))
    
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    max_no_improve = 20
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            for j in range(i + 1, 12):
                if time.time() >= deadline:
                    break
                for cand_i in candidate_list[:50]:
                    if tuple(cand_i) == tuple(current[i]):
                        continue
                    if any(tuple(cand_i) == tuple(current[k]) for k in range(12) if k != i):
                        continue
                    for cand_j in candidate_list[:50]:
                        if tuple(cand_j) == tuple(current[j]):
                            continue
                        if tuple(cand_j) == tuple(cand_i):
                            continue
                        if any(tuple(cand_j) == tuple(current[k]) for k in range(12) if k != j):
                            continue
                        
                        new_pts = [list(p) for p in current]
                        new_pts[i] = list(cand_i)
                        new_pts[j] = list(cand_j)
                        
                        if not _is_valid(new_pts):
                            continue
                        
                        score = _count_ordinary_fast(new_pts)
                        if score < current_score:
                            current = new_pts
                            current_score = score
                            improved = True
                            no_improve = 0
                            if current_score < best_score:
                                best_score = current_score
                                best = [list(p) for p in current]
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
    
    return best, best_score

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Start with best structured configuration
    configs = _generate_structured_candidates()
    best_points = None
    best_score = float('inf')
    
    for cfg in configs:
        if not _is_valid(cfg):
            continue
        s = _count_ordinary_fast(cfg)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in cfg]
    
    if best_points is None:
        # Fallback: simple grid
        best_points = [[i, j] for i in range(4) for j in range(3)]
        best_score = _count_ordinary_fast(best_points)
    
    # Local search from best configuration
    if time.time() < deadline - 15:
        ls_points, ls_score = _local_search(best_points, deadline - 5)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # Two-point swap search
    if best_score > 6 and time.time() < deadline - 15:
        sw_points, sw_score = _two_point_swap_search(best_points, deadline - 5)
        if sw_score < best_score:
            best_score = sw_score
            best_points = sw_points
    
    # More random restarts if needed
    if best_score > 6 and time.time() < deadline - 20:
        random.seed(42)
        for restart in range(30):
            if time.time() >= deadline - 10:
                break
            
            # Generate random structured configuration
            pts = []
            # Choose 3 lines with 4 points each
            m1 = random.randint(-3, 3)
            b1 = random.randint(-5, 5)
            for x in random.sample(range(-10, 10), 4):
                pts.append([x, m1 * x + b1])
            
            m2 = random.randint(-3, 3)
            b2 = random.randint(-5, 5)
            for x in random.sample(range(-10, 10), 4):
                pt = [x, m2 * x + b2]
                if all(tuple(pt) != tuple(p) for p in pts):
                    pts.append(pt)
            
            m3 = random.randint(-3, 3)
            b3 = random.randint(-5, 5)
            for x in random.sample(range(-10, 10), 4):
                pt = [x, m3 * x + b3]
                if all(tuple(pt) != tuple(p) for p in pts):
                    pts.append(pt)
            
            # Fill remaining with random points
            while len(pts) < 12:
                pt = [random.randint(-10, 10), random.randint(-10, 10)]
                if all(tuple(pt) != tuple(p) for p in pts):
                    pts.append(pt)
            
            if len(pts) != 12 or not _is_valid(pts):
                continue
            
            s = _count_ordinary_fast(pts)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
            
            # Local search
            if time.time() < deadline - 10:
                ls_points, ls_score = _local_search(pts, deadline - 5)
                if ls_score < best_score:
                    best_score = ls_score
                    best_points = ls_points
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END

