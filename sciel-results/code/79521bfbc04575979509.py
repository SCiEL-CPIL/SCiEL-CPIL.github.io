"""Rewritten program: local search maximizing 3-collinear triples under no-4-collinear."""
import random
import time
from itertools import combinations

DEADLINE = 140.0  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical representation of the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 1, x1)
    if dy == 0:
        return (1, 0, y1)
    # Normalize so that gcd(dx, dy) = 1 and dx > 0, or dx == 0 and dy > 0
    import math
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # Line: dx*(y - y1) = dy*(x - x1) => dy*x - dx*y = dy*x1 - dx*y1
    a = dy
    b = -dx
    c = dy * x1 - dx * y1
    # Normalize sign: make (a, b) have a fixed sign convention
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _count_collinear_triples(points):
    """Count number of 3-point collinear subsets."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_map.setdefault(key, set()).add((i, j))
    
    triple_count = 0
    for key, pairs in line_map.items():
        # Find all points on this line
        pts_on_line = set()
        for (i, j) in pairs:
            pts_on_line.add(i)
            pts_on_line.add(j)
        k = len(pts_on_line)
        if k >= 3:
            triple_count += k * (k - 1) * (k - 2) // 6
    return triple_count

def _has_four_collinear(points):
    """Check if any 4 points are collinear."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, pts in line_map.items():
        if len(pts) >= 4:
            return True
    return False

def _largest_no_three_collinear(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    # Precompute which triples are collinear
    collinear = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Check collinearity
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    collinear[i][j] = True
                    collinear[i][k] = True
                    collinear[j][k] = True
    
    # For each pair, find all points collinear with them
    line_members = {}
    for i in range(n):
        for j in range(i + 1, n):
            if collinear[i][j]:
                key = (i, j)
                if key not in line_members:
                    line_members[key] = set()
                line_members[key].add(i)
                line_members[key].add(j)
                for k in range(n):
                    if k != i and k != j and collinear[i][k] and collinear[j][k]:
                        line_members[key].add(k)
    
    # Branch and bound
    best = [0]
    
    def bound(remaining, current_size):
        # Upper bound: current_size + len(remaining)
        return current_size + len(remaining)
    
    def search(remaining, current_size, selected):
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Prune
        if bound(remaining, current_size) <= best[0]:
            return
        
        # Try adding the first remaining point
        first = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include first
        # Check if first conflicts with any pair in selected
        valid = True
        for (i, j) in combinations(selected, 2):
            if collinear[i][j] and (collinear[i][first] or collinear[j][first]):
                # Actually we need to check if adding first creates a collinear triple
                # with two already selected points
                pass
        # More careful: adding point p is valid if no two points in selected are collinear with p
        # i.e., for all pairs (a, b) in selected, not (collinear[a][p] and collinear[b][p])
        # But actually we need: no three collinear, so for any two a, b in selected, 
        # we cannot have a, b, p all collinear.
        # So: for all a, b in selected with a != b, if collinear[a][b] then p must not be on that line.
        # But collinear[a][b] means a, b, and some third... no, collinear[a][b] just means the line through a,b.
        # We need: for all a, b in selected, a != b: NOT (p is on line ab)
        # Which is: NOT (collinear[a][p] and collinear[b][p]) for all a, b in selected.
        
        can_add = True
        for a in selected:
            for b in selected:
                if a != b and collinear[a][p] and collinear[b][p]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            search(rest, current_size + 1, selected + [first])
        
        # Option 2: exclude first
        search(rest, current_size, selected)
    
    search(list(range(n)), 0, [])
    return best[0]

def construct():
    start_time = time.time()
    
    # Start with a random configuration
    def random_config():
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        return pts
    
    best_config = None
    best_triples = -1
    best_score = 20  # worst case
    
    # Initial: try a few random configs and a structured one
    configs = []
    
    # Structured: points on a circle-like pattern
    for r in range(1, 10):
        for angle_mult in range(1, 5):
            pass  # skip complex generation
    
    # Just start with random
    for _ in range(5):
        config = random_config()
        if not _has_four_collinear(config):
            triples = _count_collinear_triples(config)
            configs.append((triples, config))
    
    if not configs:
        # Fallback: parabola (no three collinear, score = 20)
        return [[x, x * x] for x in range(-10, 10)]
    
    configs.sort(reverse=True)
    best_config = configs[0][1]
    best_triples = configs[0][0]
    
    # Local search: move one point at a time to maximize triples while maintaining no-4-collinear
    for iteration in range(5000):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        config = list(best_config)
        improved = False
        
        for idx in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            old_pt = config[idx]
            
            # Try moving this point to a random new location
            for attempt in range(20):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                new_pt = (random.randint(-100, 100), random.randint(-100, 100))
                if new_pt == old_pt:
                    continue
                
                config[idx] = new_pt
                if new_pt in config[:idx] or new_pt in config[idx+1:]:
                    config[idx] = old_pt
                    continue
                
                if _has_four_collinear(config):
                    config[idx] = old_pt
                    continue
                
                triples = _count_collinear_triples(config)
                if triples > best_triples:
                    best_triples = triples
                    best_config = list(config)
                    improved = True
                    break
                else:
                    config[idx] = old_pt
            
            if improved:
                break
    
    # Final evaluation: compute the actual score (largest no-three-collinear subset)
    # For the best config found, run the exact branch and bound
    if time.time() - start_time < DEADLINE - 10:
        score = _largest_no_three_collinear(best_config)
    else:
        score = 20  # fallback
    
    return [[x, y] for x, y in best_config]
