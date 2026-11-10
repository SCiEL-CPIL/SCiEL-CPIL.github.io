import time
import random
from math import gcd
from functools import lru_cache

_START_TIME = None
_DEADLINE = 145.0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return 0.0
    return time.time() - _START_TIME

def _check_no_four_collinear(points):
    """Returns True if no 4 points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _build_conflict_masks(points):
    """
    For each point i, compute a bitmask of points j such that there exists
    a point k where {i, j, k} are collinear. This is the "conflict" set.
    Also returns list of all collinear triples as frozensets.
    """
    n = len(points)
    conflict = [0] * n
    triples = []
    seen_triples = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            line_pts = [i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    line_pts.append(k)
            m = len(line_pts)
            if m >= 3:
                for a in range(m):
                    for b in range(a + 1, m):
                        for c in range(b + 1, m):
                            triple = frozenset([line_pts[a], line_pts[b], line_pts[c]])
                            if triple not in seen_triples:
                                seen_triples.add(triple)
                                triples.append(triple)
                                for p in triple:
                                    for q in triple:
                                        if p != q:
                                            conflict[p] |= (1 << q)
    return conflict, triples

def _max_subset_no_three_collinear(points):
    """
    Exact branch and bound using bitmasks to find largest subset with no three collinear.
    """
    n = len(points)
    conflict, triples = _build_conflict_masks(points)
    
    if not triples:
        return n
    
    # Sort by degree for better pruning
    degrees = [bin(conflict[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Map from original index to position in sorted order
    pos_of = [0] * n
    for pos, orig in enumerate(order):
        pos_of[orig] = pos
    
    # Rebuild conflict in sorted order
    sorted_conflict = [0] * n
    for i in range(n):
        for j in range(n):
            if conflict[i] & (1 << j):
                sorted_conflict[pos_of[i]] |= (1 << pos_of[j])
    
    best = [0]
    deadline = _DEADLINE - 3
    
    def backtrack(pos, count, selected_mask):
        if _elapsed() > deadline:
            return
        if pos == n:
            if count > best[0]:
                best[0] = count
            return
        
        # Upper bound: can't add more than n - pos points
        if count + (n - pos) <= best[0]:
            return
        
        # Try including point at position pos
        idx = pos
        can_include = True
        # Check if adding idx creates a triple with any two already selected
        # For each selected point s, check if there's another selected point t
        # such that {idx, s, t} is a triple, i.e., s and t are both in conflict[idx]
        # Actually: idx conflicts with s means there exists some k where {idx,s,k} collinear.
        # We need to check: for each pair of selected points (s, t), is there a triple containing idx, s, t?
        # Better: for each selected point s, the set of points that form a triple with idx and s
        # is (conflict[idx] & (1 << s)) ... no.
        # conflict[idx] has bit s set if there exists k such that {idx, s, k} is a triple.
        # So if s is selected AND s is in conflict[idx], we need to check if the "witness" k is also selected.
        # This is tricky. Let me use a different approach.
        
        # Alternative: precompute for each pair (i,j) whether they are in a triple with some k.
        # Actually, let's just check: for each selected point s, if s is in conflict[idx],
        # then there exists some k with {idx, s, k} collinear. We need k to also be selected.
        # So: for each s in selected where s is in conflict[idx], check if any k in (conflict[idx] & conflict[s] & selected) 
        # Actually no. Let me think differently.
        
        # A triple {a, b, c} exists. If we select a and b, we can't select c.
        # So: idx can be added iff for no pair (s, t) in selected does {idx, s, t} form a triple.
        # {idx, s, t} forms a triple iff s is in conflict[idx] AND t is in conflict[idx] AND s is in conflict[t]
        # (and the triple actually exists, not just pairwise conflicts).
        
        # Simpler: just iterate over selected points and check.
        # For each selected s: if s in conflict[idx], find all k such that {idx, s, k} is a triple.
        # That's (conflict[idx] & conflict[s]) minus idx and s... but that's not quite right either.
        
        # Let me just do it directly: for each selected s, for each selected t (s < t in original),
        # check if {idx, s, t} is collinear. But that's O(n^2) per step which is slow.
        
        # Better approach: precompute triple_membership. For each point i, list of triples containing i.
        # Then when considering adding idx, check each triple containing idx: if the other two are both selected, can't add.
        
        pass
    
    # Let me use a cleaner approach with triple lists
    triples_with_point = [[] for _ in range(n)]
    for t_idx, t in enumerate(triples):
        for p in t:
            triples_with_point[pos_of[p]].append(t_idx)
    
    triples_as_list = [set(t) for t in triples]
    # Convert to sorted-order sets
    triples_sorted = [set(pos_of[p] for p in t) for t in triples]
    triples_with_point = [[] for _ in range(n)]
    for t_idx, t in enumerate(triples_sorted):
        for p in t:
            triples_with_point[p].append(t_idx)
    
    selected = [False] * n
    
    def backtrack2(pos, count):
        if _elapsed() > deadline:
            return
        if pos == n:
            if count > best[0]:
                best[0] = count
            return
        
        if count + (n - pos) <= best[0]:
            return
        
        idx = pos
        
        # Check if we can include idx
        can_include = True
        for t_idx in triples_with_point[idx]:
            t = triples_sorted[t_idx]
            others = [p for p in t if p != idx]
            if len(others) == 2:
                if selected[others[0]] and selected[others[1]]:
                    can_include = False
                    break
        
        if can_include:
            selected[idx] = True
            backtrack2(pos + 1, count + 1)
            selected[idx] = False
        
        backtrack2(pos + 1, count)
    
    backtrack2(0, 0)
    return best[0]

def _parabolic_config():
    """
    Generate a parabolic configuration: points on y = x^2.
    For integer x values, y = x^2 gives collinear triples when x values satisfy certain conditions.
    Specifically, three points (a, a^2), (b, b^2), (c, c^2) are collinear iff a+b+c = 0 (for distinct a,b,c).
    Wait, let me verify: slope from (a,a^2) to (b,b^2) is (b^2-a^2)/(b-a) = a+b.
    Slope from (b,b^2) to (c,c^2) is b+c.
    For collinearity: a+b = b+c, so a=c. That's not right for three distinct points.
    
    Actually, three points (a,a^2), (b,b^2), (c,c^2) are collinear iff the determinant is zero:
    |a  a^2  1|
    |b  b^2  1| = 0
    |c  c^2  1|
    = a(b^2-c^2) - a^2(b-c) + (bc^2 - b^2c) = (b-c)[a(b+c) - a^2 + bc] = (b-c)[ab+ac-a^2+bc]
    = (b-c)[a(b+c-a) + bc]
    For this to be zero with distinct a,b,c: a(b+c-a) + bc = 0
    => ab + ac - a^2 + bc = 0
    => a^2 = ab + ac + bc
    
    This is a known relation. For example, a=1: 1 = b + c + bc, so (b+1)(c+1) = 2, giving b=0,c=1 (not distinct from a=1) or b=-2,c=-1.
    So (1,1), (0,0), (-2,4) are collinear? Check: slope (0,0) to (1,1) is 1. Slope (1,1) to (-2,4) is (4-1)/(-2-1) = -1. Not collinear.
    
    Let me just use a different approach: use points on a parabola and rely on the natural structure.
    Actually, for the purpose of this problem, we want MANY collinear triples but NO four collinear.
    A parabola y=x^2 has the property that no three points are collinear unless they satisfy the special condition above.
    So a pure parabola has FEW triples, which is bad for us.
    
    Instead, let's use a configuration based on a projective plane or Fano plane extension.
    """
    # Use a structured grid with perturbations to avoid 4-collinear
    # Start with a 5x5 grid and pick 20 points
    points = []
    used = set()
    # Use points from a 6x6 grid
    for x in range(-5, 5):
        for y in range(-5, 5):
            if len(points) >= 20:
                break
            key = (x, y)
            if key not in used:
                used.add(key)
                points.append([x, y])
        if len(points) >= 20:
            break
    
    # Perturb to remove 4-collinear
    for _ in range(100):
        if _check_no_four_collinear(points):
            break
        # Find a 4-collinear set and move one point
        found = False
        for i in range(20):
            for j in range(i + 1, 20):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and dy == 0:
                    continue
                line_pts = []
                for k in range(20):
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        line_pts.append(k)
                if len(line_pts) >= 4:
                    idx = random.choice(line_pts)
                    old_x, old_y = points[idx]
                    for _ in range(20):
                        new_x = old_x + random.randint(-3, 3)
                        new_y = old_y + random.randint(-3, 3)
                        if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                            if (new_x, new_y) not in [(p[0], p[1]) for k, p in enumerate(points) if k != idx]:
                                points[idx] = [new_x, new_y]
                                found = True
                                break
                    if found:
                        break
            if found:
                break
    
    return points

def _random_valid_config():
    """Generate a random valid configuration."""
    while True:
        if _elapsed() > _DEADLINE - 20:
            return None
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-15, 15)
            y = random.randint(-15, 15)
            key = (x, y)
            if key not in seen:
                seen.add(key)
                points.append([x, y])
        if _check_no_four_collinear(points):
            return points

def _count_triples(points):
    """Count collinear triples."""
    n = len(points)
    count = 0
    seen_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Normalize line
            if dx == 0:
                a, b, c = 1, 0, -x1
            elif dy == 0:
                a, b, c = 0, 1, -y1
            else:
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                a, b = dy, -dx
                c = -(a * x1 + b * y1)
                g2 = gcd(gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
            line_key = (a, b, c)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            line_pts = []
            for k in range(n):
                x3, y3 = points[k]
                if a * x3 + b * y3 + c == 0:
                    line_pts.append(k)
            m = len(line_pts)
            if m >= 3:
                count += m * (m - 1) * (m - 2) // 6
    return count

def _improve_triples(points, iterations=200):
    """Try to increase the number of triples while maintaining validity."""
    current_triples = _count_triples(points)
    for _ in range(iterations):
        if _elapsed() > _DEADLINE - 15:
            break
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        
        improved = False
        for _ in range(5):
            new_x = random.randint(-15, 15)
            new_y = random.randint(-15, 15)
            if new_x == old_x and new_y == old_y:
                continue
            if (new_x, new_y) in [(p[0], p[1]) for i, p in enumerate(points) if i != idx]:
                continue
            
            new_points = [list(p) for p in points]
            new_points[idx] = [new_x, new_y]
            
            if not _check_no_four_collinear(new_points):
                continue
            
            new_triples = _count_triples(new_points)
            if new_triples > current_triples:
                points = new_points
                current_triples = new_triples
                improved = True
                break
    
    return points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 21
    
    candidates = []
    
    # Generate initial candidates
    for _ in range(3):
        if _elapsed() > _DEADLINE - 30:
            break
        config = _random_valid_config()
        if config:
            candidates.append(config)
    
    # Parabolic/grid config
    if _elapsed() < _DEADLINE - 30:
        config = _parabolic_config()
        if _check_no_four_collinear(config):
            candidates.append(config)
    
    # Evaluate and improve
    for config in candidates:
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Try to improve
        improved = _improve_triples(config, iterations=100)
        
        if _elapsed() < _DEADLINE - 5:
            s = _max_subset_no_three_collinear(improved)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in improved]
    
    # If we have a best, try more improvements
    if best_points and _elapsed() < _DEADLINE - 10:
        for _ in range(3):
            if _elapsed() > _DEADLINE - 5:
                break
            improved = _improve_triples(best_points, iterations=50)
            s = _max_subset_no_three_collinear(improved)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in improved]
    
    if best_points is None:
        # Fallback: return any valid config
        config = _random_valid_config()
        if config is None:
            # Last resort: simple grid
            config = [[x, y] for x in range(5) for y in range(4)]
        best_points = config
    
    return best_points
