"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd
import sys

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: (x,y) such that (y - y1)*dx - (x - x1)*dy == 0
    # Canonical: use the line equation ax + by + c = 0 with gcd(a,b,1)=1 and a>0 or (a==0 and b>0)
    a = dy
    b = -dx
    c = -(a * x1 + b * y1)
    g2 = gcd(abs(a), abs(b))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    # Normalize sign
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _get_lines(points):
    """Return dict: line_key -> list of point indices on that line."""
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _has_four_collinear(points):
    lines = _get_lines(points)
    for pts in lines.values():
        if len(pts) >= 4:
            return True
    return False

def _count_triples(points):
    """Count number of collinear triples."""
    lines = _get_lines(points)
    count = 0
    for pts in lines.values():
        n = len(pts)
        if n >= 3:
            count += n * (n - 1) * (n - 2) // 6
    return count

def _max_3free_subset(points):
    """Find the largest subset of points with no three collinear using branch and bound."""
    n = len(points)
    # Build list of collinear triples
    lines = _get_lines(points)
    triples = []
    for pts in lines.values():
        if len(pts) >= 3:
            idx_list = sorted(pts)
            for combo in combinations(idx_list, 3):
                triples.append(combo)
    
    # For each point, which triples does it participate in?
    point_triples = [[] for _ in range(n)]
    for t in triples:
        for i in t:
            point_triples[i].append(t)
    
    # Branch and bound
    best = 0
    # Greedy initial upper bound: try to find a large 3-free set greedily
    # Use recursive B&B
    
    # Precompute: for each triple, a bitmask? n=20 so we can use integers
    # Represent selected set as a frozenset or bitmask
    # Use bitmask for speed
    
    # For B&B: state = (candidate_points_remaining, current_selected, current_size)
    # We select points one by one. If adding a point creates a triple with two already selected, skip.
    
    # Better: recursive with pruning
    # selected_mask: which points are in the current 3-free set
    # candidate_mask: which points can still be added
    # A point can be added if it doesn't form a triple with any two already selected
    
    # Precompute for each pair of points, whether there's a third point on the same line
    # pair_conflict[i][j] = set of k such that i,j,k are collinear (and k != i,j)
    pair_third = {}
    for t in triples:
        a, b, c = t
        pair_third.setdefault((a, b), set()).add(c)
        pair_third.setdefault((b, a), set()).add(c)
        pair_third.setdefault((a, c), set()).add(b)
        pair_third.setdefault((c, a), set()).add(b)
        pair_third.setdefault((b, c), set()).add(a)
        pair_third.setdefault((c, b), set()).add(a)
    
    def can_add(point, selected_mask):
        """Check if point can be added to selected set without creating a triple."""
        # For each pair in selected, check if point completes a triple
        for i in range(n):
            if selected_mask & (1 << i):
                key = (i, point)
                if key in pair_third:
                    for k in pair_third[key]:
                        if selected_mask & (1 << k):
                            return False
        return True
    
    # B&B with memoization-like pruning
    # Order points by "conflict degree" (number of triples they're in) to prune early
    
    # Use a recursive approach with bitmasks
    # State: (remaining_candidates_mask, selected_mask)
    # Upper bound: len(selected) + popcount(remaining_candidates)
    
    # For better pruning, use a greedy lower bound on the remaining
    # Actually, let's use a simpler but effective B&B:
    
    # Sort points by number of conflicts (descending) - process high-conflict points first
    conflict_count = [len(point_triples[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -conflict_count[i])
    
    # We'll do DFS in this order
    # selected: set of points chosen
    # For pruning: upper bound = current_size + number of remaining points
    
    # Additional pruning: if a point conflicts with two selected points, it can't be added
    # We can precompute for each point, given the selected set, whether it's available
    
    # Let's use a cleaner recursive B&B:
    
    best_size = [0]
    
    def dfs(idx, selected_mask, current_size):
        if _elapsed() > 115:  # safety
            return
        if idx == n:
            if current_size > best_size[0]:
                best_size[0] = current_size
            return
        
        point = order[idx]
        
        # Upper bound: current_size + (n - idx)
        if current_size + (n - idx) <= best_size[0]:
            return
        
        # Option 1: skip this point
        dfs(idx + 1, selected_mask, current_size)
        
        # Option 2: add this point if possible
        if can_add(point, selected_mask):
            dfs(idx + 1, selected_mask | (1 << point), current_size + 1)
    
    dfs(0, 0, 0)
    return best_size[0]

def _validate(points):
    """Check that points are valid: 20 distinct, |x|,|y|<=100, no four collinear."""
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if _has_four_collinear(points):
        return False
    return True

def _initial_grid():
    """Create a 4x5 grid with perturbations to avoid 4-collinear."""
    points = []
    # 4 rows, 5 columns
    for r in range(4):
        for c in range(5):
            x = c * 20 - 40  # -40, -20, 0, 20, 40
            y = r * 20 - 30  # -30, -10, 10, 30
            # Small perturbation to break perfect grid lines
            px = x + (r % 2) * 3 + (c % 3) * 2
            py = y + (c % 2) * 3 + (r % 3) * 2
            points.append([px, py])
    return points

def _improve(points, max_iters=5000):
    """Local search to reduce the 3-free subset size."""
    current = [list(p) for p in points]
    current_score = _max_3free_subset(current)
    current_triples = _count_triples(current)
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iters):
        if _elapsed() > 110:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = current[idx]
        
        # Try a few random new positions
        found = False
        for _ in range(30):
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            if abs(new_x) > 100 or abs(new_y) > 100:
                continue
            # Check distinctness
            valid = True
            for j in range(20):
                if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                    valid = False
                    break
            if not valid:
                continue
            
            current[idx] = [new_x, new_y]
            
            if _has_four_collinear(current):
                current[idx] = [old_x, old_y]
                continue
            
            # Evaluate
            new_triples = _count_triples(current)
            if new_triples > current_triples:
                # Better: more triples means harder to find 3-free set
                # But we need to check actual score
                # Only compute exact score occasionally for speed
                if random.random() < 0.3 or it % 100 == 0:
                    new_score = _max_3free_subset(current)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best = [list(p) for p in current]
                            found = True
                        break
                    else:
                        current[idx] = [old_x, old_y]
                else:
                    # Accept based on triples for speed
                    current_triples = new_triples
                    found = True
                    break
            else:
                current[idx] = [old_x, old_y]
        
        if not found:
            # Try larger moves
            for _ in range(10):
                new_x = old_x + random.randint(-15, 15)
                new_y = old_y + random.randint(-15, 15)
                if abs(new_x) > 100 or abs(new_y) > 100:
                    continue
                valid = True
                for j in range(20):
                    if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                        valid = False
                        break
                if not valid:
                    continue
                
                current[idx] = [new_x, new_y]
                
                if _has_four_collinear(current):
                    current[idx] = [old_x, old_y]
                    continue
                
                new_triples = _count_triples(current)
                if new_triples > current_triples:
                    if random.random() < 0.5:
                        new_score = _max_3free_subset(current)
                        if new_score < current_score:
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best = [list(p) for p in current]
                            break
                    else:
                        current_triples = new_triples
                        break
                else:
                    current[idx] = [old_x, old_y]
    
    return best, best_score

def construct():
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    # Start with perturbed grid
    points = _initial_grid()
    
    # Validate initial
    if not _validate(points):
        # Try different perturbations
        for attempt in range(20):
            points = []
            for r in range(4):
                for c in range(5):
                    x = c * 18 - 36
                    y = r * 18 - 27
                    px = x + random.randint(-3, 3)
                    py = y + random.randint(-3, 3)
                    points.append([px, py])
            if _validate(points):
                break
    
    if not _validate(points):
        # Fallback: parabola (valid but poor score)
        return [[x, x * x] for x in range(-10, 10)]
    
    # Local search
    points, score = _improve(points, max_iters=8000)
    
    # Final validation
    if not _validate(points):
        return [[x, x * x] for x in range(-10, 10)]
    
    # Verify score
    final_score = _max_3free_subset(points)
    
    return points
# EVOLVE-BLOCK-END
