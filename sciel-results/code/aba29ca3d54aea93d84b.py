import time
import random
from itertools import combinations

_DEADLINE = 110.0

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _valid_no_four(points):
    n = len(points)
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                return False
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
            if slopes[(dx, dy)] >= 3:
                return False
    return True

def _build_conflict_graph(points):
    """Build conflict graph: edge between i,j if they share a collinear triple.
    Returns list of bitmasks where conflict_mask[i] has bit j set if i and j conflict."""
    n = len(points)
    # First find all collinear triples
    triple_pairs = [set() for _ in range(n)]
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes.setdefault((dx, dy), []).append(j)
        for k in slopes.values():
            if len(k) >= 2:
                for a_idx in range(len(k)):
                    for b_idx in range(a_idx + 1, len(k)):
                        a, b = k[a_idx], k[b_idx]
                        triple_pairs[i].add((a, b))
                        triple_pairs[a].add((i, b))
                        triple_pairs[b].add((i, a))
    
    # Build conflict masks
    conflict_mask = [0] * n
    for i in range(n):
        for a, b in triple_pairs[i]:
            if a < i:
                conflict_mask[i] |= (1 << a)
                conflict_mask[a] |= (1 << i)
            elif a > i:
                # a is already > i, so this pair (i,a) is recorded when we process a's triples
                pass
        # Also need to handle pairs where i is the middle element
        # Actually, let me rebuild more carefully
    
    # Simpler: for each triple (i,a,b), add edges i-a, i-b, a-b
    conflict_mask = [0] * n
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes.setdefault((dx, dy), []).append(j)
        for k in slopes.values():
            if len(k) >= 2:
                for a_idx in range(len(k)):
                    for b_idx in range(a_idx + 1, len(k)):
                        a, b = k[a_idx], k[b_idx]
                        # Triple (i, a, b): add edges i-a, i-b, a-b
                        conflict_mask[i] |= (1 << a) | (1 << b)
                        conflict_mask[a] |= (1 << i) | (1 << b)
                        conflict_mask[b] |= (1 << i) | (1 << a)
    
    return conflict_mask

def _max_independent_set(conflict_mask, n, deadline_start):
    """Find maximum independent set in the conflict graph using branch and bound.
    An independent set in the conflict graph = set with no three collinear."""
    best = [0]
    all_mask = (1 << n) - 1
    
    # Degree of each vertex in conflict graph
    degrees = [bin(conflict_mask[i]).count('1') for i in range(n)]
    
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # For each vertex in order, precompute its conflict mask restricted to later vertices in order
    # We'll process vertices in order, and at each step decide include/exclude
    
    # Use a different approach: recursive with available mask
    # available_mask: vertices that can still be added
    # When we add vertex v, we remove v and all its neighbors from available
    
    def dfs(available, count):
        if time.time() - deadline_start > 109.0:
            return
        if count + bin(available).count('1') <= best[0]:
            return
        if available == 0:
            if count > best[0]:
                best[0] = count
            return
        
        # Pick a vertex: use heuristic - pick vertex with max degree in available subgraph
        # For speed, just pick the first available vertex in our ordering
        # Find first available vertex
        v = None
        for idx in order:
            if available & (1 << idx):
                v = idx
                break
        if v is None:
            return
        
        # Option 1: exclude v
        dfs(available & ~(1 << v), count)
        
        # Option 2: include v, remove v and its neighbors
        new_available = available & ~(1 << v) & ~conflict_mask[v]
        dfs(new_available, count + 1)
    
    dfs(all_mask, 0)
    return best[0]

def _score(points):
    n = len(points)
    conflict_mask = _build_conflict_graph(points)
    return _max_independent_set(conflict_mask, n, time.time())

def _triple_count(points):
    """Count number of collinear triples (for heuristic)."""
    n = len(points)
    count = 0
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
        for k in slopes.values():
            if k >= 3:
                count += combinations(k, 2)
    return count

def construct():
    start = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = 21
    
    def consider(pts):
        nonlocal best_pts, best_score
        if not _valid_no_four(pts):
            return
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_pts = list(pts)
    
    # Strategy A: Perturbed 5x4 grid
    base = [(x, y) for x in range(0, 5) for y in range(0, 4)]
    for attempt in range(15000):
        if time.time() - start > 100.0:
            break
        pts = list(base)
        for _ in range(rng.randint(3, 8)):
            idx = rng.randrange(20)
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            x, y = pts[idx]
            nx, ny = x + dx, y + dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                pts[idx] = (nx, ny)
        consider(pts)
        if best_score <= 12:
            break
    
    # Strategy B: Simulated annealing local search
    if best_pts:
        current_pts = list(best_pts)
        current_score = best_score
        current_tc = _triple_count(current_pts)
        
        temp = 5.0
        for step in range(30000):
            if time.time() - start > 108.0:
                break
            temp *= 0.9995
            
            idx = rng.randrange(20)
            x, y = current_pts[idx]
            
            # Try multiple random moves
            best_move = None
            best_tc = current_tc
            for _ in range(8):
                nx = x + rng.randint(-3, 3)
                ny = y + rng.randint(-3, 3)
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            tc = _triple_count(current_pts)
                            delta_tc = tc - current_tc
                            if delta_tc < 0 or rng.random() < pow(2.718281828, -delta_tc / max(temp, 0.1)):
                                if tc < best_tc:
                                    best_tc = tc
                                    best_move = new_pt
                        current_pts[idx] = old_pt
            
            if best_move is not None:
                old_pt = current_pts[idx]
                current_pts[idx] = best_move
                current_tc = best_tc
                new_score = _score(current_pts)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_pts = list(current_pts)
                elif rng.random() < pow(2.718281828, -(new_score - current_score) / max(temp, 0.1)):
                    current_score = new_score
            else:
                pass
    
    # Strategy C: Random small box
    for attempt in range(20000):
        if time.time() - start > 109.0:
            break
        pts = []
        for _ in range(20):
            pts.append((rng.randint(-8, 8), rng.randint(-8, 8)))
        consider(pts)
        if best_score <= 12:
            break
    
    # Strategy D: Structured configurations
    # Try a few hand-crafted dense patterns
    configs = [
        # 5x4 grid with one point moved per row
        [(x, y) for x in range(5) for y in range(4) if not (x == 4 and y == 0)] + [(4, 10)],
        # Perturbed
        [(x, y + (x % 2)) for x in range(5) for y in range(4)],
        [(x, y + (x // 2)) for x in range(5) for y in range(4)],
    ]
    for cfg in configs:
        if time.time() - start > 109.5:
            break
        consider(cfg)
        if best_score <= 12:
            break
    
    if best_pts is None:
        return [[x, x * x] for x in range(-10, 10)]
    
    return [list(p) for p in best_pts]
