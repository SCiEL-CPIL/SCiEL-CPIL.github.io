import time
import random
from math import gcd

_DEADLINE_SECONDS = 140.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _count_collinear_triples(points):
    """Count total collinear triples in the set."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    count += 1
    return count


def _check_validity(points):
    """Check no 4 collinear and distinct points in bounds."""
    n = len(points)
    seen = set()
    for p in points:
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    
    # Check no 4 collinear
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                return False # Distinctness already checked, but just in case
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dxn, dyn = dx // g, dy // g
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn, dyn = -dxn, -dyn
            
            cnt = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if (points[k][0] - points[i][0]) * dyn == (points[k][1] - points[i][1]) * dxn:
                    cnt += 1
                    if cnt >= 2:
                        return False
    return True


def _max_no_three_collinear(points, timeout=None):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    
    # Precompute collinear pairs
    # collinear_pairs[(i, j)] is a set of k such that i, j, k are collinear
    collinear_pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            collinear_pairs[(i, j)] = set()
            for k in range(n):
                if k != i and k != j and _collinear(points[i], points[j], points[k]):
                    collinear_pairs[(i, j)].add(k)
    
    best = [0]
    chosen = [0] * n
    deadline = time.time() + (timeout if timeout else 5.0)
    
    # Order points by number of collinear triples they are in (heuristic)
    degrees = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            deg = len(collinear_pairs.get((i, j), set()))
            degrees[i] += deg
            degrees[j] += deg
    
    order = sorted(range(n), key=lambda x: -degrees[x])
    # We will branch in this order
    # But the B&B logic below uses indices 0..n-1. 
    # It's easier to just use the original indices and let the heuristic be in the 'include' choice.
    
    def can_add(i):
        # i must not be collinear with any pair already chosen
        for a in range(n):
            if not chosen[a]:
                continue
            for b in range(a + 1, n):
                if chosen[b] and i in collinear_pairs.get((a, b), set()):
                    return False
        return True

    def branch(start, size):
        if time.time() > deadline:
            return
        if size > best[0]:
            best[0] = size
        if start >= n:
            return
        # Upper bound
        if size + (n - start) <= best[0]:
            return
        
        # Heuristic: Try to include the point with highest degree first?
        # The current loop goes 0 to n-1.
        # To improve, we could sort the points. But 'chosen' is index-based.
        # Let's stick to 0..n-1 but maybe swap? No, too complex.
        
        # Try include
        if can_add(start):
            chosen[start] = 1
            branch(start + 1, size + 1)
            chosen[start] = 0
        # Try exclude
        branch(start + 1, size)
        
    branch(0, 0)
    return best[0]


def _construct_initial(rng):
    """Generate a valid initial configuration."""
    # Try to create a grid-like structure which has many collinear triples
    # 4x5 grid is invalid (has 4 collinear).
    # Perturb a grid.
    pts = []
    used = set()
    # Base grid
    for i in range(4):
        for j in range(5):
            x = i * 10
            y = j * 10
            # Perturb
            found = False
            for _ in range(50):
                px = x + rng.randint(-5, 5)
                py = y + rng.randint(-5, 5)
                if -100 <= px <= 100 and -100 <= py <= 100 and (px, py) not in used:
                    pts.append([px, py])
                    if _check_validity(pts):
                        used.add((px, py))
                        found = True
                        break
                    else:
                        pts.pop()
            if not found:
                # Try random
                for _ in range(50):
                    px = rng.randint(-100, 100)
                    py = rng.randint(-100, 100)
                    if (px, py) not in used:
                        pts.append([px, py])
                        if _check_validity(pts):
                            used.add((px, py))
                            found = True
                            break
                        else:
                            pts.pop()
                if not found:
                    # Give up, return None
                    return None
    return pts


def _local_search(points, rng, max_iter=5000):
    """Local search maximizing collinear triples."""
    pts = [p[:] for p in points]
    current_triples = _count_collinear_triples(pts)
    best_triples = current_triples
    best_pts = [p[:] for p in pts]
    best_exact_score = _max_no_three_collinear(pts, timeout=1.0) # Initial exact check
    
    for _ in range(max_iter):
        if _remaining() < 20.0:
            break
        
        i = rng.randint(0, 19)
        orig_i = pts[i][:]
        
        # Try moving point i
        for _ in range(10):
            new_x = rng.randint(-100, 100)
            new_y = rng.randint(-100, 100)
            if (new_x, new_y) in [tuple(p) for p in pts if p != orig_i]:
                continue
            
            pts[i] = [new_x, new_y]
            
            if not _check_validity(pts):
                continue
            
            new_triples = _count_collinear_triples(pts)
            if new_triples > current_triples:
                current_triples = new_triples
                best_triples = new_triples
                best_pts = [p[:] for p in pts]
                # Update exact score occasionally or if improvement is significant
                # To save time, only update exact if we have time
                if _remaining() > 30.0:
                    s = _max_no_three_collinear(pts, timeout=1.0)
                    if s < best_exact_score:
                        best_exact_score = s
                break
            elif new_triples == current_triples and rng.random() < 0.05:
                # Sideways move
                current_triples = new_triples
                best_triples = new_triples
                best_pts = [p[:] for p in pts]
                break
        
        if _remaining() < 20.0:
            break
            
    return best_pts, best_exact_score


def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    
    rng = random.Random(42)
    
    # Try multiple initial configurations
    for seed in range(10):
        if _remaining() < 30.0:
            break
        rng = random.Random(seed * 100 + 42)
        pts = _construct_initial(rng)
        if pts is None:
            continue
        
        s = _max_no_three_collinear(pts, timeout=2.0)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            
        # Local search
        if _remaining() > 20.0:
            improved_pts, improved_score = _local_search(pts, rng, max_iter=1000)
            if improved_score < best_score:
                best_score = improved_score
                best_points = improved_pts
    
    # Fallback
    if best_points is None:
        # Simple parabola
        best_points = [[x, x * x] for x in range(-10, 10)]
        # Ensure validity (parabola has no 3 collinear? No, y=x^2 has no 3 integer collinear? 
        # Actually, 3 points on y=x^2 are never collinear. So score is 20.
        # But we need no 4 collinear. Parabola is fine.
        best_score = 20

    return best_points
