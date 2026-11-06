import time
import random
from itertools import combinations

_DEADLINE = 110.0

def _elapsed(start):
    return time.time() - start

def _is_collinear(p, q, r):
    (x1, y1), (x2, y2), (x3, y3) = p, q, r
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

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

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _build_triples(points):
    n = len(points)
    triples = set()
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
            for a, b in combinations(k, 2):
                triples.add(frozenset((i, a, b)))
    return [tuple(t) for t in triples]

def _min_hitting_set(triples_list, n, deadline_start):
    if not triples_list:
        return 0
    
    m = len(triples_list)
    
    # Map vertex to bitmask of triples it hits
    # Since m can be up to C(20,3)=1140, we can't use simple int bitmasks for 1140 bits easily in Python without big ints, but Python supports arbitrary precision ints.
    # However, iterating over bits of a 1140-bit int is slow.
    # Alternative: List of lists.
    
    v_to_triples = [set() for _ in range(n)]
    for idx, t in enumerate(triples_list):
        for v in t:
            v_to_triples[v].add(idx)
    
    best = [n]
    
    # Precompute degree of each vertex (number of triples it hits)
    # This is static for the whole problem instance.
    
    # DFS
    def dfs(excluded_count, active_triples):
        if time.time() - deadline_start > 109.0:
            return
        if not active_triples:
            if excluded_count < best[0]:
                best[0] = excluded_count
            return
        
        if excluded_count >= best[0]:
            return
            
        # Lower bound:
        # Max triples any single vertex can cover from active_triples
        max_cover = 0
        for v in range(n):
            if v_to_triples[v]: # Optimization: skip if empty
                count = 0
                for t_idx in v_to_triples[v]:
                    if t_idx in active_triples:
                        count += 1
                if count > max_cover:
                    max_cover = count
                if max_cover == 0:
                    break # No active triples? Should have returned earlier.
        
        if max_cover == 0:
             # This shouldn't happen if active_triples is not empty
             return
             
        lb = (len(active_triples) + max_cover - 1) // max_cover
        if excluded_count + lb >= best[0]:
            return

        # Heuristic: Pick a triple. 
        # Which one? One that contains a vertex with high "activity".
        # Let's pick a random active triple to diversify, or the one with smallest "options"?
        # Actually, standard B&B for Hitting Set: pick an element (triple) and branch on its attributes (vertices).
        
        # To speed up, pick the triple that has the vertex with the highest degree in the *induced* subgraph?
        # Or just pick the first one.
        
        # Let's pick the triple containing the vertex with the highest global degree to reduce branching factor?
        # No, we must pick a triple.
        
        # Let's pick a triple that is "most constrained".
        # For each active triple, count how many of its vertices are "popular" (high degree).
        # This is expensive.
        
        # Simple heuristic: Pick the first active triple.
        t_idx = next(iter(active_triples))
        t = triples_list[t_idx]
        
        # Order vertices in t by their degree in the active hypergraph (descending)
        # This helps find a good solution (small hitting set) faster, improving pruning.
        
        # Calculate active degree for vertices in t
        v_scores = []
        for v in t:
            deg = 0
            for t_idx2 in v_to_triples[v]:
                if t_idx2 in active_triples:
                    deg += 1
            v_scores.append((deg, v))
            
        v_scores.sort(reverse=True)
        
        for _, v in v_scores:
            new_excluded_count = excluded_count + 1
            if new_excluded_count >= best[0]:
                continue
            
            # Remove all triples containing v from active
            new_active = active_triples - v_to_triples[v]
            dfs(new_excluded_count, new_active)

    initial_active = set(range(m))
    dfs(0, initial_active)
    return best[0]

def _score(points):
    triples = _build_triples(points)
    n = len(points)
    hs_size = _min_hitting_set(triples, n, time.time())
    return n - hs_size

def construct():
    start = time.time()
    rng = random.Random(12345)

    best_pts = None
    best_score = 21

    def consider(pts):
        nonlocal best_pts, best_score
        if not _valid_no_four(pts):
            return
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts

    # Strategy A: Perturbed 5x4 grid
    base = [(x, y) for x in range(0, 5) for y in range(0, 4)]
    for attempt in range(20000):
        if time.time() - start > 105.0:
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

    # Strategy B: Targeted Local Search
    # Identify collinear triples in best_pts and move points to break them.
    if best_pts:
        current_pts = list(best_pts)
        current_score = best_score
        
        # Pre-calculate triples for current best
        # We will re-evaluate score after each move
        
        for step in range(2000):
            if time.time() - start > 108.0:
                break
            
            # Get triples in current configuration
            triples = _build_triples(current_pts)
            if not triples:
                break # Score is 20, can't do better? No, we want to MINIMIZE score.
                # Wait. Score = n - min_hitting_set.
                # If no triples, min_hitting_set = 0, score = 20.
                # We want to MINIMIZE score. So we want MORE triples?
                # "The score is the largest subset with no three collinear".
                # If no 3 collinear, the largest subset is 20.
                # If many 3 collinear, the largest subset is smaller.
                # So we WANT to create more collinearities to reduce the score.
                # The constraint is "no four collinear".
                
                # So my goal is to MAXIMIZE the number of collinear triples (while keeping no 4 collinear)
                # to MINIMIZE the largest 3-collinear-free subset.
                
                # My previous logic "break collinearities" was WRONG.
                # I want to ADD collinearities.
                
                # Let's re-read: "Minimize. A valid configuration with score s certifies g(20) <= s".
                # Score = largest subset with no 3 collinear.
                # If I have many 3-collinear points, I have to remove more points to make the set 3-collinear-free.
                # So yes, I want to MAXIMIZE collinearities.
                
                # The seed program (parabola) has 0 collinear triples -> Score 20.
                # Grid has many -> Score lower.
                
                # So Strategy B should try to CREATE collinearities.
                
                # How? Move points to align with existing pairs.
                
                # For each pair of points, find a 3rd point that can be moved to align with them.
                # Or move one of the pair to align with a 3rd.
                
                # Heuristic:
                # 1. Pick a random pair (i, j).
                # 2. Pick a random point k (not i, j).
                # 3. Try to move k to a position collinear with i and j.
                # 4. Check validity (no 4 collinear, bounds, distinct).
                # 5. Evaluate new score.
                
                pass # Skip old logic
                
        # New Strategy B: Collinearity Creation
        if best_pts:
            current_pts = list(best_pts)
            current_score = best_score
            
            for step in range(3000):
                if time.time() - start > 108.5:
                    break
                
                # Pick a random point to move
                idx = rng.randrange(20)
                x, y = current_pts[idx]
                
                # Pick a random pair to align with
                i, j = rng.sample(range(20), 2)
                if i == idx or j == idx:
                    continue
                
                p1 = current_pts[i]
                p2 = current_pts[j]
                
                # Direction vector
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # Try moving idx to a point on the line defined by p1, p2
                # Line: (x, y) = p1 + t * (dx, dy)
                # We need integer coordinates.
                # So dx, dy should be small or we pick t such that coords are integer.
                # Since p1, p2 are integers, any integer t gives integer coords.
                
                # Try a few t values
                found_improvement = False
                for t in range(-10, 11):
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        new_pt = (nx, ny)
                        if new_pt == current_pts[idx]:
                            continue
                        if new_pt in current_pts:
                            continue
                        
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        
                        if _valid_no_four(current_pts):
                            new_score = _score(current_pts)
                            if new_score < current_score:
                                current_score = new_score
                                if current_score < best_score:
                                    best_score = current_score
                                    best_pts = list(current_pts)
                                found_improvement = True
                                break
                        current_pts[idx] = old_pt
                
                if not found_improvement:
                    # Try random move if no alignment worked
                    # (Keep some randomness)
                    nx = x + rng.randint(-2, 2)
                    ny = y + rng.randint(-2, 2)
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        new_pt = (nx, ny)
                        if new_pt not in current_pts:
                            old_pt = current_pts[idx]
                            current_pts[idx] = new_pt
                            if _valid_no_four(current_pts):
                                new_score = _score(current_pts)
                                if new_score < current_score:
                                    current_score = new_score
                                    if current_score < best_score:
                                        best_score = current_score
                                        best_pts = list(current_pts)
                            current_pts[idx] = old_pt

    # Strategy C: Random small box
    for attempt in range(30000):
        if time.time() - start > 109.5:
            break
        pts = []
        for _ in range(20):
            pts.append((rng.randint(-6, 6), rng.randint(-6, 6)))
        consider(pts)
        if best_score <= 12:
            break

    if best_pts is None:
        return [[x, x * x] for x in range(-10, 10)]

    return [list(p) for p in best_pts]
