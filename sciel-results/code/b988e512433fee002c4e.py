"""Search for 20 integer points with no four collinear and a small no-three-collinear subset."""
import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

_DEADLINE_SECONDS = 135.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _has_four_collinear(points):
    """Check if any 4 points are collinear. Returns True if found."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dxn, dyn = dx // g, dy // g
            else:
                dxn, dyn = dx, dy
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn, dyn = -dxn, -dyn
            cnt = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if (points[k][0] - points[i][0]) * dyn == (points[k][1] - points[i][1]) * dxn:
                    cnt += 1
                    if cnt >= 2:
                        return True
    return False


def _count_triples(points):
    """Count all collinear triples in the point set."""
    n = len(points)
    triples = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.add((i, j, k))
    return triples


def _max_no_three_collinear(points, deadline=None):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    if n <= 2:
        return n

    # Precompute for each point i, which pairs (a,b) would form a triple if i is also chosen
    forbidden_pairs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    forbidden_pairs[i].append((j, k))
                    forbidden_pairs[j].append((i, k))
                    forbidden_pairs[k].append((i, j))

    best = [0]

    def can_add(i, chosen):
        for (a, b) in forbidden_pairs[i]:
            if (chosen >> a) & 1 and (chosen >> b) & 1:
                return False
        return True

    def branch(start, size, chosen):
        if deadline is not None and time.time() > deadline:
            return
        if size > best[0]:
            best[0] = size
        if start >= n:
            return
        if size + (n - start) <= best[0]:
            return
        if can_add(start, chosen):
            branch(start + 1, size + 1, chosen | (1 << start))
        branch(start + 1, size, chosen)

    branch(0, 0, 0)
    return best[0]


def _greedy_triple_packing(points, rng, deadline=None):
    """
    Greedy approach: try to create as many disjoint collinear triples as possible.
    For each potential line direction, try to place 3 points on it.
    Returns modified points and number of disjoint triples created.
    """
    pts = [p[:] for p in points]
    n = len(pts)
    
    # Track which points are already in a triple
    in_triple = [False] * n
    
    # Generate candidate line directions
    directions = []
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx, dy = dx // g, dy // g
            # Normalize
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            directions.append((dx, dy))
    
    rng.shuffle(directions)
    
    triples_created = 0
    
    for dx, dy in directions:
        if deadline is not None and time.time() > deadline:
            break
        if triples_created >= 8:  # Max possible is floor(20/3)=6, but allow some slack
            break
        
        # Try to find 3 points that can be moved onto a line with this direction
        # Find a base point and try to extend the line
        for base_idx in range(n):
            if in_triple[base_idx]:
                continue
            bx, by = pts[base_idx]
            
            # Try offsets for the other two points on this line
            for off1 in range(1, 8):
                for off2 in range(off1 + 1, 15):
                    # Points would be at base + off1*(dx,dy) and base + off2*(dx,dy)
                    p1x, p1y = bx + off1 * dx, by + off1 * dy
                    p2x, p2y = bx + off2 * dx, by + off2 * dy
                    
                    if not (-100 <= p1x <= 100 and -100 <= p1y <= 100):
                        continue
                    if not (-100 <= p2x <= 100 and -100 <= p2y <= 100):
                        continue
                    
                    # Find two other points that can be moved to these positions
                    candidates1 = []
                    candidates2 = []
                    for idx in range(n):
                        if idx == base_idx or in_triple[idx]:
                            continue
                        if (pts[idx][0], pts[idx][1]) != (p1x, p1y):
                            candidates1.append(idx)
                        if (pts[idx][0], pts[idx][1]) != (p2x, p2y):
                            candidates2.append(idx)
                    
                    if not candidates1 or not candidates2:
                        continue
                    
                    # Try combinations
                    for c1 in candidates1[:5]:
                        for c2 in candidates2[:5]:
                            if c1 == c2:
                                continue
                            
                            # Save original positions
                            orig_c1 = pts[c1][:]
                            orig_c2 = pts[c2][:]
                            
                            # Move points
                            pts[c1] = [p1x, p1y]
                            pts[c2] = [p2x, p2y]
                            
                            # Check no four collinear
                            if _has_four_collinear(pts):
                                pts[c1] = orig_c1
                                pts[c2] = orig_c2
                                continue
                            
                            # Check no duplicate points
                            valid = True
                            seen = set()
                            for p in pts:
                                if (p[0], p[1]) in seen:
                                    valid = False
                                    break
                                seen.add((p[0], p[1]))
                            if not valid:
                                pts[c1] = orig_c1
                                pts[c2] = orig_c2
                                continue
                            
                            # Accept this triple
                            in_triple[base_idx] = True
                            in_triple[c1] = True
                            in_triple[c2] = True
                            triples_created += 1
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
            
            if triples_created >= 8:
                break
        
        if triples_created >= 8:
            break
    
    return pts, triples_created


def _local_search_triple_refine(points, rng, deadline=None):
    """
    Local search: try to increase the number of disjoint collinear triples
    by swapping point positions.
    """
    pts = [p[:] for p in points]
    n = len(pts)
    
    def get_triple_count(pts):
        """Count maximum disjoint collinear triples via greedy."""
        in_triple = [False] * n
        count = 0
        # Try all triples in random order
        all_triples = list(combinations(range(n), 3))
        rng.shuffle(all_triples)
        for i, j, k in all_triples:
            if in_triple[i] or in_triple[j] or in_triple[k]:
                continue
            if _collinear(pts[i], pts[j], pts[k]):
                in_triple[i] = in_triple[j] = in_triple[k] = True
                count += 1
        return count
    
    current_triples = get_triple_count(pts)
    best_triples = current_triples
    best_pts = [p[:] for p in pts]
    
    no_improve = 0
    max_iter = 3000
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        i = rng.randint(0, n - 1)
        orig = pts[i][:]
        
        # Try several new positions
        improved = False
        for attempt in range(30):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try moving near an existing point to create collinearity
            j = rng.randint(0, n - 1)
            if j == i:
                j = (j + 1) % n
            
            # Place near the line through two other points
            k = rng.randint(0, n - 1)
            if k == i or k == j:
                k = (k + 1) % n
            
            # Try to place on line through j and k
            dx = pts[j][0] - pts[k][0]
            dy = pts[j][1] - pts[k][1]
            
            # Parametric: point = k + t*(dx, dy)
            # Try several t values
            for t in range(-10, 11):
                if t == 0:
                    continue
                nx = pts[k][0] + t * dx
                ny = pts[k][1] + t * dy
                
                if not (-100 <= nx <= 100 and -100 <= ny <= 100):
                    continue
                if (nx, ny) == (orig[0], orig[1]):
                    continue
                
                # Check not duplicate
                dup = False
                for m, p in enumerate(pts):
                    if m != i and p[0] == nx and p[1] == ny:
                        dup = True
                        break
                if dup:
                    continue
                
                pts[i] = [nx, ny]
                
                if _has_four_collinear(pts):
                    pts[i] = orig
                    continue
                
                new_triples = get_triple_count(pts)
                if new_triples > best_triples:
                    best_triples = new_triples
                    best_pts = [p[:] for p in pts]
                    improved = True
                    break
                elif new_triples > current_triples:
                    current_triples = new_triples
                    improved = True
                    break
                else:
                    pts[i] = orig
            
            if improved:
                break
        
        if not improved:
            pts[i] = orig
            no_improve += 1
            if no_improve > 500:
                break
        else:
            no_improve = 0
    
    return best_pts, best_triples


def _construct_parabola(rng):
    """Construct points on a parabola y = x^2 + c*x + d with perturbations."""
    pts = []
    used = set()
    for x in range(-10, 10):
        y = x * x + rng.randint(-3, 3)
        for attempt in range(30):
            px = x + rng.randint(-1, 1)
            py = y + rng.randint(-2, 2)
            if -100 <= px <= 100 and -100 <= py <= 100 and (px, py) not in used:
                used.add((px, py))
                pts.append([px, py])
                break
        else:
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
    return pts if len(pts) == 20 else None


def _construct_random(rng):
    """Random construction avoiding 4-collinear."""
    pts = []
    used = set()
    attempts = 0
    while len(pts) < 20 and attempts < 5000:
        attempts += 1
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        if (x, y) in used:
            continue
        new_pts = pts + [[x, y]]
        if _has_four_collinear(new_pts):
            continue
        used.add((x, y))
        pts.append([x, y])
    return pts if len(pts) == 20 else None


def _local_search(points, rng, deadline=None):
    """Local search: try moving points to reduce the no-three-collinear score."""
    pts = [p[:] for p in points]
    current_score = _max_no_three_collinear(pts, deadline=deadline)
    best_score = current_score
    best_pts = [p[:] for p in pts]
    
    n = len(pts)
    for _ in range(2000):
        if deadline is not None and time.time() > deadline:
            break
        
        i = rng.randint(0, n - 1)
        orig_i = pts[i][:]
        
        found_improvement = False
        for attempt in range(20):
            if deadline is not None and time.time() > deadline:
                break
            new_x = rng.randint(-60, 60)
            new_y = rng.randint(-60, 60)
            if (new_x, new_y) == (orig_i[0], orig_i[1]):
                continue
            if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(pts) if j != i):
                continue
            
            pts[i] = [new_x, new_y]
            
            if not _has_four_collinear(pts):
                s = _max_no_three_collinear(pts, deadline=deadline)
                if s < best_score:
                    best_score = s
                    best_pts = [p[:] for p in pts]
                    found_improvement = True
                    break
                else:
                    pts[i] = orig_i
            else:
                pts[i] = orig_i
        
        if not found_improvement:
            pts[i] = orig_i
    
    return best_pts, best_score


def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    rng = random.Random(42)
    
    # Strategy 1: Greedy triple packing on random/structured starting points
    for attempt in range(100):
        if _remaining() < 25.0:
            break
        
        # Start from a random or structured configuration
        if attempt % 3 == 0:
            pts = _construct_parabola(rng)
        else:
            pts = _construct_random(rng)
        
        if pts is None:
            continue
        if not _has_four_collinear(pts):
            # Apply greedy triple packing
            packed_pts, triples = _greedy_triple_packing(pts, rng, deadline=_START + _DEADLINE_SECONDS - 10)
            if packed_pts is not None and not _has_four_collinear(packed_pts):
                s = _max_no_three_collinear(packed_pts, deadline=_START + _DEADLINE_SECONDS - 5)
                if s < best_score:
                    best_score = s
                    best_points = [p[:] for p in packed_pts]
                    if best_score <= 12:
                        break
    
    # Strategy 2: Direct random construction
    for attempt in range(50):
        if _remaining() < 20.0:
            break
        pts = _construct_random(rng)
        if pts is None:
            continue
        s = _max_no_three_collinear(pts, deadline=_START + _DEADLINE_SECONDS - 5)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            if best_score <= 12:
                break
    
    # Strategy 3: Triple-refine local search from best found
    if best_points is not None and _remaining() > 20.0:
        refined_pts, refined_triples = _local_search_triple_refine(
            best_points, rng, deadline=_START + _DEADLINE_SECONDS - 10.0
        )
        if refined_pts is not None and not _has_four_collinear(refined_pts):
            s = _max_no_three_collinear(refined_pts, deadline=_START + _DEADLINE_SECONDS - 5)
            if s < best_score:
                best_score = s
                best_points = refined_pts
    
    # Strategy 4: Standard local search as fallback
    if best_points is not None and _remaining() > 15.0:
        improved_pts, improved_score = _local_search(
            best_points, rng, deadline=_START + _DEADLINE_SECONDS - 5.0
        )
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts
    
    # Fallback: simple parabola
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    return best_points
