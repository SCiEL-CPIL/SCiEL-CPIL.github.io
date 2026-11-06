"""Structural change: model as minimum hitting set of collinear triples.
Construct configurations with many forced collinearities using line-based placement,
then refine via local search minimizing the hitting set size (= 20 - score)."""
import random
import time
import math
from math import gcd
from itertools import combinations

DEADLINE = time.time() + 135  # safety margin before 150s kill

def _collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx, dy = dx // g, dy // g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    c = dx * p1[1] - dy * p1[0]
    gc = gcd(gcd(abs(dx), abs(dy)), abs(c))
    if gc > 1:
        dx, dy, c = dx // gc, dy // gc, c // gc
    return (dx, dy, c)

def _get_lines_and_triples(points):
    """Get all lines with >=3 points and all collinear triples."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(i)
    
    lines = {}
    for key, pts in line_map.items():
        if len(pts) >= 3:
            lines[key] = pts
    
    triples = []
    for key, pts in lines.items():
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                for k in range(j + 1, len(pts)):
                    triples.append((pts[i], pts[j], pts[k]))
    
    return lines, triples

def _no_four_collinear(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    for key, count in line_counts.items():
        k_sq = 1 + 8 * count
        k_int = math.isqrt(k_sq)
        if k_int * k_int == k_sq:
            k = (1 + k_int) // 2
            if k >= 4:
                return False
    return True

def _min_hitting_set_bnb(points, triples, deadline):
    """Find minimum hitting set of collinear triples via branch and bound.
    A hitting set H hits every triple (intersects it).
    Score = n - |H| = size of largest 3-collinear-free subset."""
    n = len(points)
    if not triples:
        return n
    
    # Sort triples by size (all size 3 here) and by frequency of points
    point_freq = [0] * n
    for t in triples:
        for p in t:
            point_freq[p] += 1
    
    # Order points by frequency descending
    order = sorted(range(n), key=lambda i: -point_freq[i])
    pos_in_order = {order[i]: i for i in range(n)}
    
    best_hs_size = [n]  # worst case: remove all
    current_hs = [False] * n
    
    # Precompute for each point, which triples it hits
    point_triples = [[] for _ in range(n)]
    for ti, t in enumerate(triples):
        for p in t:
            point_triples[p].append(ti)
    
    def bnb(idx, hs_count, remaining_triples):
        if time.time() > deadline:
            return
        if hs_count >= best_hs_size[0]:
            return
        if not remaining_triples:
            best_hs_size[0] = hs_count
            return
        
        # Lower bound: each remaining triple needs at least one more point
        # but one point can hit multiple triples
        # Simple bound: we need at least 1 more if any remain
        if hs_count + 1 >= best_hs_size[0] and len(remaining_triples) <= 1:
            return
        
        # Pick the next point to consider
        if idx >= n:
            return
        
        pt = order[idx]
        
        # Option 1: don't include pt in hitting set
        # Triples not hit by pt remain
        bnb(idx + 1, hs_count, remaining_triples)
        
        # Option 2: include pt in hitting set
        # Remove all triples hit by pt
        new_remaining = []
        for ti in remaining_triples:
            if pt not in triples[ti]:
                new_remaining.append(ti)
        bnb(idx + 1, hs_count + 1, new_remaining)
    
    all_triple_indices = list(range(len(triples)))
    bnb(0, 0, all_triple_indices)
    
    return n - best_hs_size[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    if not _no_four_collinear(points):
        return 21  # invalid
    _, triples = _get_lines_and_triples(points)
    return _min_hitting_set_bnb(points, triples, DEADLINE)

def _build_line_based_config(seed):
    """Build a configuration by placing points on carefully chosen lines.
    Strategy: use 6-8 lines with 3 points each, ensuring no line has 4 points
    and minimizing accidental collinearities."""
    rng = random.Random(seed)
    
    # Define a set of lines with specific slopes and intercepts
    # to control where collinearities occur
    line_specs = []
    
    # Horizontal lines
    for y in [-40, -20, 0, 20, 40]:
        line_specs.append(('h', y))
    
    # Vertical lines
    for x in [-40, -20, 0, 20, 40]:
        line_specs.append(('v', x))
    
    # Diagonal lines with various slopes
    for slope in [1, -1, 2, -2]:
        for intercept in [-30, 0, 30]:
            line_specs.append(('d', slope, intercept))
    
    rng.shuffle(line_specs)
    
    used = set()
    points = []
    
    # Try to place 3 points on each selected line
    for spec in line_specs:
        if len(points) >= 20:
            break
        if len(spec) == 2:
            if spec[0] == 'h':
                y = spec[1]
                candidates = [(x, y) for x in range(-100, 101) if (x, y) not in used]
            else:  # 'v'
                x = spec[1]
                candidates = [(x, y) for y in range(-100, 101) if (x, y) not in used]
        else:  # diagonal
            slope, intercept = spec[1], spec[2]
            candidates = []
            for x in range(-100, 101):
                y = slope * x + intercept
                if -100 <= y <= 100 and (x, y) not in used:
                    candidates.append((x, y))
        
        if len(candidates) < 3:
            continue
        
        # Pick 3 points spread out on this line
        rng.shuffle(candidates)
        # Prefer points far apart to reduce accidental collinearities with other lines
        selected = []
        for p in candidates:
            if len(selected) >= 3:
                break
            # Check if adding this point creates a 4-collinear situation
            test_points = list(used) + selected + [p]
            if _no_four_collinear(test_points):
                selected.append(p)
        
        if len(selected) == 3:
            used.update(selected)
            points.extend(selected)
    
    # Fill remaining points
    if len(points) < 20:
        all_points = [(x, y) for x in range(-100, 101) for y in range(-100, 101) if (x, y) not in used]
        rng.shuffle(all_points)
        for p in all_points:
            if len(points) >= 20:
                break
            test_points = points + [p]
            if _no_four_collinear(test_points):
                points.append(p)
    
    if len(points) != 20:
        return None
    
    return points

def _local_search(points, rng, time_limit):
    """Local search: swap points to reduce score."""
    current_points = list(points)
    _, current_triples = _get_lines_and_triples(current_points)
    current_score = _min_hitting_set_bnb(current_points, current_triples, time_limit)
    
    while time.time() < time_limit:
        improved = False
        for _ in range(200):
            if time.time() > time_limit:
                break
            i = rng.randrange(20)
            new_pt = (rng.randint(-100, 100), rng.randint(-100, 100))
            if new_pt in current_points:
                continue
            candidate = list(current_points)
            candidate[i] = new_pt
            if not _no_four_collinear(candidate):
                continue
            _, cand_triples = _get_lines_and_triples(candidate)
            new_score = _min_hitting_set_bnb(candidate, cand_triples, time_limit)
            if new_score < current_score:
                current_points = candidate
                current_score = new_score
                improved = True
                break
        if not improved:
            break
    
    return current_points, current_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = 21
    
    # Phase 1: Try line-based constructions
    for seed in range(30):
        if time.time() > DEADLINE:
            break
        points = _build_line_based_config(seed)
        if points is None:
            continue
        if len(set(points)) != 20:
            continue
        if not all(abs(x) <= 100 and abs(y) <= 100 for x, y in points):
            continue
        if not _no_four_collinear(points):
            continue
        
        score = _score(points)
        if score < best_score:
            best_score = score
            best_points = list(points)
        if best_score <= 12:
            break
    
    # Phase 2: Try random constructions with dense collinearities
    if best_points is None:
        for seed in range(50):
            if time.time() > DEADLINE:
                break
            rng = random.Random(seed)
            pool = [(x, y) for x in range(-60, 61) for y in range(-60, 61)]
            used = set()
            points = []
            
            # First, create some collinear triples
            for _ in range(8):
                if len(points) >= 20:
                    break
                p1 = rng.choice(pool)
                p2 = rng.choice(pool)
                if p1 == p2:
                    continue
                line_pts = [p for p in pool if _collinear(p1, p2, p)]
                if len(line_pts) < 3:
                    continue
                rng.shuffle(line_pts)
                triple = []
                for p in line_pts:
                    if p not in used:
                        triple.append(p)
                        if len(triple) == 3:
                            break
                if len(triple) == 3:
                    candidate = list(used) + triple
                    if _no_four_collinear(candidate):
                        used.update(triple)
                        points.extend(triple)
            
            # Fill rest
            remaining = [p for p in pool if p not in used]
            rng.shuffle(remaining)
            for p in remaining:
                if len(points) >= 20:
                    break
                if _no_four_collinear(points + [p]):
                    points.append(p)
            
            if len(points) != 20:
                continue
            if not _no_four_collinear(points):
                continue
            
            score = _score(points)
            if score < best_score:
                best_score = score
                best_points = list(points)
    
    if best_points is None:
        # Fallback
        fallback = [
            (-10, 0), (-5, 0), (0, 0),
            (5, 10), (10, 10), (15, 10),
            (-15, -10), (-20, -10), (-25, -10),
            (20, -5), (20, 0), (20, 5),
            (-30, -25), (-30, -20), (-30, -15),
            (30, 20), (30, 25),
            (40, 30), (40, 35),
        ]
        if _no_four_collinear(fallback):
            return [[x, y] for x, y in fallback]
        return [[x, x * x] for x in range(-10, 10)]
    
    # Phase 3: Local search refinement
    remaining_time = DEADLINE - time.time()
    if remaining_time > 15:
        rng = random.Random(42)
        improved_points, improved_score = _local_search(best_points, rng, time.time() + remaining_time - 5)
        if improved_score < best_score:
            best_points = improved_points
            best_score = improved_score
    
    return [[x, y] for x, y in best_points]
