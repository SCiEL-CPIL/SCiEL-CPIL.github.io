"""Structural: conflict-graph greedy scoring + systematic collinearity-breaking search."""
import time
import random
from itertools import combinations
from math import gcd

_DEADLINE = 140.0

def _elapsed(start):
    return time.time() - start

def _check_no_four_collinear(points):
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

def _build_conflict_graph(points):
    """Build adjacency: edge (i,j) if points i,j,k are collinear for some k.
    Actually, for the independent set problem, we need: a subset S is valid
    iff no three points in S are collinear.
    
    Reformulation: Create a hypergraph where hyperedges are collinear triples.
    We want the largest subset with no hyperedge fully contained.
    
    For greedy: iteratively pick points that 'break' the most triples.
    """
    n = len(points)
    # Find all collinear triples
    triples = []
    seen = set()
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
                            t = frozenset([line_pts[a], line_pts[b], line_pts[c]])
                            if t not in seen:
                                seen.add(t)
                                triples.append(t)
    return triples

def _score_greedy(points, timeout=5.0, start_time=None):
    """
    Exact-ish scoring via greedy + local improvement on the hypergraph.
    We want: max |S| such that no triple is fully in S.
    
    Strategy: Start with S = all points. While there's a violated triple,
    remove the point from it that appears in the most other violated triples.
    This gives a valid (independent) set. Then try to add back points.
    
    For exactness on small instances, this greedy is often tight.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(points)
    triples = _build_conflict_graph(points)
    
    if not triples:
        return n
    
    # Greedy removal: repeatedly remove the point that hits the most remaining violated triples
    in_set = [True] * n
    # Count how many triples each point is in (among those still "alive")
    # A triple is "alive" if all 3 points are still in_set
    
    # Precompute point -> list of triple indices
    point_triples = [[] for _ in range(n)]
    for ti, t in enumerate(triples):
        for p in t:
            point_triples[p].append(ti)
    
    # Iterative: find violated triples, remove best point
    removed = 0
    max_iter = n * len(triples) + 100
    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
        # Find all violated triples (all 3 points in_set)
        violated = []
        for ti, t in enumerate(triples):
            if in_set[list(t)[0]] and in_set[list(t)[1]] and in_set[list(t)[2]]:
                violated.append(ti)
        
        if not violated:
            break
        
        # Count for each point in violated triples, how many violated triples it's in
        point_count = {}
        for ti in violated:
            t = triples[ti]
            for p in t:
                if in_set[p]:
                    point_count[p] = point_count.get(p, 0) + 1
        
        # Remove the point with highest count
        best_p = max(point_count, key=lambda p: point_count[p])
        in_set[best_p] = False
        removed += 1
    
    current_size = sum(in_set)
    
    # Try to add back removed points if they don't create violations
    for p in range(n):
        if in_set[p]:
            continue
        if time.time() - start_time > timeout:
            break
        # Check if adding p back creates any violated triple
        can_add = True
        for ti in point_triples[p]:
            t = triples[ti]
            others = [x for x in t if x != p]
            if len(others) == 2 and in_set[others[0]] and in_set[others[1]]:
                can_add = False
                break
        if can_add:
            in_set[p] = True
            current_size += 1
    
    return current_size

def _score_exact_bb(points, timeout=8.0, start_time=None):
    """Exact branch and bound using the conflict hypergraph.
    For n=20, this is feasible if we prune well.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(points)
    triples = _build_conflict_graph(points)
    
    if not triples:
        return n
    
    # Convert triples to bitmasks for speed
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    # Precompute: for each point, which triple masks contain it
    point_in = [[] for _ in range(n)]
    for ti, tm in enumerate(triple_masks):
        for p in range(n):
            if tm & (1 << p):
                point_in[p].append(ti)
    
    degrees = [len(point_in[p]) for p in range(n)]
    
    best = [0]
    
    def search(available, active_triples, removed):
        if time.time() - start_time > timeout:
            return
        
        # Upper bound
        ub = available.bit_count()
        if ub <= best[0]:
            return
        
        if not active_triples:
            if ub > best[0]:
                best[0] = ub
            return
        
        # Find most constrained triple (fewest available points)
        best_ti = -1
        best_cands = None
        best_cnt = n + 1
        
        for ti in active_triples:
            tm = triple_masks[ti]
            cands = [p for p in range(n) if (tm >> p) & 1 and (available >> p) & 1]
            if len(cands) < best_cnt:
                best_cnt = len(cands)
                best_cands = cands
                best_ti = ti
            if best_cnt <= 1:
                break
        
        if best_cnt == 0:
            # Triple already broken
            search(available, active_triples[1:], removed)
            return
        
        if best_cnt == 1:
            p = best_cands[0]
            new_avail = available & ~(1 << p)
            new_active = [ti for ti in active_triples if not (triple_masks[ti] & (1 << p))]
            search(new_avail, new_active, removed + 1)
            return
        
        # Branch: remove each candidate, ordered by degree
        best_cands.sort(key=lambda p: -degrees[p])
        for p in best_cands:
            new_avail = available & ~(1 << p)
            new_active = [ti for ti in active_triples if not (triple_masks[ti] & (1 << p))]
            search(new_avail, new_active, removed + 1)
            if time.time() - start_time > timeout:
                return
    
    all_mask = (1 << n) - 1
    search(all_mask, list(range(len(triple_masks))), 0)
    return best[0]

def _score(points, start_time=None, timeout=8.0):
    """Score: first try greedy (fast), then exact B&B if time permits."""
    if start_time is None:
        start_time = time.time()
    
    # Fast greedy score first
    g_score = _score_greedy(points, timeout=1.0, start_time=start_time)
    
    # If we have time, verify with exact B&B
    if _elapsed(start_time) < timeout - 1.0:
        e_score = _score_exact_bb(points, timeout=timeout, start_time=start_time)
        return e_score
    
    return g_score

def _two_parabola(shift=1, x_range=None):
    if x_range is None:
        x_range = list(range(-5, 5))
    points = []
    for x in x_range:
        points.append([x, x * x])
    for x in x_range:
        points.append([x, x * x + shift])
    
    seen = set()
    unique = []
    for p in points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    if len(unique) != 20:
        return None
    if not all(abs(p[0]) <= 100 and abs(p[1]) <= 100 for p in unique):
        return None
    return unique

def _random_config(seed=None):
    rng = random.Random(seed)
    points = []
    seen = set()
    attempts = 0
    while len(points) < 20 and attempts < 50000:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        key = (x, y)
        if key not in seen:
            seen.add(key)
            points.append([x, y])
        attempts += 1
    if len(points) != 20:
        return None
    return points

def _find_collinear_lines(points):
    """Return list of lines, each a list of point indices on that line (len >= 3)."""
    n = len(points)
    lines_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                key = (0, 1, x1)
            elif dy == 0:
                key = (1, 0, y1)
            else:
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                c = dy_n * x1 - dx_n * y1
                key = (dx_n, dy_n, c)
            
            if key not in lines_dict:
                lines_dict[key] = set()
            lines_dict[key].add(i)
            lines_dict[key].add(j)
    
    result = []
    for key, pts in lines_dict.items():
        if len(pts) >= 3:
            result.append(sorted(pts))
    return result

def _break_collinearity(points, rng):
    """
    Systematic perturbation: find a collinear line with >= 3 points,
    and move one of them off the line. This directly reduces triple count.
    """
    lines = _find_collinear_lines(points)
    if not lines:
        return None  # No collinear triples to break
    
    # Pick the longest line (most triples to break)
    lines.sort(key=len, reverse=True)
    line = lines[0]
    
    new_points = [list(p) for p in points]
    
    # Try moving each point on the line
    for idx in line:
        x, y = new_points[idx]
        # Try several small moves
        for dx in [-3, -2, -1, 1, 2, 3]:
            for dy in [-3, -2, -1, 1, 2, 3]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (-100 <= nx <= 100 and -100 <= ny <= 100):
                    continue
                # Check distinctness
                test_pt = [nx, ny]
                if any(test_pt == p for p in new_points):
                    continue
                new_points[idx] = test_pt
                if _check_no_four_collinear(new_points):
                    return new_points
                new_points[idx] = [x, y]
    
    return None

def _perturb_random(points, rng):
    """Random perturbation as fallback."""
    new_points = [list(p) for p in points]
    num_perturb = rng.choice([1, 1, 2])
    for _ in range(num_perturb):
        idx = rng.randint(0, 19)
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        nx = new_points[idx][0] + dx
        ny = new_points[idx][1] + dy
        if -100 <= nx <= 100 and -100 <= ny <= 100:
            new_points[idx] = [nx, ny]
    
    seen = set()
    for p in new_points:
        key = tuple(p)
        if key in seen:
            return None
        seen.add(key)
    return new_points

def _local_search(initial_points, start_time, deadline, rng):
    """Hill-climbing with systematic + random perturbations."""
    current = [list(p) for p in initial_points]
    current_score = _score(current, start_time=start_time, timeout=8.0)
    best_points = [list(p) for p in current]
    best_score = current_score
    
    stall = 0
    while _elapsed(start_time) < deadline:
        # Alternate between systematic and random perturbation
        if rng.random() < 0.5:
            perturbed = _break_collinearity(current, rng)
        else:
            perturbed = _perturb_random(current, rng)
        
        if perturbed is None:
            stall += 1
            if stall > 80:
                break
            continue
        
        if not _check_no_four_collinear(perturbed):
            stall += 1
            if stall > 80:
                break
            continue
        
        s = _score(perturbed, start_time=start_time, timeout=8.0)
        if s < current_score:
            current = perturbed
            current_score = s
            best_points = [list(p) for p in current]
            best_score = s
            stall = 0
        else:
            stall += 1
            if stall > 120:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_points = None
    best_score = 999
    rng = random.Random(42)
    
    initial_configs = []
    
    # Two-parabola configurations
    for shift in [1, 2, 3, -1, -2, 5, 7, -3, 4, -4]:
        config = _two_parabola(shift)
        if config is not None and _check_no_four_collinear(config):
            initial_configs.append(config)
    
    # Different x ranges
    for shift in [1, 2, -1, 3]:
        for xr in [list(range(-6, 4)), list(range(-4, 6)), list(range(-7, 3)), list(range(-8, 2))]:
            config = _two_parabola(shift, xr)
            if config is not None and _check_no_four_collinear(config):
                initial_configs.append(config)
    
    # Random configs
    for seed in [1, 2, 3, 4, 5]:
        config = _random_config(seed)
        if config is not None and _check_no_four_collinear(config):
            initial_configs.append(config)
    
    if not initial_configs:
        initial_configs.append([[x, x * x] for x in range(-10, 10)])
    
    # Sort initial configs by greedy score to prioritize promising ones
    scored_configs = []
    for config in initial_configs:
        if _elapsed(start_time) > deadline - 20:
            break
        s = _score_greedy(config, timeout=2.0, start_time=start_time)
        scored_configs.append((s, config))
    
    scored_configs.sort(key=lambda x: x[0])
    
    for s, config in scored_configs:
        if _elapsed(start_time) > deadline - 15:
            break
        
        result_points, result_score = _local_search(config, start_time, deadline - 5, rng)
        
        if result_score < best_score:
            best_score = result_score
            best_points = [list(p) for p in result_points]
        
        if best_score <= 10:
            break
    
    # Final validity check
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    if len(set(tuple(p) for p in best_points)) != 20:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    if not _check_no_four_collinear(best_points):
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    for p in best_points:
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            best_points = [[x, x * x] for x in range(-10, 10)]
            break
    
    return best_points
