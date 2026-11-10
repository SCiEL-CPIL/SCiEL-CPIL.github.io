"""Minimize largest 3-collinear-free subset for 20 points."""
import time
import random
import itertools
from collections import defaultdict

def _no_four_collinear(points):
    """Check no four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
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

def _collinear_triples(points):
    """Return list of collinear triples (as index tuples)."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            line_pts = [i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    line_pts.append(k)
            if len(line_pts) >= 3:
                for a in range(len(line_pts)):
                    for b in range(a+1, len(line_pts)):
                        for c in range(b+1, len(line_pts)):
                            triples.append(tuple(sorted([line_pts[a], line_pts[b], line_pts[c]])))
    return list(set(triples))

def _greedy_independent(points, triples):
    """Greedy upper bound on largest 3-collinear-free subset."""
    n = len(points)
    in_set = [True] * n
    degree = [0] * n
    for t in triples:
        for idx in t:
            degree[idx] += 1
    order = sorted(range(n), key=lambda i: -degree[i])
    result = []
    for idx in order:
        if not in_set[idx]:
            continue
        ok = True
        for t in triples:
            if idx in t:
                others = [x for x in t if x != idx]
                if all(in_set[o] and o in result for o in others):
                    ok = False
                    break
        if ok:
            result.append(idx)
    return len(result)

def _exact_max_independent(points, triples, deadline):
    """Branch and bound for exact largest 3-collinear-free subset."""
    n = len(points)
    
    # Build: for each point, list of (other1, other2) pairs such that {point, other1, other2} is a triple
    # Also: for each pair (a,b), the set of third points c such that {a,b,c} is a triple
    pair_thirds = defaultdict(set)
    for t in triples:
        a, b, c = t
        pair_thirds[(a, b)].add(c)
        pair_thirds[(b, a)].add(c)
        pair_thirds[(a, c)].add(b)
        pair_thirds[(c, a)].add(b)
        pair_thirds[(b, c)].add(a)
        pair_thirds[(c, b)].add(a)
    
    best = [0]
    
    def branch(start_idx, current_size, selected_set):
        nonlocal best
        if time.time() > deadline:
            return
        remaining = n - start_idx
        if current_size + remaining <= best[0]:
            return
        if start_idx == n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Option 1: skip point start_idx
        branch(start_idx + 1, current_size, selected_set)
        
        # Option 2: include point start_idx if no conflict
        can_include = True
        for p in selected_set:
            if (start_idx, p) in pair_thirds:
                for third in pair_thirds[(start_idx, p)]:
                    if third in selected_set:
                        can_include = False
                        break
            if not can_include:
                break
        if can_include:
            selected_set.add(start_idx)
            branch(start_idx + 1, current_size + 1, selected_set)
            selected_set.discard(start_idx)
    
    branch(0, 0, set())
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    random.seed(42)
    
    best_points = None
    best_score = 21
    
    # Helper to evaluate a point configuration
    def evaluate(points):
        if not _no_four_collinear(points):
            return None
        triples = _collinear_triples(points)
        greedy = _greedy_independent(points, triples)
        return greedy, triples
    
    # Strategy 1: Structured initial configs (two shifted parabolas)
    initial_configs = []
    
    # Two shifted parabolas
    for offset in [1, 2, 3, 5, 7]:
        pts = []
        for i in range(10):
            pts.append((i, i * i))
            pts.append((i, i * i + offset))
        initial_configs.append(pts)
    
    # Different x-ranges
    for x_start in range(-5, 6):
        for offset in [1, 2, 3]:
            pts = []
            for i in range(10):
                x = x_start + i
                pts.append((x, x * x))
                pts.append((x, x * x + offset))
            initial_configs.append(pts)
    
    # Random initial configs
    for attempt in range(30):
        pts = []
        attempts = 0
        while len(pts) < 20 and attempts < 500:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in pts:
                pts.append(p)
            attempts += 1
        if len(pts) == 20:
            initial_configs.append(pts)
    
    # Evaluate all initial configs
    for pts in initial_configs:
        if time.time() > deadline - 40:
            break
        if len(set(pts)) != 20:
            continue
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
        result = evaluate(pts)
        if result is None:
            continue
        greedy, triples = result
        if greedy < best_score:
            best_score = greedy
            best_points = list(pts)
    
    # Local search on best found
    if best_points is not None and time.time() < deadline - 20:
        current_points = list(best_points)
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for i in range(20):
                if time.time() > deadline - 20:
                    break
                
                # Try random replacements
                for _ in range(5):
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    new_p = (new_x, new_y)
                    if new_p in current_points:
                        continue
                    
                    new_points = list(current_points)
                    new_points[i] = new_p
                    result = evaluate(new_points)
                    if result is None:
                        continue
                    new_greedy, _ = result
                    if new_greedy < current_score:
                        current_points = new_points
                        current_score = new_greedy
                        if current_score < best_score:
                            best_score = current_score
                            best_points = list(current_points)
                        improved = True
                        break
                
                # Try small perturbations
                if not improved and time.time() < deadline - 20:
                    px, py = current_points[i]
                    for dx in range(-3, 4):
                        if time.time() > deadline - 20:
                            break
                        for dy in range(-3, 4):
                            if dx == 0 and dy == 0:
                                continue
                            new_x = px + dx
                            new_y = py + dy
                            if abs(new_x) > 100 or abs(new_y) > 100:
                                continue
                            new_p = (new_x, new_y)
                            if new_p in current_points:
                                continue
                            new_points = list(current_points)
                            new_points[i] = new_p
                            result = evaluate(new_points)
                            if result is None:
                                continue
                            new_greedy, _ = result
                            if new_greedy < current_score:
                                current_points = new_points
                                current_score = new_greedy
                                if current_score < best_score:
                                    best_score = current_score
                                    best_points = list(current_points)
                                improved = True
                                break
                        if improved:
                            break
    
    # Final exact computation on best found
    if best_points is None:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    if time.time() < deadline - 5:
        triples = _collinear_triples(best_points)
        exact_score = _exact_max_independent(best_points, triples, deadline - 2)
        best_score = exact_score
    
    return [[p[0], p[1]] for p in best_points]
