import time
import random
import math

# EVOLVE-BLOCK-START

def _collinear3(x1, y1, x2, y2, x3, y3):
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _no_four_collinear(points):
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        slopes = {}
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0:
                key = (0, 1)
            elif dy == 0:
                key = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx, dy = -dx, -dy
                key = (dx, dy)
            slopes[key] = slopes.get(key, 0) + 1
        for c in slopes.values():
            if c >= 3:
                return False
    return True

def _precompute_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            for k in range(j + 1, n):
                xk, yk = points[k]
                if dx * (yk - yi) == dy * (xk - xi):
                    triples.append((i, j, k))
    return triples

def _score(points, time_limit=3.0):
    n = len(points)
    triples = _precompute_triples(points)
    if not triples:
        return n
    
    # Build bitmask representation
    triple_masks = [(1 << i) | (1 << j) | (1 << k) for (i, j, k) in triples]
    
    # Degree and ordering
    degree = [0] * n
    for (i, j, k) in triples:
        degree[i] += 1
        degree[j] += 1
        degree[k] += 1
    order = sorted(range(n), key=lambda x: -degree[x])
    
    # For each point, list of triple masks containing it
    point_triple_masks = [[] for _ in range(n)]
    for tm in triple_masks:
        for p in range(n):
            if tm & (1 << p):
                point_triple_masks[p].append(tm)
    
    start_time = time.time()
    best_size = [0]
    
    def bnb(idx, cur_mask, cur_size):
        if time.time() - start_time > time_limit:
            return
        if cur_size + (n - idx) <= best_size[0]:
            return
        if idx == n:
            if cur_size > best_size[0]:
                best_size[0] = cur_size
            return
        p = order[idx]
        new_mask = cur_mask | (1 << p)
        valid = True
        for tm in point_triple_masks[p]:
            if (new_mask & tm) == tm:
                valid = False
                break
        if valid:
            bnb(idx + 1, new_mask, cur_size + 1)
        bnb(idx + 1, cur_mask, cur_size)
    
    bnb(0, 0, 0)
    return best_size[0]

def _initial_config():
    """
    Start with points on/near a parabola y = x^2, perturbed to create triples.
    A parabola has no 3 collinear points, so we perturb to create controlled collinearities.
    """
    random.seed(42)
    # Base points on parabola
    base_x = list(range(-10, 10))
    points = []
    for x in base_x:
        y = x * x
        # Small perturbation
        px = x + random.randint(-1, 1)
        py = y + random.randint(-3, 3)
        # Clamp to bounds
        px = max(-100, min(100, px))
        py = max(-100, min(100, py))
        points.append([px, py])
    
    # Try to repair if invalid
    if not _no_four_collinear(points):
        points = _repair_config(points)
    
    # Ensure distinctness
    seen = set()
    unique = []
    for p in points:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            unique.append(p)
    while len(unique) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    
    return unique[:20]

def _repair_config(points):
    """Move points to remove 4-collinearities."""
    random.seed(99)
    for _ in range(2000):
        if _no_four_collinear(points):
            return points
        # Find a 4-collinear violation
        violated = False
        for i in range(20):
            xi, yi = points[i]
            slopes = {}
            for j in range(20):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0:
                    key = (0, 1)
                elif dy == 0:
                    key = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx //= g
                    dy //= g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    key = (dx, dy)
                slopes[key] = slopes.get(key, 0) + 1
            for key, cnt in slopes.items():
                if cnt >= 3:
                    violated = True
                    # Move one of the collinear points
                    # Find a point on this line from i
                    target_idx = None
                    for j in range(20):
                        if i == j:
                            continue
                        xj, yj = points[j]
                        dx = xj - xi
                        dy = yj - yi
                        if dx == 0:
                            k2 = (0, 1)
                        elif dy == 0:
                            k2 = (1, 0)
                        else:
                            g = math.gcd(abs(dx), abs(dy))
                            dx2 = dx // g
                            dy2 = dy // g
                            if dx2 < 0:
                                dx2, dy2 = -dx2, -dy2
                            k2 = (dx2, dy2)
                        if k2 == key:
                            target_idx = j
                            break
                    if target_idx is not None:
                        points[target_idx] = [random.randint(-50, 50), random.randint(-50, 50)]
                    break
            if violated:
                break
    return points

def _generate_neighbor(points, idx, rng):
    """Generate a candidate move for point idx, biased toward creating triples."""
    xi, yi = points[idx]
    # With some probability, move to a point that creates a new triple with two existing points
    if rng.random() < 0.4:
        # Pick two other points and try to find a third collinear point
        others = [j for j in range(20) if j != idx]
        j, k = rng.sample(others, 2)
        xj, yj = points[j]
        xk, yk = points[k]
        # Line through j and k: find integer points on it within bounds
        dx = xk - xj
        dy = yk - yj
        g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
        if g == 0:
            # Same point, skip
            return None
        step_x = dx // g
        step_y = dy // g
        # Parametrize: (xj + t*step_x, yj + t*step_y)
        # Find t such that point is in bounds and not equal to j or k
        candidates = []
        for t in range(-200, 201):
            nx = xj + t * step_x
            ny = yj + t * step_y
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                if (nx, ny) != (xj, yj) and (nx, ny) != (xk, yk):
                    if (nx, ny) != (xi, yi):
                        candidates.append([nx, ny])
        if candidates:
            return rng.choice(candidates)
    
    # Otherwise, random move near current position
    radius = rng.choice([3, 5, 10, 20, 50])
    nx = xi + rng.randint(-radius, radius)
    ny = yi + rng.randint(-radius, radius)
    nx = max(-100, min(100, nx))
    ny = max(-100, min(100, ny))
    return [nx, ny]

def construct():
    deadline = time.time() + 115
    rng = random.Random(123)
    
    points = _initial_config()
    if not _no_four_collinear(points):
        points = _repair_config(points)
    
    best_points = [p[:] for p in points]
    best_score = _score(best_points, time_limit=4.0)
    
    current_points = [p[:] for p in points]
    current_score = best_score
    
    # Simulated annealing
    T = 8.0
    T_min = 0.1
    alpha = 0.995
    
    while time.time() < deadline - 5:
        idx = rng.randint(0, 19)
        new_pos = _generate_neighbor(current_points, idx, rng)
        if new_pos is None:
            continue
        if [new_pos[0], new_pos[1]] in current_points:
            continue
        
        candidate = [p[:] for p in current_points]
        candidate[idx] = [new_pos[0], new_pos[1]]
        
        if not _no_four_collinear(candidate):
            continue
        
        # Fast preliminary score: count triples as a proxy
        new_score = _score(candidate, time_limit=3.0)
        
        if new_score < current_score:
            current_points = candidate
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = candidate
        else:
            delta = new_score - current_score
            if delta == 0:
                if rng.random() < 0.5:
                    current_points = candidate
                    current_score = new_score
            else:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_points = candidate
                    current_score = new_score
        
        T = max(T_min, T * alpha)
    
    return best_points

# EVOLVE-BLOCK-END
