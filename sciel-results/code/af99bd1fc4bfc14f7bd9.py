import time
import random
from math import gcd

_START_TIME = None
_DEADLINE = 135.0  # Leave 15s margin before 150s kill

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
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

def _count_triples(points):
    """Count the number of collinear triples."""
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
            m = 0
            for k in range(n):
                x3, y3 = points[k]
                if a * x3 + b * y3 + c == 0:
                    m += 1
            if m >= 3:
                count += m * (m - 1) * (m - 2) // 6
    return count

def _build_triple_conflicts(points):
    """Build list of collinear triples as bitmasks."""
    n = len(points)
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
            line_points = [i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    line_points.append(k)
            m = len(line_points)
            if m >= 3:
                for a in range(m):
                    for b in range(a + 1, m):
                        for c in range(b + 1, m):
                            triple = (line_points[a], line_points[b], line_points[c])
                            if triple not in seen:
                                seen.add(triple)
                                triples.append(triple)
    return triples

def _max_independent_set_size(triples, n):
    """
    Exact branch and bound for maximum subset with no three collinear.
    Uses bitmask representation for speed.
    """
    if not triples:
        return n
    
    # Build conflict graph: for each point, which pairs with it form a triple
    # conflict[i] = set of (j,k) such that {i,j,k} is a triple
    conflict = [set() for _ in range(n)]
    for (a, b, c) in triples:
        conflict[a].add((b, c))
        conflict[b].add((a, c))
        conflict[c].add((a, b))
    
    best = [0]
    selected = [False] * n
    
    # Order by degree (most constrained first)
    degrees = [len(conflict[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    def backtrack(pos, count):
        if _elapsed() > _DEADLINE:
            return
        if pos == n:
            if count > best[0]:
                best[0] = count
            return
        
        if count + (n - pos) <= best[0]:
            return
        
        idx = order[pos]
        
        # Try including idx
        can_include = True
        for (j, k) in conflict[idx]:
            if selected[j] and selected[k]:
                can_include = False
                break
        
        if can_include:
            selected[idx] = True
            backtrack(pos + 1, count + 1)
            selected[idx] = False
        
        # Try excluding idx
        backtrack(pos + 1, count)
    
    backtrack(0, 0)
    return best[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    triples = _build_triple_conflicts(points)
    return _max_independent_set_size(triples, len(points))

def _greedy_construct():
    """
    Greedy construction that maximizes collinear triples from the start.
    Builds points one by one, choosing positions that create the most new triples.
    """
    random.seed(42)
    points = []
    seen = set()
    
    # Start with a few anchor points that form a triple
    # Use a line with 3 points, then build around it
    anchor_line = [(0, 0), (1, 0), (2, 0)]
    for p in anchor_line:
        points.append([p[0], p[1]])
        seen.add(p)
    
    # Now greedily add points that maximize triples
    for _ in range(17):
        if _elapsed() > _DEADLINE - 30:
            break
            
        best_pos = None
        best_triple_gain = -1
        
        # Try candidate positions
        candidates = []
        for x in range(-50, 51):
            for y in range(-50, 51):
                if (x, y) in seen:
                    continue
                candidates.append((x, y))
        
        # Sample a subset of candidates for speed
        if len(candidates) > 2000:
            random.shuffle(candidates)
            candidates = candidates[:2000]
        
        for x, y in candidates:
            new_points = points + [[x, y]]
            
            # Quick check: no 4 collinear
            valid = True
            for i in range(len(new_points)):
                x1, y1 = new_points[i]
                # Only need to check lines through the new point
                if i == len(new_points) - 1:
                    continue
                for j in range(len(new_points)):
                    if i == j or j == len(new_points) - 1:
                        continue
                    x2, y2 = new_points[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        continue
                    count = 2
                    for k in range(len(new_points)):
                        if k == i or k == j:
                            continue
                        x3, y3 = new_points[k]
                        if dx * (y3 - y1) == dy * (x3 - x1):
                            count += 1
                    if count >= 4:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Count triples involving the new point
            triple_gain = 0
            for i in range(len(new_points)):
                x1, y1 = new_points[i]
                for j in range(i + 1, len(new_points)):
                    if i == len(new_points) - 1 and j == len(new_points) - 1:
                        continue
                    x2, y2 = new_points[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        continue
                    count = 0
                    for k in range(len(new_points)):
                        x3, y3 = new_points[k]
                        if dx * (y3 - y1) == dy * (x3 - x1):
                            count += 1
                    if count >= 3:
                        # Check if this line includes the new point
                        new_x, new_y = new_points[-1]
                        if dx * (new_y - y1) == dy * (new_x - x1):
                            triple_gain += 1
            
            if triple_gain > best_triple_gain:
                best_triple_gain = triple_gain
                best_pos = (x, y)
        
        if best_pos is not None:
            points.append([best_pos[0], best_pos[1]])
            seen.add(best_pos)
        else:
            # Fallback: add a random valid point
            while True:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in seen:
                    new_points = points + [[x, y]]
                    if _check_no_four_collinear(new_points):
                        points.append([x, y])
                        seen.add((x, y))
                        break
    
    return points

def _parabolic_config():
    """
    Generate a configuration based on a parabola y = x^2 with perturbations.
    """
    random.seed(42)
    points = []
    x_vals = list(range(-5, 6))
    random.shuffle(x_vals)
    x_vals = x_vals[:20]
    
    for x in x_vals:
        y = x * x
        px = random.randint(-2, 2)
        py = random.randint(-3, 3)
        points.append([x + px, y + py])
    
    seen = set()
    final_points = []
    for p in points:
        x, y = p
        if x < -100: x = -100
        if x > 100: x = 100
        if y < -100: y = -100
        if y > 100: y = 100
        if (x, y) not in seen:
            seen.add((x, y))
            final_points.append([x, y])
    
    while len(final_points) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            final_points.append([x, y])
    
    return final_points

def _grid_config():
    """Generate a grid-based configuration with perturbations."""
    random.seed(123)
    candidates = []
    for x in range(-4, 5):
        for y in range(-4, 5):
            candidates.append([x, y])
    random.shuffle(candidates)
    points = [list(p) for p in candidates[:20]]
    
    for _ in range(100):
        if _check_no_four_collinear(points):
            break
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
                    new_x = old_x + random.randint(-3, 3)
                    new_y = old_y + random.randint(-3, 3)
                    if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                        if (new_x, new_y) not in [(p[0], p[1]) for k, p in enumerate(points) if k != idx]:
                            points[idx] = [new_x, new_y]
                            found = True
                            break
            if found:
                break
    return points

def _random_valid_config():
    """Generate a random valid configuration."""
    while True:
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            key = (x, y)
            if key not in seen:
                seen.add(key)
                points.append([x, y])
        if _check_no_four_collinear(points):
            return points

def _improve_triples(points, max_time=30.0):
    """
    Local search to maximize the number of collinear triples.
    Uses simulated annealing-like acceptance.
    """
    current_triples = _count_triples(points)
    best_triples = current_triples
    best_points = [list(p) for p in points]
    start_time = time.time()
    temp = 1.0
    
    while time.time() - start_time < max_time:
        if _elapsed() > _DEADLINE - 10:
            break
        
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        
        best_move = None
        best_delta = 0
        
        for attempt in range(20):
            if attempt < 10:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                new_x = old_x + dx
                new_y = old_y + dy
            else:
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
            
            if new_x == old_x and new_y == old_y:
                continue
            if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                continue
            
            if (new_x, new_y) in [(p[0], p[1]) for i, p in enumerate(points) if i != idx]:
                continue
            
            new_points = [list(p) for p in points]
            new_points[idx] = [new_x, new_y]
            
            if not _check_no_four_collinear(new_points):
                continue
            
            new_triples = _count_triples(new_points)
            delta = new_triples - current_triples
            
            if delta > 0:
                best_move = new_points
                best_delta = delta
                break
            else:
                import math
                prob = math.exp(max(delta * 0.5, -10) / temp)
                if random.random() < prob:
                    best_move = new_points
                    best_delta = delta
        
        if best_move:
            points = best_move
            current_triples += best_delta
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = [list(p) for p in points]
        
        temp = max(0.1, temp * 0.99)
    
    return best_points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 21
    
    candidates = []
    
    # Greedy construction (primary strategy)
    candidates.append(_greedy_construct())
    
    # Parabolic configuration
    candidates.append(_parabolic_config())
    
    # Grid configuration
    candidates.append(_grid_config())
    
    # Random configurations
    for _ in range(3):
        if _elapsed() > _DEADLINE - 30:
            break
        candidates.append(_random_valid_config())
    
    # Evaluate and improve each candidate
    for config in candidates:
        if _elapsed() > _DEADLINE - 15:
            break
        
        if not _check_no_four_collinear(config):
            continue
        
        improved = _improve_triples(config, max_time=20.0)
        
        if _elapsed() > _DEADLINE - 5:
            break
        
        s = _score(improved)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in improved]
    
    # Final improvement on the best
    if best_points and _elapsed() < _DEADLINE - 10:
        improved = _improve_triples(best_points, max_time=15.0)
        if _elapsed() < _DEADLINE - 3:
            s = _score(improved)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in improved]
    
    if best_points is None:
        best_points = _random_valid_config()
    
    return best_points
