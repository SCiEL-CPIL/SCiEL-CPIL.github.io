import time
import random
from math import gcd
from collections import defaultdict

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
    """Count the number of collinear triples using line counting."""
    n = len(points)
    count = 0
    line_counts = defaultdict(int)
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
            line_counts[line_key] += 1
    # For each line with m points, number of triples = C(m,3)
    # But we need to count from pairs: each line with m points contributes C(m,3) triples
    # However, line_counts counts pairs, so we need to reconstruct m from pair count
    # C(m,2) = line_count => m = (1 + sqrt(1 + 8*line_count)) / 2
    for lc in line_counts.values():
        # Solve C(m,2) = lc for m
        # m*(m-1)/2 = lc => m^2 - m - 2*lc = 0
        disc = 1 + 8 * lc
        m = int((1 + disc ** 0.5) / 2)
        if m * (m - 1) // 2 == lc and m >= 3:
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

def _parabolic_config():
    """
    Generate a configuration based on a parabola y = x^2 with perturbations.
    Points on a parabola have the property that any 3 determine a unique line,
    so we can create many collinear triples while avoiding 4-collinear.
    """
    random.seed(42)
    points = []
    # Use x values that give distinct y on parabola, then perturb
    x_vals = list(range(-5, 6))
    random.shuffle(x_vals)
    x_vals = x_vals[:20]
    
    for x in x_vals:
        y = x * x
        # Add small perturbation to avoid exact parabola (which would have 4-collinear issues)
        px = random.randint(-2, 2)
        py = random.randint(-3, 3)
        points.append([x + px, y + py])
    
    # Ensure distinctness and bounds
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
    
    # If we lost some, add random points
    while len(final_points) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            final_points.append([x, y])
    
    return final_points

def _greedy_triple_config():
    """
    Greedily construct points to maximize collinear triples while maintaining
    no-four-collinear. Place points on a few lines with exactly 3 points each.
    """
    random.seed(99)
    points = []
    seen = set()
    
    # Strategy: create several lines, each with exactly 3 points
    # Pick random lines and place 3 points on each
    used_lines = set()
    attempts = 0
    while len(points) < 20 and attempts < 500:
        attempts += 1
        # Pick a random line
        x1 = random.randint(-40, 40)
        y1 = random.randint(-40, 40)
        x2 = random.randint(-40, 40)
        y2 = random.randint(-40, 40)
        if (x1, y1) == (x2, y2):
            continue
        
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            continue
        dx_n = dx // g
        dy_n = dy // g
        
        # Generate candidate points on this line
        candidates = []
        for t in range(-10, 11):
            px = x1 + t * dx_n
            py = y1 + t * dy_n
            if -100 <= px <= 100 and -100 <= py <= 100:
                key = (px, py)
                if key not in seen:
                    candidates.append(key)
        
        if len(candidates) < 3:
            continue
        
        # Check if adding 3 from this line would create a 4-collinear issue
        # Count how many existing points are on this line
        line_count = 0
        for p in points:
            if dx_n * (p[1] - y1) == dy_n * (p[0] - x1):
                line_count += 1
        
        if line_count >= 3:
            continue  # Already 3 on this line, adding more would make 4+
        
        # Pick 3 candidates (or fewer if we don't need all)
        need = min(3, 20 - len(points))
        random.shuffle(candidates)
        for cand in candidates[:need]:
            if len(points) >= 20:
                break
            # Verify no 4-collinear with this addition
            test_points = points + [list(cand)]
            if _check_no_four_collinear(test_points):
                points.append(list(cand))
                seen.add(cand)
    
    # Fill remaining with random points
    while len(points) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            test_points = points + [[x, y]]
            if _check_no_four_collinear(test_points):
                points.append([x, y])
                seen.add((x, y))
    
    return points

def _grid_config():
    """Generate a grid-based configuration with perturbations."""
    random.seed(123)
    # Start with a 5x5 grid, pick 20 points
    candidates = []
    for x in range(-4, 5):
        for y in range(-4, 5):
            candidates.append([x, y])
    random.shuffle(candidates)
    points = [list(p) for p in candidates[:20]]
    
    # Perturb to fix 4-collinear
    for _ in range(100):
        if _check_no_four_collinear(points):
            break
        # Find a 4-collinear set
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
    Uses incremental evaluation for speed.
    """
    import math
    
    current_triples = _count_triples(points)
    best_triples = current_triples
    best_points = [list(p) for p in points]
    start_time = time.time()
    temp = 2.0
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time:
        if _elapsed() > _DEADLINE - 10:
            break
        if no_improve > max_no_improve:
            break
        
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        
        # Try several candidate moves
        best_move = None
        best_delta = 0
        
        for attempt in range(30):
            # Mix of random and local moves
            if attempt < 15:
                # Small local move
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
                new_x = old_x + dx
                new_y = old_y + dy
            elif attempt < 25:
                # Medium move
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_x = old_x + dx
                new_y = old_y + dy
            else:
                # Larger random move
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
            
            if new_x == old_x and new_y == old_y:
                continue
            if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                continue
            
            # Check distinctness
            if (new_x, new_y) in [(p[0], p[1]) for i, p in enumerate(points) if i != idx]:
                continue
            
            new_points = [list(p) for p in points]
            new_points[idx] = [new_x, new_y]
            
            if not _check_no_four_collinear(new_points):
                continue
            
            new_triples = _count_triples(new_points)
            delta = new_triples - current_triples
            
            # Accept if better, or with probability exp(-delta/temp) if worse
            if delta > 0:
                best_move = new_points
                best_delta = delta
                break
            else:
                prob = math.exp(max(delta * 0.3, -10) / temp)
                if random.random() < prob:
                    best_move = new_points
                    best_delta = delta
        
        if best_move:
            points = best_move
            current_triples += best_delta
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = [list(p) for p in points]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Cool down
        temp = max(0.1, temp * 0.98)
    
    return best_points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 21
    
    # Generate multiple candidate configurations
    candidates = []
    
    # Greedy triple configuration (prioritize this)
    candidates.append(_greedy_triple_config())
    
    # Parabolic configuration
    candidates.append(_parabolic_config())
    
    # Grid configuration
    candidates.append(_grid_config())
    
    # Random configurations
    for _ in range(3):
        if _elapsed() > _DEADLINE - 40:
            break
        candidates.append(_random_valid_config())
    
    # Evaluate and improve each candidate
    for config in candidates:
        if _elapsed() > _DEADLINE - 20:
            break
        
        # Quick validity check
        if not _check_no_four_collinear(config):
            continue
        
        # Improve using local search
        improved = _improve_triples(config, max_time=18.0)
        
        if _elapsed() > _DEADLINE - 8:
            break
        
        # Score
        s = _score(improved)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in improved]
    
    # If we still have time, do more improvement on the best
    if best_points and _elapsed() < _DEADLINE - 15:
        improved = _improve_triples(best_points, max_time=20.0)
        if _elapsed() < _DEADLINE - 5:
            s = _score(improved)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in improved]
    
    # Final fallback: return a simple valid config if needed
    if best_points is None:
        best_points = _random_valid_config()
    
    return best_points
