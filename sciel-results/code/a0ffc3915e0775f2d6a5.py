"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

def _elapsed(start):
    return time.time() - start

def _line_key(p1, p2):
    """Return a normalized line key for two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
    if g == 0:
        g = 1
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: dx*(y - y1) = dy*(x - x1) => dy*x - dx*y = dy*x1 - dx*y1
    c = dy * p1[0] - dx * p1[1]
    return (dx, dy, c)

def _get_lines(points):
    """Get all lines with their point indices."""
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    lines = _get_lines(points)
    for pts in lines.values():
        if len(pts) >= 4:
            return True
    return False

def _count_triples(points):
    """Count collinear triples."""
    lines = _get_lines(points)
    count = 0
    for pts in lines.values():
        if len(pts) >= 3:
            count += len(pts) * (len(pts) - 1) * (len(pts) - 2) // 6
    return count

def _max_3free_subset(points, deadline=None):
    """Find largest subset with no three collinear using branch and bound."""
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute collinear pairs for each point
    lines = _get_lines(points)
    collinear_pairs = [set() for _ in range(n)]
    for pts in lines.values():
        if len(pts) >= 3:
            pts_list = list(pts)
            for i in range(len(pts_list)):
                for j in range(i + 1, len(pts_list)):
                    for k in range(j + 1, len(pts_list)):
                        a, b, c = pts_list[i], pts_list[j], pts_list[k]
                        collinear_pairs[a].add(frozenset((b, c)))
                        collinear_pairs[b].add(frozenset((a, c)))
                        collinear_pairs[c].add(frozenset((a, b)))
    
    best = [0]
    
    def can_add(p, S):
        for pair in collinear_pairs[p]:
            if pair.issubset(S):
                return False
        return True
    
    def bnb(S, R, depth):
        if deadline is not None and time.time() > deadline:
            return
        if len(S) + len(R) <= best[0]:
            return
        if not R:
            if len(S) > best[0]:
                best[0] = len(S)
            return
        
        p = min(R)
        
        if can_add(p, S):
            new_S = S | frozenset([p])
            new_R = R - frozenset([p])
            bnb(new_S, new_R, depth + 1)
        
        new_R = R - frozenset([p])
        bnb(S, new_R, depth + 1)
    
    bnb(frozenset(), frozenset(range(n)), 0)
    return best[0]

def _validate(points):
    """Validate that points are distinct, in bounds, and no four collinear."""
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return not _has_four_collinear(points)

def _initial_grid():
    """Create an initial configuration on a grid-like pattern."""
    points = []
    # Use a 5x4 grid with some perturbation
    for i in range(5):
        for j in range(4):
            x = -40 + i * 20 + random.randint(-3, 3)
            y = -30 + j * 20 + random.randint(-3, 3)
            x = max(-100, min(100, x))
            y = max(-100, min(100, y))
            if (x, y) not in {(p[0], p[1]) for p in points}:
                points.append([x, y])
    # Fill remaining if needed
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in {(p[0], p[1]) for p in points}:
            points.append([x, y])
    return points[:20]

def _improve(points, deadline):
    """Simulated annealing to maximize collinear triples while avoiding four-collinear."""
    start_time = time.time()
    current = points[:]
    current_triples = _count_triples(current)
    best = points[:]
    best_triples = current_triples
    best_score = _max_3free_subset(current, deadline)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current[idx][:]
        
        # Try a random new position
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        new_p = [x, y]
        
        # Check if new position is valid (not same as another point)
        valid = True
        for i, p in enumerate(current):
            if i != idx and p[0] == x and p[1] == y:
                valid = False
                break
        
        if not valid:
            T *= cooling
            continue
        
        # Apply move
        current[idx] = new_p
        
        # Check four-collinear
        if _has_four_collinear(current):
            # Revert
            current[idx] = old_p
            T *= cooling
            continue
        
        new_triples = _count_triples(current)
        delta_triples = new_triples - current_triples
        
        # Accept if better, or with probability exp(delta/T)
        if delta_triples >= 0:
            current_triples = new_triples
            if new_triples > best_triples:
                best_triples = new_triples
                best = current[:]
                # Check score periodically
                if time.time() < deadline - 10:
                    score = _max_3free_subset(current, deadline)
                    if score < best_score:
                        best_score = score
                        best = current[:]
        else:
            prob = math.exp(delta_triples / T)
            if random.random() < prob:
                current_triples = new_triples
            else:
                current[idx] = old_p
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with grid-based initial configuration
    points = _initial_grid()
    best_points, best_score = _improve(points, deadline)
    
    # Try a few more random restarts if time permits
    while time.time() < deadline - 30:
        # Generate random valid configuration
        rand_points = []
        attempts = 0
        while len(rand_points) < 20 and attempts < 2000:
            attempts += 1
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if (x, y) not in {(pt[0], pt[1]) for pt in rand_points}:
                test = rand_points + [p]
                if not _has_four_collinear(test):
                    rand_points.append(p)
        
        if len(rand_points) == 20:
            candidate_points, candidate_score = _improve(rand_points, min(deadline, time.time() + 20))
            if candidate_score < best_score:
                best_score = candidate_score
                best_points = candidate_points[:]
    
    return best_points
# EVOLVE-BLOCK-END
