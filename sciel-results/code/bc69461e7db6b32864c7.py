import random
import time
from itertools import combinations

N = 64
DEADLINE = 320  # safety margin before 330s

def _squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _is_valid_point(candidate, points):
    """Check if candidate can be added to points without forming isosceles triangle."""
    n = len(points)
    cand_dists = [_squared_dist(candidate, p) for p in points]
    
    # Check: for each pair (i, j), candidate doesn't create isosceles
    # dist(cand, i) == dist(cand, j) -> isosceles with apex at cand
    # dist(cand, i) == dist(i, j) -> isosceles with apex at i or j
    # dist(cand, j) == dist(i, j) -> isosceles with apex at i or j
    
    for i in range(n):
        for j in range(i + 1, n):
            dij2 = _squared_dist(points[i], points[j])
            if cand_dists[i] == cand_dists[j]:
                return False
            if cand_dists[i] == dij2:
                return False
            if cand_dists[j] == dij2:
                return False
    return True

def _greedy_with_random_order(points, random_order, deadline):
    """Try to add points in the given random order, return best found."""
    current = list(points)
    used_positions = set(points)
    
    for pos in random_order:
        if time.time() > deadline:
            break
        if pos in used_positions:
            continue
        if _is_valid_point(pos, current):
            current.append(pos)
            used_positions.add(pos)
    
    return current

def _generate_random_order():
    """Generate a random ordering of all grid positions."""
    positions = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(positions)
    return positions

def _greedy_with_priority(deadline):
    """Greedy approach with priority based on distance to existing points."""
    current = []
    used_positions = set()
    
    # Start with a random point
    start = (random.randint(0, N-1), random.randint(0, N-1))
    current.append(start)
    used_positions.add(start)
    
    while time.time() < deadline:
        # Find the best candidate: one that maximizes minimum distance to existing points
        best_pos = None
        best_min_dist = -1
        
        # Sample a subset of candidates for speed
        candidates = []
        for _ in range(500):
            x = random.randint(0, N-1)
            y = random.randint(0, N-1)
            pos = (x, y)
            if pos not in used_positions:
                candidates.append(pos)
        
        for pos in candidates:
            if time.time() > deadline:
                break
            min_dist = float('inf')
            for p in current:
                d = _squared_dist(pos, p)
                if d < min_dist:
                    min_dist = d
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos
        
        if best_pos is None or not _is_valid_point(best_pos, current):
            # Try a few more random candidates
            found = False
            for _ in range(200):
                x = random.randint(0, N-1)
                y = random.randint(0, N-1)
                pos = (x, y)
                if pos not in used_positions and _is_valid_point(pos, current):
                    current.append(pos)
                    used_positions.add(pos)
                    found = True
                    break
            if not found:
                break
        else:
            current.append(best_pos)
            used_positions.add(best_pos)
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_points = []
    
    # Multiple restarts with different strategies
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Strategy 1: Random order greedy
        random_order = _generate_random_order()
        result = _greedy_with_random_order([], random_order, deadline)
        if len(result) > len(best_points):
            best_points = result
        
        # Strategy 2: Priority-based greedy
        if time.time() > deadline - 5:
            break
        result = _greedy_with_priority(deadline)
        if len(result) > len(best_points):
            best_points = result
    
    return best_points
