import time
import random
from math import gcd
from collections import defaultdict
import sys

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

def _get_triples(points):
    """Get all collinear triples as a list of (i, j, k) tuples."""
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

def _build_conflict_masks(triples, n):
    """Build bitmask conflict sets for each point."""
    masks = [0] * n
    for (a, b, c) in triples:
        masks[a] |= (1 << b) | (1 << c)
        masks[b] |= (1 << a) | (1 << c)
        masks[c] |= (1 << a) | (1 << b)
    return masks

def _max_independent_set_bitmask(conflict_masks, n, deadline=130.0):
    """
    Find maximum independent set using bitmask B&B.
    conflict_masks[i] = bitmask of points that conflict with i (i.e., form a triple with i)
    """
    best = [0]
    full_mask = (1 << n) - 1
    
    # Order vertices by degree (conflict count)
    degrees = [bin(conflict_masks[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Reorder conflict masks
    remap = [0] * n
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    
    new_masks = [0] * n
    for i in range(n):
        for j in range(n):
            if conflict_masks[i] & (1 << j):
                new_masks[remap[i]] |= (1 << remap[j])
    
    def backtrack(remaining, count):
        if time.time() > deadline:
            return
        if not remaining:
            if count > best[0]:
                best[0] = count
            return
        
        # Upper bound: count + popcount(remaining)
        remaining_count = bin(remaining).count('1')
        if count + remaining_count <= best[0]:
            return
        
        # Pick vertex with max degree in remaining subgraph
        # Find the vertex in remaining with max conflicts within remaining
        max_deg = -1
        chosen = -1
        r = remaining
        while r:
            lsb = r & (-r)
            v = lsb.bit_length() - 1
            deg = bin(new_masks[v] & remaining).count('1')
            if deg > max_deg:
                max_deg = deg
                chosen = v
            r ^= lsb
        
        if chosen == -1:
            if count + remaining_count > best[0]:
                best[0] = count + remaining_count
            return
        
        # Branch 1: Include chosen
        new_remaining = remaining & ~(1 << chosen) & ~new_masks[chosen]
        backtrack(new_remaining, count + 1)
        
        # Branch 2: Exclude chosen
        new_remaining = remaining & ~(1 << chosen)
        backtrack(new_remaining, count)
    
    backtrack(full_mask, 0)
    return best[0]

def _score(points, deadline=None):
    """Compute the score: largest subset with no three collinear."""
    if deadline is None:
        deadline = _DEADLINE - 5
    
    triples = _get_triples(points)
    if not triples:
        return len(points)
    
    n = len(points)
    conflict_masks = _build_conflict_masks(triples, n)
    return _max_independent_set_bitmask(conflict_masks, n, deadline)

def _parabolic_config():
    """Generate a parabolic configuration that tends to have fewer collinear triples."""
    points = []
    # Use points on or near a parabola y = x^2 / k
    # This naturally avoids many collinearities
    used = set()
    for x in range(-10, 11):
        y = x * x // 4
        if (x, y) not in used:
            used.add((x, y))
            points.append([x, y])
    
    # If we need more, add perturbed points
    while len(points) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in used:
            used.add((x, y))
            points.append([x, y])
    
    # Perturb to ensure no four collinear
    for attempt in range(200):
        if _check_no_four_collinear(points):
            break
        # Find a 4-collinear set and move one point
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
                    for _ in range(10):
                        new_x = old_x + random.randint(-5, 5)
                        new_y = old_y + random.randint(-5, 5)
                        if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                            if (new_x, new_y) not in [(p[0], p[1]) for k, p in enumerate(points) if k != idx]:
                                points[idx] = [new_x, new_y]
                                found = True
                                break
                    if found:
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
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            key = (x, y)
            if key not in seen:
                seen.add(key)
                points.append([x, y])
        if _check_no_four_collinear(points):
            return points

def _local_search(points, max_time=10.0, deadline=None):
    """
    Local search to minimize the score (largest no-three-collinear subset).
    We want to maximize collinear triples to reduce the independent set size.
    """
    if deadline is None:
        deadline = _DEADLINE - 10
    
    start_time = time.time()
    current_points = [list(p) for p in points]
    current_score = _score(current_points, deadline - 2)
    best_points = [list(p) for p in current_points]
    best_score = current_score
    
    no_improve = 0
    max_no_improve = 30
    
    while time.time() - start_time < max_time and time.time() < deadline:
        if no_improve > max_no_improve:
            break
            
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        
        moved = False
        for _ in range(30):
            dx = random.randint(-8, 8)
            dy = random.randint(-8, 8)
            new_x = old_x + dx
            new_y = old_y + dy
            
            if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                continue
            if (new_x, new_y) in [(p[0], p[1]) for i, p in enumerate(current_points) if i != idx]:
                continue
                
            new_points = [list(p) for p in current_points]
            new_points[idx] = [new_x, new_y]
            
            if not _check_no_four_collinear(new_points):
                continue
                
            new_score = _score(new_points, deadline - 2)
            
            if new_score < current_score:
                current_points = new_points
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
                    no_improve = 0
                else:
                    no_improve += 1
                moved = True
                break
            else:
                no_improve += 1
        
        if not moved:
            no_improve += 1
            
    return best_points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 21
    
    candidates = []
    
    # Parabolic config (should have good structure)
    candidates.append(_parabolic_config())
    
    # Random configs
    for _ in range(8):
        if _elapsed() > _DEADLINE - 60:
            break
        candidates.append(_random_valid_config())
    
    # Improve each candidate
    for config in candidates:
        if _elapsed() > _DEADLINE - 20:
            break
        
        if not _check_no_four_collinear(config):
            continue
            
        improved = _local_search(config, max_time=12.0, deadline=_DEADLINE - 10)
        
        if _elapsed() > _DEADLINE - 10:
            break
            
        s = _score(improved, _DEADLINE - 5)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in improved]
            
    # Final polish on the best
    if best_points and _elapsed() < _DEADLINE - 15:
        improved = _local_search(best_points, max_time=15.0, deadline=_DEADLINE - 5)
        if _elapsed() < _DEADLINE - 5:
            s = _score(improved, _DEADLINE - 3)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in improved]
                
    if best_points is None:
        best_points = _random_valid_config()
        
    return best_points
