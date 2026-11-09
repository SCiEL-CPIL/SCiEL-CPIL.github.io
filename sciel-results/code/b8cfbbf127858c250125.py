import math
import time
import random

def _hexdist(q1, r1, q2, r2):
    """Calculate hex distance between two axial coordinates."""
    return max(abs(q1 - q2), abs(r1 - r2), abs((q1 + r1) - (q2 + r2)))

def _count_edges(points):
    """Count the number of unit-distance pairs in the set of points."""
    n = len(points)
    count = 0
    for i in range(n):
        q1, r1 = points[i]
        for j in range(i + 1, n):
            q2, r2 = points[j]
            if _hexdist(q1, r1, q2, r2) == 1:
                count += 1
    return count

def _neighbor_count_map(points):
    """Build a map of point -> count of neighbors at distance 1."""
    point_set = set(points)
    nc = {}
    for p in points:
        nc[p] = 0
    for p in points:
        q, r = p
        for dq, dr in [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]:
            n = (q+dq, r+dr)
            if n in point_set:
                nc[p] += 1
    return nc

def _delta_edges(points, old_pt, new_pt):
    """
    Compute the change in edge count when replacing old_pt with new_pt.
    Returns delta = new_edges - old_edges.
    """
    point_set = set(points)
    point_set.discard(old_pt)
    point_set.add(new_pt)
    
    # Edges lost: neighbors of old_pt that are still in the set
    lost = 0
    q1, r1 = old_pt
    for dq, dr in [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]:
        n = (q1+dq, r1+dr)
        if n in point_set:
            lost += 1
    
    # Edges gained: neighbors of new_pt that are in the set (excluding old_pt which is removed)
    gained = 0
    q2, r2 = new_pt
    for dq, dr in [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]:
        n = (q2+dq, r2+dr)
        if n in point_set:
            gained += 1
    
    return gained - lost

def _generate_optimal_32():
    """
    Generate the optimal configuration of 32 points.
    Hexagon of radius 2 (19 points) plus best 13 contiguous points from ring 3.
    """
    base_points = set()
    for q in range(-2, 3):
        for r in range(-2, 3):
            if _hexdist(0, 0, q, r) <= 2:
                base_points.add((q, r))
                
    # Get ring 3 points sorted by angle
    ring3 = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if _hexdist(0, 0, q, r) == 3:
                x = q + 0.5 * r
                y = math.sqrt(3) / 2 * r
                angle = math.atan2(y, x)
                ring3.append((angle, q, r))
    ring3.sort()
    
    n_ring = len(ring3)
    # Try all contiguous segments of 13 from ring3 (circular)
    best_score = -1
    best_segment = None
    
    for start in range(n_ring):
        segment = set()
        for i in range(13):
            idx = (start + i) % n_ring
            _, q, r = ring3[idx]
            segment.add((q, r))
        
        all_pts = list(base_points) + list(segment)
        score = _count_edges(all_pts)
        if score > best_score:
            best_score = score
            best_segment = segment
    
    points = base_points | best_segment
    return list(points)

def _simulated_annealing(points, start_time, deadline):
    """
    Simulated annealing with O(1) edge-delta evaluation.
    """
    current_points = [tuple(p) for p in points]
    current_set = set(current_points)
    current_score = _count_edges(current_points)
    
    best_points = list(current_points)
    best_score = current_score
    
    directions = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Temperature schedule
    t0 = 2.0
    t_min = 0.01
    # We'll run until deadline
    n = len(current_points)
    
    while time.time() < deadline:
        # Pick a random point and a random direction
        i = random.randint(0, n - 1)
        q, r = current_points[i]
        dq, dr = directions[random.randint(0, 5)]
        new_q, new_r = q + dq, r + dr
        
        if abs(new_q) > 64 or abs(new_r) > 64:
            continue
        if (new_q, new_r) in current_set:
            continue
        
        # Compute delta in O(1)
        old_pt = (q, r)
        new_pt = (new_q, new_r)
        delta = _delta_edges(current_points, old_pt, new_pt)
        
        # Temperature based on remaining time
        elapsed = time.time() - start_time
        total_time = deadline - start_time
        progress = elapsed / total_time if total_time > 0 else 1.0
        temp = t0 * (t_min / t0) ** progress
        
        # Accept if better or with probability exp(delta/temp)
        if delta >= 0:
            accept = True
        else:
            accept = random.random() < math.exp(delta / temp)
        
        if accept:
            current_points[i] = new_pt
            current_set.discard(old_pt)
            current_set.add(new_pt)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds safety margin (evaluator kills at 150s)
    
    # 1. Generate the known optimal candidate
    candidate = _generate_optimal_32()
    
    # Ensure exactly 32 distinct points
    if len(candidate) != 32:
        # Fallback to spiral generation
        candidates = []
        for q in range(-10, 11):
            for r in range(-10, 11):
                dist = max(abs(q), abs(r), abs(q + r))
                x = q + 0.5 * r
                y = math.sqrt(3) / 2 * r
                angle = math.atan2(y, x)
                candidates.append((dist, angle, q, r))
        candidates.sort(key=lambda x: (x[0], x[1]))
        candidate = [c[2:] for c in candidates[:32]]
        
    # 2. Verify the score
    score = _count_edges(candidate)
    
    # 3. If score is optimal (76), return immediately
    if score == 76:
        return [[q, r] for q, r in candidate]
        
    # 4. If not optimal, try simulated annealing with O(1) deltas
    if time.time() < deadline:
        optimized_points, opt_score = _simulated_annealing(candidate, start_time, deadline)
        if opt_score > score:
            candidate = optimized_points
            score = opt_score
            
    # 5. Return the best found
    return [[q, r] for q, r in candidate]
