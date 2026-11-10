import time
import random
import sys
from itertools import combinations
from math import gcd

def are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def count_collinear_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def has_four_collinear(points):
    n = len(points)
    # For each pair, count points on the line
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if are_collinear(p1, p2, points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

def _build_collinear_triples(points):
    """Build a list of collinear triples and pair_to_third mapping."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    
    # pair_to_third[i][j] = set of k such that {i, j, k} is collinear
    pair_to_third = {}
    for i in range(n):
        pair_to_third[i] = {}
        for j in range(n):
            if i == j:
                continue
            pair_to_third[i][j] = set()
    
    for i, j, k in triples:
        pair_to_third[i][j].add(k)
        pair_to_third[i][k].add(j)
        pair_to_third[j][i].add(k)
        pair_to_third[j][k].add(i)
        pair_to_third[k][i].add(j)
        pair_to_third[k][j].add(i)
    
    return triples, pair_to_third

def _max_subset_no_three_collinear(points):
    """
    Compute the size of the largest subset of points with no three collinear.
    Uses Branch and Bound with bitmasks.
    """
    n = len(points)
    if n <= 2:
        return n
    
    triples, pair_to_third = _build_collinear_triples(points)
    
    if not triples:
        return n
    
    best_size = 0
    full_mask = (1 << n) - 1
    
    # Precompute conflict masks for speed
    # conflict_with[i] = bitmask of points j such that i and j are in some triple
    # Actually we need: for each point p, and each already-selected r, 
    # the set of points forbidden when adding p
    
    def popcount(mask):
        return bin(mask).count('1')
    
    def solve(current_size, current_mask, remaining_mask):
        nonlocal best_size
        
        if not remaining_mask:
            if current_size > best_size:
                best_size = current_size
            return
        
        # Pruning
        if current_size + popcount(remaining_mask) <= best_size:
            return
        
        # Pick a point from remaining
        # Heuristic: pick the point that conflicts with the most remaining points
        # This tends to prune more effectively
        best_p = -1
        max_conflicts = -1
        
        rm = remaining_mask
        while rm:
            lsb = rm & -rm
            i = lsb.bit_length() - 1
            # Count conflicts: how many points in remaining would be forbidden if we pick i
            # A point q is forbidden if it forms a triple with i and some r in current_mask
            # Or if it forms a triple with i and some other point we haven't picked yet?
            # No, the conflict is: if we pick i, then for each r in current_mask,
            # any q in pair_to_third[i][r] is forbidden.
            # Also, if we pick i, we can't pick any q such that there exists r in current_mask
            # where {i, r, q} is collinear.
            
            # For heuristic, let's count how many points in remaining would be eliminated
            forbidden = 0
            cm = current_mask
            while cm:
                lsb2 = cm & -cm
                r = lsb2.bit_length() - 1
                for q in pair_to_third[i][r]:
                    if remaining_mask & (1 << q):
                        forbidden |= (1 << q)
                cm ^= lsb2
            
            conflicts = popcount(forbidden)
            if conflicts > max_conflicts:
                max_conflicts = conflicts
                best_p = i
            
            rm ^= lsb
        
        p = best_p
        if p == -1:
            return
        
        # Option 1: Exclude p
        solve(current_size, current_mask, remaining_mask & ~(1 << p))
        
        # Option 2: Include p
        # Check compatibility
        compatible = True
        cm = current_mask
        while cm:
            lsb = cm & -cm
            r = lsb.bit_length() - 1
            if pair_to_third[p][r] & current_mask:
                compatible = False
                break
            cm ^= lsb
        
        if compatible:
            new_current_mask = current_mask | (1 << p)
            new_remaining = remaining_mask & ~(1 << p)
            
            # Remove forbidden points
            forbidden_mask = 0
            ncm = new_current_mask
            while ncm:
                lsb = ncm & -ncm
                r = lsb.bit_length() - 1
                for q in pair_to_third[p][r]:
                    if new_remaining & (1 << q):
                        forbidden_mask |= (1 << q)
                ncm ^= lsb
            
            new_remaining &= ~forbidden_mask
            solve(current_size + 1, new_current_mask, new_remaining)
    
    solve(0, 0, full_mask)
    return best_size

def _score(points):
    """Compute the score for a set of points."""
    if not has_four_collinear(points):
        return _max_subset_no_three_collinear(points)
    return 20  # Invalid, return max possible

def _check_valid(points):
    """Check if points form a valid configuration."""
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
    if has_four_collinear(points):
        return False
    return True

def _greedy_construct():
    """
    Greedily construct 20 points that maximize collinear triples.
    Start with a cluster and add points that create the most new collinear triples.
    """
    # Start with a small grid cluster
    random.seed(42)  # Deterministic for reproducibility in this call
    
    # Try multiple starting configurations
    best_points = None
    best_triples = -1
    
    for attempt in range(50):
        # Random starting point
        ox = random.randint(-80, 80)
        oy = random.randint(-80, 80)
        
        # Start with a few points forming a triangle or line
        points = []
        seen = set()
        
        # Add 3-5 points in a tight cluster
        for _ in range(random.randint(3, 5)):
            x = ox + random.randint(-5, 5)
            y = oy + random.randint(-5, 5)
            if (x, y) not in seen and -100 <= x <= 100 and -100 <= y <= 100:
                seen.add((x, y))
                points.append((x, y))
        
        while len(points) < 20:
            # Find the point that maximizes new collinear triples
            best_new_p = None
            best_new_triples = -1
            
            # Try candidate points
            candidates = set()
            # Points near existing points
            for p in points:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        candidates.add((p[0] + dx, p[1] + dy))
            # Also try some random points
            for _ in range(200):
                candidates.add((random.randint(-100, 100), random.randint(-100, 100)))
            
            for cand in candidates:
                if cand in seen:
                    continue
                if not (-100 <= cand[0] <= 100 and -100 <= cand[1] <= 100):
                    continue
                
                # Check if adding cand creates 4-collinear
                temp_points = points + [cand]
                if has_four_collinear(temp_points):
                    continue
                
                # Count new triples
                new_triples = 0
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        if are_collinear(points[i], points[j], cand):
                            new_triples += 1
                
                if new_triples > best_new_triples:
                    best_new_triples = new_triples
                    best_new_p = cand
            
            if best_new_p is None:
                # No valid point found, try random
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in seen:
                        best_new_p = (x, y)
                        break
            
            points.append(best_new_p)
            seen.add(best_new_p)
        
        # Final check
        if not has_four_collinear(points):
            tc = count_collinear_triples(points)
            if tc > best_triples:
                best_triples = tc
                best_points = points.copy()
    
    return best_points

def _local_search(points, time_limit=10.0):
    """
    Local search to maximize collinear triples.
    """
    if points is None:
        return None
    
    start = time.time()
    current_points = points.copy()
    current_tc = count_collinear_triples(current_points)
    
    best_points = current_points.copy()
    best_tc = current_tc
    
    while time.time() - start < time_limit:
        improved = False
        for idx in range(20):
            # Try moving point idx to a better location
            best_new_p = None
            best_new_tc = current_tc
            
            # Try nearby positions
            old_p = current_points[idx]
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    new_p = (old_p[0] + dx, old_p[1] + dy)
                    if new_p == old_p:
                        continue
                    if new_p in current_points:
                        continue
                    if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                        continue
                    
                    candidate = current_points.copy()
                    candidate[idx] = new_p
                    
                    if has_four_collinear(candidate):
                        continue
                    
                    new_tc = count_collinear_triples(candidate)
                    if new_tc > best_new_tc:
                        best_new_tc = new_tc
                        best_new_p = new_p
            
            if best_new_p is not None and best_new_tc > current_tc:
                current_points[idx] = best_new_p
                current_tc = best_new_tc
                improved = True
                
                if current_tc > best_tc:
                    best_tc = current_tc
                    best_points = current_points.copy()
        
        if not improved:
            break
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Step 1: Greedy construction
    points = _greedy_construct()
    
    if points is None:
        # Fallback to random
        points = []
        seen = set()
        while len(points) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in seen:
                seen.add(p)
                points.append(p)
    
    # Ensure valid
    if not _check_valid(points):
        # Fix by random restarts
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            points = []
            seen = set()
            while len(points) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in seen:
                    seen.add(p)
                    points.append(p)
            if not has_four_collinear(points):
                break
    
    # Step 2: Local search to maximize triples
    time_left = deadline - time.time() - 30.0  # Reserve 30s for scoring
    if time_left > 5:
        improved_points = _local_search(points, time_left)
        if improved_points is not None:
            points = improved_points
    
    # Step 3: Compute final score
    if time.time() > deadline - 25:
        # Not enough time, return what we have
        if not _check_valid(points):
            # Emergency fix
            for i in range(20):
                if has_four_collinear(points):
                    while True:
                        p = (random.randint(-100, 100), random.randint(-100, 100))
                        if p not in points:
                            points[i] = p
                            break
        return [list(p) for p in points]
    
    score = _score(points)
    
    # If score is poor, try a few more local search iterations
    if score > 12 and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            # Perturb and local search
            perturbed = points.copy()
            for i in range(random.randint(1, 5)):
                idx = random.randint(0, 19)
                while True:
                    p = (random.randint(-100, 100), random.randint(-100, 100))
                    if p not in perturbed:
                        perturbed[idx] = p
                        break
            if not has_four_collinear(perturbed):
                ls_result = _local_search(perturbed, 5.0)
                if ls_result is not None:
                    new_score = _score(ls_result)
                    if new_score < score:
                        points = ls_result
                        score = new_score
    
    # Final validation
    if not _check_valid(points):
        # Last resort: simple valid set
        points = [(i, i * i) for i in range(20)]
        # Check no 4 collinear - parabola y=x^2 has no 3 collinear except trivial
        # Actually we need to verify
        if has_four_collinear(points):
            # Fallback to random valid
            for _ in range(1000):
                pts = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(20)]
                if len(set(pts)) == 20 and not has_four_collinear(pts):
                    points = pts
                    break
    
    return [list(p) for p in points]
