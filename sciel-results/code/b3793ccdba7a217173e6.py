import time
import random
import sys

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
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

def _max_independent_set(points):
    """
    Fast exact branch and bound for the largest subset with no three collinear.
    Uses precomputed collinear triples and bitmask-based pruning.
    """
    n = len(points)
    
    # Precompute collinear triples: for each pair (i,j), which k are collinear
    # Also precompute for each point i, a bitmask of points that form a collinear triple
    # with i and some other point? No, that's not directly useful.
    
    # Better: Precompute all collinear triples as a list of (i,j,k) with i<j<k.
    # Then, a subset S is valid if it contains no such triple.
    
    # For fast checking during backtracking:
    # When we have a current set S (as a bitmask), and we want to add point p,
    # we need to check if there exist i, j in S such that (i, j, p) is a collinear triple.
    # Equivalently, for each i in S, is there a j in S such that (i, j, p) is collinear?
    # Or: for each pair (i, p) where i in S, is there a j in S such that (i, j, p) is collinear?
    
    # Let's precompute for each point p, a list of pairs (i, j) such that i, j, p are collinear.
    # Then when adding p to S, we check if any (i, j) in this list has both i and j in S.
    
    collinear_pairs_for_p = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    # i, j, k are collinear
                    collinear_pairs_for_p[k].add((i, j))
                    collinear_pairs_for_p[i].add((j, k))
                    collinear_pairs_for_p[j].add((i, k))

    best = 0
    
    def backtrack(current_mask, candidates_mask):
        nonlocal best
        # Pruning
        if bin(current_mask).count('1') + bin(candidates_mask).count('1') <= best:
            return
            
        if candidates_mask == 0:
            if bin(current_mask).count('1') > best:
                best = bin(current_mask).count('1')
            return

        # Pick a candidate
        # Heuristic: pick the one with the most conflicts (i.e., most collinear pairs with other candidates)
        # Or just pick the lowest set bit for speed.
        
        # For speed, let's pick the lowest set bit.
        # p_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        # But we need to iterate. Let's just pick the first one.
        
        # Actually, for better pruning, we should pick a point that conflicts with many others.
        # But computing that is expensive. Let's stick with first for now, or use a simple heuristic.
        
        # Let's pick the point with the highest "degree" in the conflict hypergraph among candidates.
        # Degree of p = number of pairs (i, j) in collinear_pairs_for_p[p] such that i and j are in candidates.
        # This is expensive to compute. Let's just pick the first candidate for simplicity and rely on pruning.
        
        p_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        p_bit = 1 << p_idx
        rest_candidates = candidates_mask & ~p_bit
        
        # Branch 1: Include p_idx
        # Check if p_idx can be added: for each (i, j) in collinear_pairs_for_p[p_idx], 
        # if both i and j are in current_mask, then it's invalid.
        valid = True
        for i, j in collinear_pairs_for_p[p_idx]:
            if (current_mask & (1 << i)) and (current_mask & (1 << j)):
                valid = False
                break
        
        if valid:
            new_mask = current_mask | p_bit
            # New candidates: remove p_idx and any point that would form a collinear triple with p_idx and some point in new_mask?
            # No, the constraint is global. When we add p_idx, we don't remove other candidates.
            # We just proceed with the same candidates minus p_idx.
            # The validity check for future additions will handle the rest.
            backtrack(new_mask, rest_candidates)
        
        # Branch 2: Exclude p_idx
        backtrack(current_mask, rest_candidates)

    backtrack(0, (1 << n) - 1)
    return best

def _generate_structured_config():
    """
    Generate a configuration with 5 groups of 4 points.
    Each group: 3 points on a line, 1 point perturbed.
    This ensures at least 5 collinear triples.
    """
    points = []
    used = set()
    
    # Define 5 different lines
    lines = [
        [(0, 0), (1, 0), (2, 0)],      # Horizontal
        [(0, 1), (1, 1), (2, 1)],      # Horizontal
        [(0, 2), (1, 2), (2, 2)],      # Horizontal
        [(0, 3), (1, 3), (2, 3)],      # Horizontal
        [(0, 4), (1, 4), (2, 4)],      # Horizontal
    ]
    
    # Perturb the 4th point in each group
    perturbations = [
        (0, 5),
        (1, 5),
        (2, 5),
        (0, 6),
        (1, 6),
    ]
    
    for i in range(5):
        line = lines[i]
        pert = perturbations[i]
        for p in line:
            if p not in used:
                points.append(list(p))
                used.add(p)
        if pert not in used:
            points.append(list(pert))
            used.add(pert)
    
    # We have 20 points. Check validity.
    if not _check_no_four_collinear(points):
        # If invalid, add random perturbations
        for i in range(len(points)):
            points[i][0] += random.randint(-2, 2)
            points[i][1] += random.randint(-2, 2)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with structured config
    current_points = _generate_structured_config()
    current_score = _max_independent_set(current_points)
    best_points = current_points[:]
    best_score = current_score
    
    # Local search: move a point to a random location
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        idx = random.randint(0, 19)
        old_p = current_points[idx][:]
        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
        
        if new_p == old_p:
            continue
        if new_p in current_points:
            continue
            
        current_points[idx] = new_p
        
        if not _check_no_four_collinear(current_points):
            current_points[idx] = old_p
            continue
            
        new_score = _max_independent_set(current_points)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            current_points[idx] = old_p
            
        T *= alpha
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
