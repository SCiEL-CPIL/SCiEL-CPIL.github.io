import random
import time
from itertools import combinations

def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def _no_four_collinear(points):
    n = len(points)
    # Check all sets of 4
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _collinear(points[i], points[j], points[k]):
                        if _collinear(points[i], points[j], points[l]):
                            return False
    return True

def _precompute_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _max_indep_bnb(points):
    n = len(points)
    triples = _precompute_triples(points)
    # Map triple to a set for fast lookup? 
    # Or just list of tuples.
    # For pruning, we need to know if adding c to current_set creates a triple.
    # current_set is a list of indices.
    
    # Optimization: Precompute for each pair (i,j), the list of k such that (i,j,k) is a triple.
    pair_to_third = {}
    for i, j, k in triples:
        pair = (min(i,j), max(i,j))
        if pair not in pair_to_third:
            pair_to_third[pair] = []
        pair_to_third[pair].append(k)
        
    def solve(current_set, candidates):
        if not candidates:
            return len(current_set)
        
        # Upper bound
        ub = len(current_set) + len(candidates)
        
        # Pruning: if ub <= best_global, return 0 (or something)
        # But we need the max, so we track best found so far in this branch?
        # No, B&B usually returns the max for the subtree.
        
        # Heuristic: pick a candidate
        # Random or fixed index
        c = candidates[0]
        rest = candidates[1:]
        
        # Option 1: Exclude c
        res1 = solve(current_set, rest)
        
        # Option 2: Include c
        # Check if c can be included
        valid = True
        for i in range(len(current_set)):
            for j in range(i+1, len(current_set)):
                p1 = current_set[i]
                p2 = current_set[j]
                pair = (min(p1, p2), max(p1, p2))
                if pair in pair_to_third:
                    if c in pair_to_third[pair]:
                        valid = False
                        break
            if not valid:
                break
                
        if valid:
            res2 = solve(current_set + [c], rest)
        else:
            res2 = 0
            
        return max(res1, res2)
        
    return solve([], list(range(n)))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial random points
    def random_points():
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        return pts
        
    best_score = 20
    best_points = random_points()
    
    # Local search
    current_points = best_points[:]
    
    while time.time() < deadline:
        # Check validity
        if not _no_four_collinear(current_points):
            # Repair: move a random point
            idx = random.randint(0, 19)
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_p = (x, y)
                if new_p not in current_points:
                    current_points[idx] = new_p
                    break
            continue
            
        # Estimate score by counting disjoint triples
        # If we have many disjoint triples, score is low
        triples = _precompute_triples(current_points)
        
        # Greedy max disjoint triples
        used = [False] * 20
        disjoint_count = 0
        # Sort triples by length? All are length 3.
        # Just pick greedily
        for t in triples:
            if not used[t[0]] and not used[t[1]] and not used[t[2]]:
                used[t[0]] = used[t[1]] = used[t[2]] = True
                disjoint_count += 1
                
        # Theoretical max independent set <= 20 - disjoint_count
        # If disjoint_count >= 6, score <= 14
        # We can compute exact score if disjoint_count is high
        
        if disjoint_count >= 5:
            score = _max_indep_bnb(current_points)
            if score < best_score:
                best_score = score
                best_points = current_points[:]
                # If we hit 14, we can stop early? 
                # 14 is likely the minimum.
                if best_score <= 14:
                    break
        
        # Mutate
        # Move a random point slightly or randomly
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        # Random jump
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        if (new_x, new_y) not in current_points:
            current_points[idx] = (new_x, new_y)
            
    return [list(p) for p in best_points]
