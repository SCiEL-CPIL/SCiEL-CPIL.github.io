import time
import random
import numpy as np

N = 64
TIME_LIMIT = 280.0

def construct():
    start_time = time.time()
    
    # Precompute all grid points
    points = [(i, j) for i in range(N) for j in range(N)]
    
    best_set = set()
    best_size = 0
    
    # Helper to check if a point can be added to a set
    # S is a list of points
    def can_add(c, S):
        # S is a list of (x, y)
        # Check against all pairs in S
        n = len(S)
        if n < 2:
            return True
        
        # Calculate distances from c to all points in S
        # And pairwise distances in S
        # To avoid O(n^2) distance calcs from scratch every time,
        # we can just compute on the fly. n is expected to be small (< 100).
        
        dists_from_c = [0] * n
        for i, p in enumerate(S):
            dx = c[0] - p[0]
            dy = c[1] - p[1]
            dists_from_c[i] = dx*dx + dy*dy
            
        for i in range(n):
            pi = S[i]
            di = dists_from_c[i]
            for j in range(i + 1, n):
                pj = S[j]
                dj = dists_from_c[j]
                if di == dj:
                    return False
                # dist(pi, pj)
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dij = dx*dx + dy*dy
                if di == dij:
                    return False
                if dj == dij:
                    return False
        return True

    # Greedy initialization
    random.shuffle(points)
    current_set = []
    used = set()
    
    for p in points:
        if can_add(p, current_set):
            current_set.append(p)
            used.add(p)
            
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Local Search / Simulated Annealing
    # We will try to improve the set.
    # Strategy: 
    # 1. Try to add unused points.
    # 2. If stuck, remove a point and try to add others.
    
    unused = [p for p in points if p not in used]
    
    # We need a faster way to check validity if we do many swaps.
    # But let's stick to the simple one first and see if it's fast enough.
    # If n is ~50, n^2 is 2500. 4000 unused points. 
    # 4000 * 2500 = 10M ops. Might be tight in Python.
    
    # Let's optimize can_add by using a set of forbidden configurations?
    # Actually, the conflict is specific to the triple.
    
    # Let's try a different approach: 
    # Maintain a list of "bad" pairs? No, it's triples.
    
    # Let's use a greedy with backtracking limited depth?
    
    # Let's just run the greedy multiple times with different shuffles.
    # And then try to add more points to the best found.
    
    for _ in range(10): # Multiple restarts
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        random.shuffle(points)
        curr = []
        used_curr = set()
        for p in points:
            if can_add(p, curr):
                curr.append(p)
                used_curr.add(p)
        
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = set(curr)
            
        # Try to improve curr by adding unused points
        remaining = [p for p in points if p not in used_curr]
        random.shuffle(remaining)
        improved = True
        while improved:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            improved = False
            for p in remaining:
                if can_add(p, curr):
                    curr.append(p)
                    used_curr.add(p)
                    remaining.remove(p)
                    improved = True
                    if len(curr) > best_size:
                        best_size = len(curr)
                        best_set = set(curr)
                    break # Restart loop to re-evaluate remaining? 
                    # Actually, removing from list while iterating is bad.
                    # Better to use a pointer or filter.
                    
    # Final check
    return list(best_set)
