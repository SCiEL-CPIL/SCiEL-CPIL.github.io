import numpy as np
import time
import random
from collections import defaultdict

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290s deadline, 40s margin
    
    best_points = []
    best_count = 0
    
    # Precompute all possible squared distances? 
    # Grid is 64x64, so 4096 points. Distances are integers from 0 to (63^2 + 63^2) = 7938.
    # We can use bitsets or boolean arrays for distance conflicts for speed.
    
    def try_add_point(S, dist_conflicts, p):
        """
        S: list of points (tuples)
        dist_conflicts: 2D structure or dict. 
        For speed, let's maintain for each point in S, a set of squared distances to other points in S.
        Also, for the new point p, we need to check:
        1. No distance from p to any q in S is already present in q's set.
        2. All distances from p to q1, q2 ... are unique among themselves.
        
        To make this faster, we can maintain a global structure? 
        Actually, the current logic is O(|S|) per check which is fine for |S| < 100.
        """
        if not S:
            S.append(p)
            dist_conflicts[p] = set()
            return True
        
        px, py = p
        new_dists = []
        
        for q in S:
            qx, qy = q
            d_sq = (px - qx)**2 + (py - qy)**2
            
            # Check if this distance already exists in q's distance set
            # This ensures no isosceles triangle with apex q
            if d_sq in dist_conflicts[q]:
                return False
            
            new_dists.append(d_sq)
            
        # Check if any two distances from p to distinct q1, q2 are equal
        # This ensures no isosceles triangle with apex p
        if len(set(new_dists)) != len(new_dists):
            return False
            
        # If valid, update structures
        for i, q in enumerate(S):
            dist_conflicts[q].add(new_dists[i])
            
        dist_conflicts[p] = set(new_dists)
        S.append(p)
        return True

    def greedy_construct(seed_val, ordering_strategy):
        random.seed(seed_val)
        S = []
        dist_conflicts = {}
        
        # Generate candidates
        candidates = [(x, y) for x in range(N) for y in range(N)]
        
        if ordering_strategy == "random":
            random.shuffle(candidates)
        elif ordering_strategy == "geometric_min_dist":
            # This requires dynamic re-ordering, which is expensive in a simple loop.
            # Instead, we can use a static heuristic order or a "center-out" or "corner-in" order.
            # Let's try "Manhattan distance from center" or similar to spread points.
            center = (N/2.0, N/2.0)
            candidates.sort(key=lambda p: abs(p[0]-center[0]) + abs(p[1]-center[1]))
            # Shuffle slightly to break symmetry
            random.shuffle(candidates)
        elif ordering_strategy == "geometric_max_dist":
            # Pick points far from each other first.
            # This is hard to do statically. 
            # Let's stick to random for simplicity in this block, 
            # but use the "try_add" logic efficiently.
            random.shuffle(candidates)
        else:
            random.shuffle(candidates)
            
        # Optimization: Instead of trying ALL candidates, 
        # we can stop early if we've tried many and added few? 
        # No, we should try all in the ordering.
        
        for p in candidates:
            if time.time() > deadline - 5:
                break
            try_add_point(S, dist_conflicts, p)
            
        return S, dist_conflicts

    def local_search(S, dist_conflicts, time_limit):
        start_ls = time.time()
        # Try to add random points again
        # We need a check function that doesn't modify
        def check_valid(S, dist_conflicts, p):
            if not S:
                return True
            px, py = p
            new_dists = []
            for q in S:
                qx, qy = q
                d_sq = (px - qx)**2 + (py - qy)**2
                if d_sq in dist_conflicts[q]:
                    return False
                new_dists.append(d_sq)
            if len(set(new_dists)) != len(new_dists):
                return False
            return True

        # Precompute a list of all points not in S for faster random selection?
        # S is small, so random.randint is fine.
        
        improvement = 0
        no_improve_counter = 0
        max_no_improve = 2000
        
        while time.time() - start_ls < time_limit and no_improve_counter < max_no_improve:
            p = (random.randint(0, N-1), random.randint(0, N-1))
            # Check if p is already in S
            if p in S:
                continue
            
            if check_valid(S, dist_conflicts, p):
                try_add_point(S, dist_conflicts, p)
                improvement += 1
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                
        return S, dist_conflicts

    # Run multiple trials with different strategies
    trial_count = 0
    strategies = ["random", "geometric_min_dist", "random", "random"]
    
    while time.time() < deadline - 10:
        trial_count += 1
        seed_val = trial_count * 1000 + int(time.time() * 1000) % 1000
        strategy = strategies[trial_count % len(strategies)]
        
        S, dist_conflicts = greedy_construct(seed_val, strategy)
        current_count = len(S)
        
        if current_count > best_count:
            best_points = S
            best_count = current_count
            
            # If we improved, spend a little time on local search
            if time.time() < deadline - 20 and current_count > 0:
                S_ls, dist_conflicts_ls = local_search(S, dist_conflicts, 5.0)
                if len(S_ls) > best_count:
                    best_points = S_ls
                    best_count = len(S_ls)
                    
        # Adaptive: if we've run many trials and haven't improved, maybe try a denser local search on the current best?
        # For now, keep it simple to ensure we finish in time.

    return best_points
