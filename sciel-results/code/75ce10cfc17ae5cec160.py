import random
import time
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    # Pre-compute all grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    
    best_set = []
    best_score = 0
    
    def squared_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def can_add(p, S_list, S_set):
        """Check if point p can be added to set S without forming isosceles triangle."""
        if p in S_set:
            return False
        if not S_list:
            return True
        
        # Check 1: Distances from p to all q in S must be distinct (Apex p)
        dists_from_p = []
        for q in S_list:
            d = squared_dist(p, q)
            if d in dists_from_p:
                return False
            dists_from_p.append(d)
            
        # Check 2: For each q in S, distance from q to p must not equal distance from q to any other r in S (Apex q)
        for q in S_list:
            d_qp = squared_dist(q, p)
            for r in S_list:
                if r == q:
                    continue
                d_qr = squared_dist(q, r)
                if d_qp == d_qr:
                    return False
        return True

    def greedy_construct(seed):
        random.seed(seed)
        S_list = []
        S_set = set()
        candidates = points.copy()
        random.shuffle(candidates)
        
        for p in candidates:
            if can_add(p, S_list, S_set):
                S_list.append(p)
                S_set.add(p)
        return S_list, S_set

    def local_search(S_list, S_set, seed):
        random.seed(seed)
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any point
            for p in points:
                if p not in S_set:
                    if can_add(p, S_list, S_set):
                        S_list.append(p)
                        S_set.add(p)
                        improved = True
                        break
            
            if improved:
                continue
                
            # 2. If stuck, try 1-out, 2-in swap
            for _ in range(5):
                if not S_list:
                    break
                p_out = random.choice(S_list)
                
                S_new_list = [p for p in S_list if p != p_out]
                S_new_set = set(S_new_list)
                
                added = []
                for p in points:
                    if p in S_new_set or p in added:
                        continue
                    if can_add(p, S_new_list + added, S_new_set | set(added)):
                        added.append(p)
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    S_list = S_new_list + added
                    S_set = S_new_set | set(added)
                    improved = True
                    break
                    
        return S_list, S_set

    # Main loop
    seed = 0
    while time.time() < deadline:
        S_list, S_set = greedy_construct(seed)
        S_list, S_set = local_search(S_list, S_set, seed)
        
        if len(S_list) > best_score:
            best_score = len(S_list)
            best_set = S_list
            
        seed += 1
        
    return best_set
