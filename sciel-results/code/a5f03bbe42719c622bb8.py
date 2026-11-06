import time
import random

N = 64
TIME_LIMIT = 330

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def check_valid(p, S, D_sets):
    if not S:
        return True
    
    dists = []
    for q in S:
        d = dist_sq(p, q)
        dists.append(d)
        
    if len(set(dists)) != len(dists):
        return False
        
    for i in range(len(S)):
        if dists[i] in D_sets[i]:
            return False
            
    return True

def add_point(p, S, D_sets):
    S.append(p)
    new_D = set()
    for i in range(len(S) - 1):
        d = dist_sq(p, S[i])
        new_D.add(d)
        D_sets[i].add(d)
    D_sets.append(new_D)

def greedy_random(points):
    S = []
    D_sets = []
    for p in points:
        if check_valid(p, S, D_sets):
            add_point(p, S, D_sets)
    return S

def greedy_max_min(points):
    S = []
    D_sets = []
    remaining = list(points)
    while remaining:
        best_p = None
        best_min_dist = -1
        for p in remaining:
            if check_valid(p, S, D_sets):
                if S:
                    min_d = min(dist_sq(p, q) for q in S)
                else:
                    min_d = float('inf')
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_p = p
        if best_p is None:
            break
        add_point(best_p, S, D_sets)
        remaining.remove(best_p)
    return S

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    all_points = [(x, y) for x in range(N) for y in range(N)]
    best_S = []
    
    # 1. Random Greedy
    # We can run this many times
    while time.time() - start_time < TIME_LIMIT - 5:
        random.shuffle(all_points)
        S = greedy_random(all_points)
        if len(S) > len(best_S):
            best_S = S
            
    # 2. Max-Min Greedy (slower, but potentially better)
    while time.time() - start_time < TIME_LIMIT - 2:
        S = greedy_max_min(all_points)
        if len(S) > len(best_S):
            best_S = S
        # Break if we can't improve much? No, just run until time limit.
        
    return best_S
# EVOLVE-BLOCK-END
