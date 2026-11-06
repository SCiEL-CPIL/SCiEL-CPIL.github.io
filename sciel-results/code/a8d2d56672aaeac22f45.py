import time
import random
import math

# Assuming N is part of the problem definition, usually fixed.
# The failed code used N=64. I will use N=64.
N = 64
TIME_LIMIT = 330
START_TIME = time.time()

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def check_valid(p, S, dist_sets):
    """
    S: list of points
    dist_sets: list of sets of squared distances from S[i] to other points in S
    Returns True if p can be added to S.
    """
    if not S:
        return True
    
    # 1. Distances from p to all points in S must be distinct (no isosceles with apex p)
    dists = []
    for q in S:
        d = dist_sq(p, q)
        dists.append(d)
    
    if len(set(dists)) != len(dists):
        return False
        
    # 2. For each q in S, d(p, q) must not be in dist_sets[i]
    # dist_sets[i] contains distances from S[i] to all other points in S.
    # If d(p, S[i]) is in dist_sets[i], it means there exists S[j] such that d(S[i], S[j]) == d(p, S[i]).
    # This forms an isosceles triangle with apex S[i].
    for i, q in enumerate(S):
        if dists[i] in dist_sets[i]:
            return False
            
    return True

def add_point(p, S, dist_sets):
    """
    Adds p to S and updates dist_sets.
    """
    S.append(p)
    new_D = set()
    for i, q in enumerate(S[:-1]):
        d = dist_sq(p, q)
        new_D.add(d)
        dist_sets[i].add(d)
    dist_sets.append(new_D)

def greedy_random(points):
    S = []
    dist_sets = []
    remaining = list(points)
    random.shuffle(remaining)
    for p in remaining:
        if check_valid(p, S, dist_sets):
            add_point(p, S, dist_sets)
    return S

def greedy_max_min(points):
    S = []
    dist_sets = []
    remaining = list(points)
    while remaining:
        best_p = None
        best_min_dist = -1
        valid_candidates = []
        for p in remaining:
            if check_valid(p, S, dist_sets):
                if S:
                    min_d = min(dist_sq(p, q) for q in S)
                else:
                    min_d = float('inf')
                valid_candidates.append((min_d, p))
        
        if not valid_candidates:
            break
            
        # Pick the one with max min_dist
        valid_candidates.sort(reverse=True)
        best_p = valid_candidates[0][1]
        add_point(best_p, S, dist_sets)
        remaining.remove(best_p)
    return S

def construct():
    global START_TIME
    all_points = [(x, y) for x in range(N) for y in range(N)]
    best_S = []
    
    deadline = START_TIME + TIME_LIMIT - 10
    
    # Run random greedy multiple times
    while time.time() < deadline:
        S = greedy_random(all_points)
        if len(S) > len(best_S):
            best_S = S
            if len(best_S) > 100: # Heuristic early stop if very good
                break
                
    # Run max-min greedy if time permits
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline:
                break
            S = greedy_max_min(all_points)
            if len(S) > len(best_S):
                best_S = S

    return best_S
