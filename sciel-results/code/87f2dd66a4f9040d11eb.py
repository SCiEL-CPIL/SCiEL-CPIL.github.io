import time
import random

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances between all pairs
dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute isosceles triangles
# For each point p, store the set of pairs (a, b) such that {p, a, b} is isosceles
# This means: if p is in S, then NOT (a in S AND b in S) for any such pair
iso_pairs = [set() for _ in range(NUM_POINTS)]

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        for k in range(j + 1, NUM_POINTS):
            d1 = dist2[i][j]
            d2 = dist2[i][k]
            d3 = dist2[j][k]
            if d1 == d2 or d1 == d3 or d2 == d3:
                iso_pairs[i].add((j, k))
                iso_pairs[j].add((i, k))
                iso_pairs[k].add((i, j))

# Precompute conflict graph: two points conflict if they can't both be in S with some third point
# Actually, for maximality checking, we need: point p can be added to S iff for all (a,b) in iso_pairs[p], NOT (a in S and b in S)

def can_add(p, S_set):
    """Check if point p can be added to S_set without creating an isosceles triangle."""
    for a, b in iso_pairs[p]:
        if a in S_set and b in S_set:
            return False
    return True

def is_maximal(S_set):
    """Check if S_set is maximal: no point outside S_set can be added."""
    for p in range(NUM_POINTS):
        if p not in S_set and can_add(p, S_set):
            return False
    return True

def greedy_construct(rng, deadline):
    """Greedy construction: add points in random order, then local search to minimize."""
    S_set = set()
    
    # Get all points and shuffle
    points = list(range(NUM_POINTS))
    rng.shuffle(points)
    
    # Greedy: add points if they can be added
    for p in points:
        if can_add(p, S_set):
            S_set.add(p)
    
    # Local search: repeatedly try to remove points while maintaining maximality
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order to avoid getting stuck in local optima
        removal_order = list(S_set)
        rng.shuffle(removal_order)
        
        for p in removal_order:
            if time.time() >= deadline:
                break
            
            # Try removing p
            trial_S = S_set - {p}
            
            # Check if trial_S is still maximal
            if is_maximal(trial_S):
                S_set = trial_S
                improved = True
                break  # Restart the removal loop with the new set
    
    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    best_set = None
    best_len = float('inf')
    
    # Multiple restarts with different seeds
    restart_count = 0
    while time.time() < deadline - 10:  # Leave 10s buffer
    
        current_rng = random.Random(restart_count * 7919 + 12345)
        
        # Greedy + local search
        S_set = greedy_construct(current_rng, deadline - 5)
        
        if len(S_set) < best_len:
            best_len = len(S_set)
            best_set = S_set
        
        restart_count += 1
        
        # If we've found a very good solution, do more intensive local search
        if best_len <= 15 and time.time() < deadline - 20:
            # Try more aggressive local search on the best set
            aggressive_rng = random.Random(restart_count * 31 + 999)
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for p in list(best_set):
                    if time.time() >= deadline - 10:
                        break
                    trial_S = best_set - {p}
                    if is_maximal(trial_S):
                        best_set = trial_S
                        best_len = len(best_set)
                        improved = True
                        break
    
    # Convert indices back to coordinates
    if best_set is None:
        return []
    return [POINTS[i] for i in best_set]
