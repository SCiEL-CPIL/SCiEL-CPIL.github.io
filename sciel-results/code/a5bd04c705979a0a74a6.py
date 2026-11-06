import time
import random

N = 16
TIME_LIMIT = 300.0
DEADLINE = time.time() + TIME_LIMIT - 10.0
TOTAL_POINTS = N * N
ALL_MASK = (1 << TOTAL_POINTS) - 1

IsoPairMask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]

def _is_isosceles(a, b, c):
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1 == d2 or d1 == d3 or d2 == d3

points = [(x, y) for x in range(N) for y in range(N)]
for i in range(TOTAL_POINTS):
    for j in range(i + 1, TOTAL_POINTS):
        mask = 0
        for k in range(TOTAL_POINTS):
            if i != k and j != k:
                if _is_isosceles(points[i], points[j], points[k]):
                    mask |= (1 << k)
        IsoPairMask[i][j] = mask
        IsoPairMask[j][i] = mask

def popcount(x):
    return bin(x).count('1')

def _greedy_random(order):
    chosen_list = []
    chosen_mask = 0
    blocked_mask = 0
    remaining_mask = ALL_MASK
    
    for p in order:
        if (remaining_mask >> p) & 1:
            if (blocked_mask >> p) & 1:
                continue
            chosen_list.append(p)
            chosen_mask |= (1 << p)
            
            for a in chosen_list[:-1]:
                blocked_mask |= IsoPairMask[p][a]
                
            remaining_mask &= ~(1 << p)
            remaining_mask &= ~blocked_mask
            
            if not remaining_mask:
                break
                
    return chosen_list

def _local_search(chosen_list):
    chosen_set = set(chosen_list)
    improved = True
    while improved:
        improved = False
        for p in list(chosen_set):
            # Check if removing p keeps maximality
            # We need to check if all points not in (chosen_set - {p}) are blocked by (chosen_set - {p})
            # This is expensive.
            # Optimization: Only try to remove p if it is not "critical".
            # Critical if removing it leaves some point uncovered.
            
            # Quick check:
            # If p is the only point in some pair that blocks a unique point v, then p is critical.
            # Too complex.
            
            # Just do the check.
            new_set = chosen_set - {p}
            # Check maximality of new_set
            # Build mask for new_set
            new_mask = 0
            for a in new_set:
                for b in new_set:
                    if a < b:
                        new_mask |= IsoPairMask[a][b]
            
            new_chosen_mask = 0
            for q in new_set:
                new_chosen_mask |= (1 << q)
                
            uncovered = ALL_MASK & ~new_chosen_mask & ~new_mask
            
            if uncovered == 0:
                chosen_set = new_set
                improved = True
                break
    return list(chosen_set)

def construct():
    start_time = time.time()
    rng = random.Random(42)
    all_points = list(range(TOTAL_POINTS))
    
    best_set = None
    best_size = float('inf')
    
    # Run many restarts
    while time.time() < DEADLINE:
        order = all_points[:]
        rng.shuffle(order)
        
        # Greedy
        chosen = _greedy_random(order)
        
        # Local Search
        chosen = _local_search(chosen)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen
            
        # If we find a very small set, we can break early?
        # Or continue to try to improve.
        
    return best_set
