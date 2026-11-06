import time
import random
import itertools
import numpy as np

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)
ALL_MASK = (1 << NUM_POINTS) - 1

# Precompute conflict masks
# bad_points[i][j] is a bitmask of points k such that (i, j, k) is isosceles
bad_points = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

def dist_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def is_isosceles(a, b, c):
    d1 = dist_sq(a, b)
    d2 = dist_sq(a, c)
    d3 = dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

# Precomputation
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        a = GRID_POINTS[i]
        b = GRID_POINTS[j]
        mask = 0
        for k in range(NUM_POINTS):
            c = GRID_POINTS[k]
            if is_isosceles(c, a, b):
                mask |= (1 << k)
        bad_points[i][j] = mask
        bad_points[j][i] = mask

def get_union_mask(S_indices):
    """Compute the union of conflict masks for all pairs in S."""
    union = 0
    for i, j in itertools.combinations(S_indices, 2):
        union |= bad_points[i][j]
    return union

def is_maximal(S_mask, union_mask):
    """Check if S is maximal given S_mask and union_mask."""
    # Points that can be added are those in ALL_MASK but not in S_mask and not in union_mask
    available = ALL_MASK & ~(S_mask | union_mask)
    return available == 0

def greedy_construct(order_indices):
    """Construct a maximal set using the given order of point indices."""
    S_indices = []
    S_mask = 0
    current_union = 0
    
    for idx in order_indices:
        if (current_union >> idx) & 1:
            continue
        
        # Update current_union
        for s in S_indices:
            current_union |= bad_points[idx][s]
            
        S_indices.append(idx)
        S_mask |= (1 << idx)
        
    return S_indices, S_mask, current_union

def local_search(S_indices, S_mask, union_mask):
    """Local search with Remove and Swap moves."""
    start_time = time.time()
    deadline = start_time + 250.0
    
    while time.time() < deadline:
        moved = False
        # Shuffle indices for randomization
        S_list = list(S_indices)
        random.shuffle(S_list)
        
        # Try Remove moves
        for p in S_list:
            if p not in S_indices: continue # might have been removed by previous move? No, we break on move
            # Check if removing p keeps it maximal
            # New S = S \ {p}
            # New Union = Union(S \ {p})
            # We need to recompute Union(S \ {p}) or update it.
            # Removing p means we lose pairs (p, s) for s in S.
            # But Union is a union, so removing a contribution doesn't simply subtract.
            # We must recompute Union for S \ {p}.
            
            S_new = [x for x in S_indices if x != p]
            if len(S_new) < 2:
                # If S has < 2 points, it's not maximal (unless grid is tiny)
                # But if we remove and it becomes non-maximal, we reject.
                # Actually, if len(S_new) < 2, it's likely not maximal.
                continue
            
            union_new = get_union_mask(S_new)
            S_mask_new = S_mask & ~(1 << p)
            
            if is_maximal(S_mask_new, union_new):
                S_indices = S_new
                S_mask = S_mask_new
                union_mask = union_new
                moved = True
                break
        
        if moved:
            continue
            
        # Try Swap moves (Remove p, Add q)
        # This allows moving to a different maximal set of same size
        # which might allow a subsequent removal.
        S_list = list(S_indices)
        random.shuffle(S_list)
        
        for p in S_list:
            if p not in S_indices: continue
            
            # Find candidates for q
            # q must not be in S
            # q must not be in union_mask (so S \ {p} \cup {q} is independent? No.)
            # Condition for S' = S \ {p} \cup {q} to be independent:
            # 1. S \ {p} is independent (yes).
            # 2. q does not form isosceles with any pair in S \ {p}.
            #    i.e., q not in Union(S \ {p}).
            # 3. q does not form isosceles with p? No, p is removed.
            
            # So we need Union(S \ {p}).
            # Computing this for every p is expensive.
            # Let's just try random q's.
            
            # Pick a random q not in S
            available_q = ALL_MASK & ~S_mask
            if available_q == 0:
                continue
            
            # Get a random bit from available_q
            # We can iterate a few times
            for _ in range(10):
                # Pick random q
                q_idx = 0
                temp_mask = available_q
                while temp_mask:
                    # Pick random set bit
                    # Simple way: get list of bits, pick random
                    # This is slow.
                    # Better: random integer, check if bit is set?
                    pass
                
                # Let's just pick a random index from 0 to NUM_POINTS-1
                q = random.randint(0, NUM_POINTS - 1)
                if (S_mask >> q) & 1:
                    continue
                if (union_mask >> q) & 1:
                    # q is blocked by current union. 
                    # But removing p might unblock q.
                    # So we must check Union(S \ {p}).
                    pass
                
                # Compute Union(S \ {p})
                S_temp = [x for x in S_indices if x != p]
                if len(S_temp) < 1:
                    union_temp = 0
                else:
                    union_temp = get_union_mask(S_temp)
                
                # Check if q is independent with S_temp
                if (union_temp >> q) & 1:
                    continue
                
                # Check if S_temp \cup {q} is maximal
                S_mask_temp = S_mask & ~(1 << p) | (1 << q)
                
                # Compute Union(S_temp \cup {q})
                # Union(S_temp \cup {q}) = Union(S_temp) | Union({q, s} for s in S_temp)
                # We can compute this incrementally or from scratch.
                # From scratch is safer.
                S_new = S_temp + [q]
                union_new = get_union_mask(S_new)
                
                if is_maximal(S_mask_temp, union_new):
                    S_indices = S_new
                    S_mask = S_mask_temp
                    union_mask = union_new
                    moved = True
                    break
            
            if moved:
                break
                
        if not moved:
            # Stagnation, break to restart
            break
            
    return S_indices, S_mask, union_mask

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_set_indices = None
    best_size = float('inf')
    
    # Initial greedy with row-major order
    initial_order = list(range(NUM_POINTS))
    S, S_mask, union_mask = greedy_construct(initial_order)
    S, S_mask, union_mask = local_search(S, S_mask, union_mask)
    if len(S) < best_size:
        best_size = len(S)
        best_set_indices = S
        
    # Iterated Local Search
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        S, S_mask, union_mask = greedy_construct(order)
        S, S_mask, union_mask = local_search(S, S_mask, union_mask)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set_indices = S
            # If we find a very small set, we might want to spend more time on it
            # But for now, we just keep going.
            
    # Convert indices back to (x, y) tuples
    return [GRID_POINTS[i] for i in best_set_indices]
