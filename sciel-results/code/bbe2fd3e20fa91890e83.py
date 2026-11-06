import time
import random
import numpy as np

N = 16
DEADLINE = 325.0
NUM_POINTS = N * N

# Precompute coordinates
xs = [i % N for i in range(NUM_POINTS)]
ys = [i // N for i in range(NUM_POINTS)]

# Precompute squared distances
D = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    for j in range(i, NUM_POINTS):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        d2 = dx*dx + dy*dy
        D[i][j] = d2
        D[j][i] = d2

# Precompute block masks
# block_mask[a][b] is an integer bitset of points q such that {a,b,q} is isosceles
block_masks = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for a in range(NUM_POINTS):
    for b in range(a+1, NUM_POINTS):
        mask = 0
        da_b = D[a][b]
        for q in range(NUM_POINTS):
            if q == a or q == b:
                continue
            da_q = D[a][q]
            db_q = D[b][q]
            if da_q == db_q or da_q == da_b or db_q == da_b:
                mask |= (1 << q)
        block_masks[a][b] = mask
        block_masks[b][a] = mask

def _is_isosceles(p, a, b):
    return D[p][a] == D[p][b] or D[p][a] == D[a][b] or D[p][b] == D[a][b]

def _can_add(p, S_list):
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if _is_isosceles(p, a, b):
                return False
    return True

def _is_maximal(S_set, S_list):
    n_S = len(S_list)
    if n_S < 2:
        return False
    
    # Compute union of block masks for all pairs in S
    union_mask = 0
    for i in range(n_S):
        a = S_list[i]
        for j in range(i + 1, n_S):
            b = S_list[j]
            union_mask |= block_masks[a][b]
            
    # Check if all points not in S are blocked
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
        if not (union_mask >> q) & 1:
            return False
            
    return True

def _reduce_to_minimal(S_list, S_set):
    changed = True
    while changed:
        changed = False
        for i in range(len(S_list)):
            p = S_list[i]
            S_test_list = S_list[:i] + S_list[i+1:]
            S_test_set = S_set - {p}
            
            if len(S_test_list) < 2:
                continue
                
            if _is_maximal(S_test_set, S_test_list):
                S_list = S_test_list
                S_set = S_test_set
                changed = True
                break
    return S_list, S_set

def _greedy_construct(rng):
    S_list = []
    S_set = set()
    
    # Pick a random start
    start = rng.randint(0, NUM_POINTS - 1)
    S_list.append(start)
    S_set.add(start)
    
    remaining = [p for p in range(NUM_POINTS) if p != start]
    rng.shuffle(remaining)
    
    for p in remaining:
        if _can_add(p, S_list):
            S_list.append(p)
            S_set.add(p)
            
    return S_list, S_set

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_size = NUM_POINTS
    best_set = None
    
    all_points = list(range(NUM_POINTS))
    
    # Run multiple attempts
    while time.time() - start_time < DEADLINE:
        # 1. Greedy Construction
        S_list, S_set = _greedy_construct(rng)
        
        # 2. Reduction
        S_list, S_set = _reduce_to_minimal(S_list, S_set)
        
        if len(S_list) < best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            
        if best_size <= 8:
            break
            
        # 3. Local Search (Swap)
        current_list = S_list[:]
        current_set = S_set.copy()
        
        # Limit local search time per attempt
        local_search_start = time.time()
        improved = True
        while improved and time.time() - local_search_start < 10.0:
            improved = False
            # Try swapping each point in current_set with a point not in current_set
            for i in range(len(current_list)):
                p_remove = current_list[i]
                S_without_list = current_list[:i] + current_list[i+1:]
                S_without_set = current_set - {p_remove}
                
                # Candidate points to add
                candidates = [p for p in all_points if p not in current_set]
                rng.shuffle(candidates)
                
                for p_add in candidates[:100]: # Limit candidates
                    if not _can_add(p_add, S_without_list):
                        continue
                    
                    S_new_list = S_without_list + [p_add]
                    S_new_set = S_without_set | {p_add}
                    
                    if _is_maximal(S_new_set, S_new_list):
                        # Accept the swap
                        current_list = S_new_list
                        current_set = S_new_set
                        improved = True
                        
                        # Try to reduce again after swap
                        current_list, current_set = _reduce_to_minimal(current_list, current_set)
                        
                        if len(current_list) < best_size:
                            best_size = len(current_list)
                            best_set = current_list[:]
                            
                        if best_size <= 8:
                            return [(p % N, p // N) for p in best_set]
                        
                        break # Restart the swap loop with new set
                if improved:
                    break
                    
    # Convert to (x, y) tuples
    if best_set is None:
        return []
        
    return [(p % N, p // N) for p in best_set]
