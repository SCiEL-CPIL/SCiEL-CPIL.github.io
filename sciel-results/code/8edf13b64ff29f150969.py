import time
import random

# Precomputation
N = 256
all_3aps = []
all_4aps = []
# 3-APs: (a, b, c) with a < b < c, b-a = c-b
for d in range(1, N):
    for a in range(1, N - 2*d + 1):
        b = a + d
        c = a + 2*d
        all_3aps.append((a, b, c))

# 4-APs: (a, b, c, d) with a < b < c < d, b-a = c-b = d-c
for d in range(1, N):
    for a in range(1, N - 3*d + 1):
        b = a + d
        c = a + 2*d
        dd = a + 3*d
        all_4aps.append((a, b, c, dd))

# Index 3-APs by element
aps_3_by_elem = {i: [] for i in range(1, N+1)}
for ap in all_3aps:
    for x in ap:
        aps_3_by_elem[x].append(ap)

# Index 4-APs by element
aps_4_by_elem = {i: [] for i in range(1, N+1)}
for ap in all_4aps:
    for x in ap:
        aps_4_by_elem[x].append(ap)

def count_3aps(A):
    count = 0
    A_set = set(A)
    for ap in all_3aps:
        if all(x in A_set for x in ap):
            count += 1
    return count

def is_4ap_free(A):
    A_set = set(A)
    for ap in all_4aps:
        if all(x in A_set for x in ap):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Greedy Initialization
    A = set()
    candidates = set(range(1, N+1))
    
    # We want to maximize 3-APs.
    # Greedy: pick x that adds most 3-APs without breaking 4-AP freedom.
    while candidates and time.time() < deadline - 10:
        best_x = None
        best_gain = -1
        best_new_score = -1
        
        for x in candidates:
            # Check 4-AP violation
            # If adding x creates a 4-AP, skip
            valid = True
            for ap in aps_4_by_elem[x]:
                # Check if other 3 are in A
                others = [y for y in ap if y != x]
                if all(y in A for y in others):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Count new 3-APs
            # New 3-APs are those containing x where the other 2 are in A
            gain = 0
            for ap in aps_3_by_elem[x]:
                others = [y for y in ap if y != x]
                if all(y in A for y in others):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            break
        
        A.add(best_x)
        candidates.remove(best_x)
        
    # Local Search
    current_score = count_3aps(A)
    improved = True
    
    while improved and time.time() < deadline - 2:
        improved = False
        best_move = None
        best_move_score = current_score
        move_type = None
        move_args = None
        
        # Try Additions
        for x in candidates:
            valid = True
            for ap in aps_4_by_elem[x]:
                others = [y for y in ap if y != x]
                if all(y in A for y in others):
                    valid = False
                    break
            if not valid:
                continue
            
            gain = 0
            for ap in aps_3_by_elem[x]:
                others = [y for y in ap if y != x]
                if all(y in A for y in others):
                    gain += 1
            
            if gain > 0:
                new_score = current_score + gain
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = x
                    move_type = 'add'
        
        # Try Swaps
        # Optimization: Only swap if it seems promising?
        # For each a in A, for each b in candidates
        # This is O(|A| * |C|).
        for a in A:
            # Remove a
            A_temp = A - {a}
            # Calculate score loss from removing a
            loss_a = 0
            for ap in aps_3_by_elem[a]:
                others = [y for y in ap if y != a]
                if all(y in A_temp for y in others):
                    loss_a += 1
            
            for b in candidates:
                # Check if adding b to A_temp is valid
                valid = True
                for ap in aps_4_by_elem[b]:
                    others = [y for y in ap if y != b]
                    if all(y in A_temp for y in others):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate gain from adding b to A_temp
                gain_b = 0
                for ap in aps_3_by_elem[b]:
                    others = [y for y in ap if y != b]
                    if all(y in A_temp for y in others):
                        gain_b += 1
                
                new_score = current_score - loss_a + gain_b
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (a, b)
                    move_type = 'swap'
        
        if move_type == 'add':
            A.add(best_move)
            candidates.remove(best_move)
            current_score = best_move_score
            improved = True
        elif move_type == 'swap':
            a, b = best_move
            A.remove(a)
            A.add(b)
            candidates.remove(b)
            candidates.add(a)
            current_score = best_move_score
            improved = True
            
    return sorted(A)
