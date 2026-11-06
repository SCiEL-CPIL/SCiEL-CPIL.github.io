import time
import random
from itertools import combinations

# Precomputation
N = 16
K = 4
ALL_SETS = []
MASKS = []

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Generate all 4-subsets of [1, 16]
    # Represent as bitmasks
    for comb in combinations(range(1, N+1), K):
        mask = 0
        for x in comb:
            mask |= (1 << (x-1))
        MASKS.append(mask)
    
    M = len(MASKS) # 1820
    
    # 2. Precompute intersections inter[i][j]
    # inter[i][j] is bitmask of MASKS[i] & MASKS[j]
    # Using a list of lists or a flat array? 
    # 1820 * 1820 is ~3.3M integers. 
    # A list of lists of ints is fine.
    inter = [[0] * M for _ in range(M)]
    for i in range(M):
        mi = MASKS[i]
        for j in range(i+1, M):
            inter[i][j] = mi & MASKS[j]
            inter[j][i] = inter[i][j]
        inter[i][i] = mi # Though we only care about distinct sets, 
                         # but for safety in logic if indices are same? 
                         # The problem says "three distinct sets". 
                         # So i, j, k must be distinct.
                         # In our check, we pick distinct i, j from F, and k is new, so k != i, k != j.
                         # i and j are distinct in F.
                         # So we never use inter[i][i] in the conflict check logic directly 
                         # except maybe if we don't enforce distinctness carefully.
                         # But F contains distinct sets.
    
    best_family = []
    
    # Helper to check if candidate k can be added to family indices
    def can_add(k, f_indices):
        # f_indices is a list of indices of sets in the current family
        # We need to check if there exist distinct i, j in f_indices
        # such that inter[i][j] == inter[j][k] == inter[k][i]
        
        # Group f_indices by their intersection with k
        # v = MASKS[j] & MASKS[k]
        groups = {}
        for j in f_indices:
            v = inter[j][k]
            if v not in groups:
                groups[v] = []
            groups[v].append(j)
            
        # For each group, check if any pair (i, j) in the group has inter[i][j] == v
        for v, members in groups.items():
            if len(members) < 2:
                continue
            # Check pairs in members
            # members are indices j such that MASKS[j] & MASKS[k] == v
            # We need to see if there are i, j in members such that MASKS[i] & MASKS[j] == v
            # Note: i and j are distinct.
            for idx_a in range(len(members)):
                i = members[idx_a]
                for idx_b in range(idx_a + 1, len(members)):
                    j = members[idx_b]
                    if inter[i][j] == v:
                        return False
        return True

    # Greedy with random restarts
    order = list(range(M))
    
    while time.time() < deadline:
        random.shuffle(order)
        current_family = []
        # We can try to add sets in the shuffled order
        for k in order:
            if can_add(k, current_family):
                current_family.append(k)
        
        if len(current_family) > len(best_family):
            best_family = current_family[:]
            
        # If we hit the max possible (250), we can stop early?
        # The task says "maximum=250". It might be the hard limit or just an upper bound for the return.
        # "Maximize family cardinality."
        # If we find 250, we can't do better.
        if len(best_family) >= 250:
            break

    # Convert indices back to sorted lists of elements
    result = []
    for idx in best_family:
        mask = MASKS[idx]
        elems = []
        for i in range(1, N+1):
            if mask & (1 << (i-1)):
                elems.append(i)
        result.append(elems)
        
    return result
