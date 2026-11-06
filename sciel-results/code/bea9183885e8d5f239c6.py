import time
import random
from itertools import combinations

_ALL_SETS = []
_all_set_masks = []
for c in combinations(range(1, 17), 4):
    mask = 0
    for x in c:
        mask |= (1 << (x - 1))
    _ALL_SETS.append(c)
    _all_set_masks.append(mask)

N_SETS = len(_ALL_SETS)

# Precompute intersection masks
_inter_mask = [[0] * N_SETS for _ in range(N_SETS)]
for i in range(N_SETS):
    mi = _all_set_masks[i]
    for j in range(i+1, N_SETS):
        m = mi & _all_set_masks[j]
        _inter_mask[i][j] = m
        _inter_mask[j][i] = m

# For each k, precompute conflict pairs
# For k, iterate over S ⊆ A_k, find i's with A_i∩A_k=S, j's with A_j∩A_k=S,
# then check pairs (i,j) where A_i∩A_j=S

conflict_pairs = [set() for _ in range(N_SETS)]

for k in range(N_SETS):
    mk = _all_set_masks[k]
    # Enumerate all subsets S of A_k
    # A_k has 4 elements, so 16 subsets
    k_elements = _ALL_SETS[k]
    for r in range(5):  # 0 to 4
        for S_tuple in combinations(k_elements, r):
            S_mask = 0
            for x in S_tuple:
                S_mask |= (1 << (x - 1))
            
            # Find all i such that A_i ∩ A_k = S_mask
            i_list = []
            for i in range(N_SETS):
                if i == k:
                    continue
                if _inter_mask[i][k] == S_mask:
                    i_list.append(i)
            
            # For each pair (i,j) in i_list, check if A_i∩A_j = S_mask
            for ii in range(len(i_list)):
                for jj in range(ii+1, len(i_list)):
                    i = i_list[ii]
                    j = i_list[jj]
                    if _inter_mask[i][j] == S_mask:
                        conflict_pairs[k].add((i, j))

def _can_add(k, family_set):
    for (a, b) in conflict_pairs[k]:
        if a in family_set and b in family_set:
            return False
    return True

def _greedy_build(order, deadline):
    family = set()
    for idx in order:
        if time.time() > deadline:
            break
        if _can_add(idx, family):
            family.add(idx)
    return family

def _local_search(family, deadline, max_iter=500):
    family = set(family)
    current_size = len(family)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        members = list(family)
        random.shuffle(members)
        
        for rem in members[:15]:
            family.discard(rem)
            
            non_members = [i for i in range(N_SETS) if i not in family]
            random.shuffle(non_members)
            
            for cand in non_members[:60]:
                if _can_add(cand, family):
                    family.add(cand)
                    if len(family) > current_size:
                        current_size = len(family)
                    improved = True
                    break
            
            if not improved:
                family.add(rem)
        
        if not improved:
            break
    
    return family

def construct():
    deadline = time.time() + 75
    
    best_family = set()
    best_size = 0
    
    order = list(range(N_SETS))
    family = _greedy_build(order, deadline)
    best_size = len(family)
    best_family = set(family)
    
    if time.time() < deadline:
        order = list(range(N_SETS-1, -1, -1))
        family = _greedy_build(order, deadline)
        if len(family) > best_size:
            best_size = len(family)
            best_family = set(family)
    
    while time.time() < deadline - 20:
        order = list(range(N_SETS))
        random.shuffle(order)
        family = _greedy_build(order, deadline)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = set(family)
        
        if time.time() < deadline - 10:
            improved = _local_search(family, deadline)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = set(improved)
        
        if time.time() < deadline - 10:
            improved2 = _local_search(best_family, deadline)
            if len(improved2) > best_size:
                best_size = len(improved2)
                best_family = set(improved2)
    
    result = [sorted(_ALL_SETS[i]) for i in sorted(best_family)]
    return result
