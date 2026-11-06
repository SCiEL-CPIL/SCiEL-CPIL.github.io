"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Generate all 4-element subsets of [1,16]
    all_sets = [frozenset(c) for c in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Precompute intersection masks
    def to_mask(fs):
        m = 0
        for x in fs:
            m |= (1 << (x - 1))
        return m
    
    masks = [to_mask(s) for s in all_sets]
    mask_array = np.array(masks, dtype=np.uint32)
    
    # Compute all pairwise intersections
    inter_matrix = np.zeros((n_sets, n_sets), dtype=np.uint32)
    for i in range(n_sets):
        inter_matrix[i, i:] = mask_array[i] & mask_array[i:]
        inter_matrix[i:, i] = inter_matrix[i, i:]
    
    # Precompute for each pair (i,j) the intersection value, and for each (i, val) 
    # the list of j such that inter_matrix[i,j] == val
    # This allows O(1) lookup: given i, j, and val=inter[i][j], find all k where inter[i][k]==val AND inter[j][k]==val
    
    # For faster sunflower checking, precompute for each i a dict: val -> list of j where inter[i][j]==val
    inter_by_val = [dict() for _ in range(n_sets)]
    for i in range(n_sets):
        for j in range(n_sets):
            val = int(inter_matrix[i, j])
            if val not in inter_by_val[i]:
                inter_by_val[i][val] = []
            inter_by_val[i][val].append(j)
    
    def _forms_sunflower_fast(new_idx, family_indices):
        """Check if new_idx forms a sunflower with any two sets in family_indices."""
        for i in range(len(family_indices)):
            a_idx = family_indices[i]
            ab_mask = int(inter_matrix[a_idx, new_idx])
            for j in range(i + 1, len(family_indices)):
                b_idx = family_indices[j]
                if int(inter_matrix[a_idx, b_idx]) == ab_mask and int(inter_matrix[b_idx, new_idx]) == ab_mask:
                    return True
        return False
    
    # Faster check using precomputed structures
    def _forms_sunflower_fast2(new_idx, family_set):
        """Check if new_idx forms a sunflower with any two sets in family_set (a set of indices)."""
        # For each pair (a, b) in family, check if inter[a][new] == inter[a][b] == inter[b][new]
        # Equivalent: for each a in family, let m = inter[a][new]. 
        # We need another b in family such that inter[a][b] == m AND inter[b][new] == m.
        # inter[b][new] == m means b is in inter_by_val[new][m] (sets with same intersection with new as a has with new)
        # AND inter[a][b] == m means b is in inter_by_val[a][m]
        
        family_list = list(family_set)
        for a_idx in family_list:
            m = int(inter_matrix[a_idx, new_idx])
            # Find all b in family such that inter[a][b] == m AND inter[b][new] == m
            # b must be in intersection of family_set, inter_by_val[a][m], and inter_by_val[new][m]
            candidates_a = inter_by_val[a_idx].get(m, [])
            candidates_new = inter_by_val[new_idx].get(m, [])
            # We need b in family_set that is in both candidates_a and candidates_new
            # And b != a_idx and b != new_idx
            for b in candidates_a:
                if b in family_set and b != a_idx and b != new_idx:
                    if int(inter_matrix[b_idx if False else b, new_idx]) == m:
                        return True
        return False
    
    # Even faster: use bitsets for family membership
    # n_sets = 1820, so we can use Python integers as bitsets
    def greedy_fast(order, time_limit=None):
        """
        Greedy with incremental conflict counting using bitsets.
        """
        if time_limit is None:
            time_limit = deadline
        in_family = [False] * n_sets
        family = []
        conflict_count = np.zeros(n_sets, dtype=np.int32)
        family_bitset = 0  # bit i is set if i is in family
        
        for idx in order:
            if time.time() > time_limit:
                break
            if in_family[idx]:
                continue
            if conflict_count[idx] == 0:
                # Add idx to family
                in_family[idx] = True
                family.append(idx)
                family_bitset |= (1 << idx)
                
                # Update conflict counts for all other candidates
                # For each candidate s not in family (s != idx):
                #   new conflicts from pairs (idx, a) for a in family (a != idx)
                #   s conflicts with pair (idx, a) iff inter[idx][a] == inter[idx][s] == inter[a][s]
                for s in range(n_sets):
                    if in_family[s] or s == idx:
                        continue
                    is_val = int(inter_matrix[idx, s])
                    # Count a in family (a != idx) such that inter[idx][a] == is_val AND inter[a][s] == is_val
                    # a must be in inter_by_val[idx][is_val] AND inter_by_val[s][is_val] AND family
                    candidates_idx = inter_by_val[idx].get(is_val, [])
                    candidates_s = inter_by_val[s].get(is_val, [])
                    # Find intersection with family
                    count = 0
                    for a in candidates_idx:
                        if a == idx or a == s:
                            continue
                        if in_family[a] and int(inter_matrix[a, s]) == is_val:
                            count += 1
                    conflict_count[s] += count
        return family
    
    best_family = []
    all_indices = list(range(n_sets))
    
    # Strategy 1: Fast greedy with lexicographic order
    if time.time() < deadline:
        family = greedy_fast(all_indices)
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Strategy 2: Multiple random orders with fast greedy
    for attempt in range(30):
        if time.time() > deadline:
            break
        random.shuffle(all_indices)
        family = greedy_fast(all_indices)
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Strategy 3: Local search - try to add any set that doesn't conflict
    if time.time() < deadline and len(best_family) > 0:
        family_set = set(best_family)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in all_indices:
                if idx in family_set:
                    continue
                if not _forms_sunflower_fast(idx, best_family):
                    best_family.append(idx)
                    family_set.add(idx)
                    improved = True
                    break
    
    # Strategy 4: Swap-based local search
    if time.time() < deadline and len(best_family) > 1:
        family_set = set(best_family)
        swapped = True
        while swapped and time.time() < deadline:
            swapped = False
            for ri in range(len(best_family)):
                if time.time() > deadline:
                    break
                remove_idx = best_family[ri]
                # Try removing remove_idx and see if we can add 2+ new sets
                temp_family = best_family[:ri] + best_family[ri+1:]
                temp_set = set(temp_family)
                added = []
                for idx in all_indices:
                    if idx in temp_set or idx == remove_idx:
                        continue
                    if not _forms_sunflower_fast(idx, temp_family):
                        temp_family.append(idx)
                        temp_set.add(idx)
                        added.append(idx)
                        if len(added) >= 2:
                            break
                if len(added) >= 2:
                    best_family = temp_family
                    family_set = temp_set
                    swapped = True
                else:
                    # Revert (nothing changed since we used temp)
                    pass
    
    return [sorted(all_sets[idx]) for idx in best_family]
# EVOLVE-BLOCK-END
