"""Rewritten program: faster greedy with vectorized conflict updates."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Generate all 4-element subsets of [1,16]
    all_sets = [frozenset(c) for c in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Convert to bitmasks
    def to_mask(fs):
        m = 0
        for x in fs:
            m |= (1 << (x - 1))
        return m
    
    masks = np.array([to_mask(s) for s in all_sets], dtype=np.uint32)
    
    # Compute pairwise intersection matrix
    # inter_matrix[i,j] = masks[i] & masks[j]
    inter_matrix = np.zeros((n_sets, n_sets), dtype=np.uint32)
    for i in range(n_sets):
        inter_matrix[i, i:] = masks[i] & masks[i:]
        inter_matrix[i:, i] = inter_matrix[i, i:]
    
    def greedy_fast(order, deadline):
        """
        Fast greedy with vectorized conflict updates.
        conflict_count[s] = number of pairs (a,b) in family such that adding s
        would form a sunflower with a and b.
        """
        in_family = np.zeros(n_sets, dtype=bool)
        family = []
        conflict_count = np.zeros(n_sets, dtype=np.int32)
        
        for idx in order:
            if time.time() > deadline:
                break
            if in_family[idx]:
                continue
            if conflict_count[idx] == 0:
                # Add idx to family
                in_family[idx] = True
                family.append(idx)
                
                # Update conflict counts for all other candidates
                # For each candidate s not in family (s != idx):
                #   new conflicts from pairs (idx, a) for a in family (a != idx)
                #   s conflicts with pair (idx, a) iff inter[idx][a] == inter[idx][s] == inter[a][s]
                
                # Vectorized: for each a in family (excluding idx), count how many s
                # have inter[idx][s] == inter[idx][a] AND inter[a][s] == inter[idx][a]
                
                # Get current family members (excluding idx which was just added)
                fam_before = family[:-1]
                if len(fam_before) == 0:
                    continue
                
                # For each a in fam_before, compute the set of s that conflict
                # inter[idx][a] is a fixed mask value m
                # We need s where inter[idx][s] == m AND inter[a][s] == m
                # i.e., inter[idx][s] == inter[idx][a] AND inter[a][s] == inter[idx][a]
                
                # Vectorized approach:
                # For each a, compute a boolean mask over all s
                # conflict_mask_a[s] = (inter[idx,s] == inter[idx,a]) & (inter[a,s] == inter[idx,a])
                
                # But we only care about s not in family
                # Let's build the conflict increment
            
            if time.time() > deadline:
                break
        
        return family
    
    def greedy_fast_v2(order, deadline):
        """
        Faster greedy: maintain for each candidate s a count.
        When adding t, for each a in existing family:
            m = inter[t][a]
            candidates_conflicting = {s : inter[t][s] == m and inter[a][s] == m}
            for s in candidates_conflicting not in family:
                conflict_count[s] += 1
        """
        in_family = np.zeros(n_sets, dtype=bool)
        family = []
        conflict_count = np.zeros(n_sets, dtype=np.int32)
        
        for idx in order:
            if time.time() > deadline:
                break
            if in_family[idx]:
                continue
            if conflict_count[idx] == 0:
                in_family[idx] = True
                family.append(idx)
                
                # Update conflict counts
                # For each a in family before idx was added:
                fam_before = family[:-1]
                if not fam_before:
                    continue
                
                # Vectorized: for each a in fam_before, find all s where
                # inter[idx][s] == inter[idx][a] and inter[a][s] == inter[idx][a]
                # Then increment conflict_count[s] for those s not in family
                
                # Get the intersection values between idx and each a
                # inter_idx_a = inter[idx, a] for a in fam_before
                # We can vectorize over a
                
                # For each a, m_a = inter[idx, a]
                # We want s such that inter[idx, s] == m_a and inter[a, s] == m_a
                # This is equivalent to: inter[idx, s] == inter[a, s] == m_a
                # Which means: (inter[idx, s] == m_a) & (inter[a, s] == m_a)
                
                # Build a vector of m values
                m_values = inter_matrix[idx, fam_before]  # shape (len(fam_before),)
                
                # For each a_j with m_j = m_values[j]:
                #   conflict_mask_j = (inter_matrix[idx, :] == m_j) & (inter_matrix[fam_before[j], :] == m_j)
                #   Then for s not in family: conflict_count[s] += conflict_mask_j[s]
                
                # This is still O(|family| * n_sets) per insertion.
                # Let's try a different vectorization.
                
                # Alternative: for each s, count how many a in fam_before satisfy
                # inter[idx, s] == inter[idx, a] and inter[a, s] == inter[idx, a]
                # Note: inter[idx, a] == inter[a, idx] (symmetric)
                # And inter[idx, s] is fixed for given s.
                # So for fixed s, we need a such that:
                #   inter[idx, a] == inter[idx, s]  AND  inter[a, s] == inter[idx, s]
                # Let m_s = inter[idx, s]
                # Then: inter[idx, a] == m_s AND inter[a, s] == m_s
                
                # For each s, this is a count over a in fam_before.
                # We can vectorize over s:
                #   m_s_vector = inter_matrix[idx, :]  # shape (n_sets,)
                #   For each a in fam_before:
                #       matches_a = (inter_matrix[idx, :] == m_s_vector) & (inter_matrix[fam_before[j], :] == m_s_vector)
                #       conflict_delta += matches_a
                #   But wait, that's not right. For each s, we count a's where
                #   inter[idx,a]==inter[idx,s] and inter[a,s]==inter[idx,s].
                #   So for fixed s, the condition on a is: inter[idx,a]==inter[idx,s] and inter[a,s]==inter[idx,s].
                #   This depends on s through inter[idx,s].
                
                # Let me just do it the straightforward way but with numpy:
                # For each a in fam_before:
                #   m = inter[idx, a]
                #   mask_s = (inter_matrix[idx, :] == m) & (inter_matrix[a, :] == m)
                #   # Only increment for s not in family
                #   mask_s[in_family] = False
                #   mask_s[idx] = False
                #   conflict_count += mask_s.astype(np.int32)
                
                for a in fam_before:
                    m = inter_matrix[idx, a]
                    mask_s = (inter_matrix[idx, :] == m) & (inter_matrix[a, :] == m)
                    mask_s[in_family] = False
                    mask_s[idx] = False
                    conflict_count += mask_s.astype(np.int32)
        
        return family
    
    def greedy_fast_v3(order, deadline):
        """
        Even faster: precompute for each pair (i,j) the list of s that would
        form a sunflower with i and j. Then when adding t, for each a in family,
        just add the precomputed list.
        
        Actually, the sunflower condition for (a, t, s) is:
        inter[a,t] == inter[a,s] == inter[t,s]
        
        For fixed a and t, the set of s that conflict is:
        {s : inter[a,s] == inter[a,t] and inter[t,s] == inter[a,t]}
        
        We can precompute for each pair (a,t) the bitmask of s's.
        But n_sets is 1820, so n_sets^2 is ~3.3M pairs, each with a 1820-bit mask.
        That's 3.3M * 228 bytes ≈ 750MB. Too much.
        
        Alternative: just use the vectorized approach but be smarter.
        """
        in_family = np.zeros(n_sets, dtype=bool)
        family = []
        conflict_count = np.zeros(n_sets, dtype=np.int32)
        
        for idx in order:
            if time.time() > deadline:
                break
            if in_family[idx]:
                continue
            if conflict_count[idx] == 0:
                in_family[idx] = True
                family.append(idx)
                
                fam_before = family[:-1]
                if not fam_before:
                    continue
                
                # Vectorized over all a in fam_before at once
                # For each s, count a in fam_before where:
                #   inter[idx,a] == inter[idx,s] and inter[a,s] == inter[idx,s]
                
                # Let's compute this differently.
                # For fixed idx and s, let m = inter[idx, s].
                # We need count of a in fam_before such that inter[idx,a] == m and inter[a,s] == m.
                
                # Vectorize over s:
                # m_s = inter_matrix[idx, :]  # (n_sets,)
                # For each a in fam_before:
                #   contrib_a = (inter_matrix[idx, :] == m_s) & (inter_matrix[a, :] == m_s)
                #   But this is wrong because m_s depends on s, and we're comparing inter[idx,a] to m_s.
                #   Actually: for fixed a, inter[idx,a] is a constant. 
                #   So: (inter_matrix[idx, a] == m_s) & (inter_matrix[a, :] == m_s)
                #   Wait, inter_matrix[a, :] == m_s means inter[a,s] == m_s for all s.
                #   And inter_matrix[idx, a] == m_s means inter[idx,a] == inter[idx,s].
                
                # So for fixed a:
                #   mask = (inter_matrix[idx, a] == inter_matrix[idx, :]) & (inter_matrix[a, :] == inter_matrix[idx, :])
                #   This gives for each s whether a and s and idx form a sunflower.
                
                # Let's just do it:
                for a in fam_before:
                    # m = inter[idx, a]
                    # For each s: check if inter[idx,s] == m and inter[a,s] == m
                    m = inter_matrix[idx, a]
                    mask_s = (inter_matrix[idx, :] == m) & (inter_matrix[a, :] == m)
                    mask_s[in_family] = False
                    mask_s[idx] = False
                    conflict_count += mask_s.astype(np.int32)
        
        return family
    
    def greedy_vectorized(order, deadline):
        """
        Fully vectorized greedy.
        """
        in_family = np.zeros(n_sets, dtype=bool)
        family = []
        conflict_count = np.zeros(n_sets, dtype=np.int32)
        
        for idx in order:
            if time.time() > deadline:
                break
            if in_family[idx]:
                continue
            if conflict_count[idx] == 0:
                in_family[idx] = True
                family.append(idx)
                
                fam_before = family[:-1]
                if not fam_before:
                    continue
                
                # Vectorize: for each a in fam_before, compute mask of conflicting s
                # and add to conflict_count
                # Do it in one pass using numpy broadcasting if possible
                
                # For all a in fam_before, we want:
                #   for each s: count of a where inter[idx,a]==inter[idx,s] and inter[a,s]==inter[idx,s]
                
                # Let's compute a 2D array: for each a in fam_before and each s:
                #   is_conflict[a_j, s] = (inter[idx, a_j] == inter[idx, s]) & (inter[a_j, s] == inter[idx, s])
                
                # inter_idx_fam = inter_matrix[idx, fam_before]  # (len(fam_before),)
                # inter_idx_all = inter_matrix[idx, :]  # (n_sets,)
                # inter_fam_all = inter_matrix[np.array(fam_before), :]  # (len(fam_before), n_sets)
                
                # is_conflict[j, s] = (inter_idx_fam[j] == inter_idx_all[s]) & (inter_fam_all[j, s] == inter_idx_all[s])
                
                # This is (len(fam_before), n_sets) boolean array.
                # Sum over j to get conflict_delta for each s.
                
                fam_arr = np.array(fam_before)
                inter_idx_fam = inter_matrix[idx, fam_arr]  # (K,)
                inter_idx_all = inter_matrix[idx, :]  # (N,)
                inter_fam_all = inter_matrix[fam_arr, :]  # (K, N)
                
                # For each j, s: (inter_idx_fam[j] == inter_idx_all[s]) & (inter_fam_all[j,s] == inter_idx_all[s])
                # = (inter_idx_fam[:, None] == inter_idx_all[None, :]) & (inter_fam_all == inter_idx_all[None, :])
                
                match1 = (inter_idx_fam[:, None] == inter_idx_all[None, :])  # (K, N)
                match2 = (inter_fam_all == inter_idx_all[None, :])  # (K, N)
                is_conflict = match1 & match2  # (K, N)
                
                # Sum over j
                conflict_delta = is_conflict.sum(axis=0).astype(np.int32)  # (N,)
                
                # Zero out family members and idx
                conflict_delta[in_family] = 0
                conflict_delta[idx] = 0
                
                conflict_count += conflict_delta
        
        return family
    
    all_indices = list(range(n_sets))
    best_family = []
    
    # Strategy 1: Lexicographic greedy
    if time.time() < deadline:
        family = greedy_vectorized(all_indices, deadline)
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Strategy 2: Multiple random restarts
    n_restarts = 50
    for attempt in range(n_restarts):
        if time.time() > deadline:
            break
        random.shuffle(all_indices)
        family = greedy_vectorized(all_indices, deadline)
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Strategy 3: Local search - try to add any compatible set
    if time.time() < deadline and len(best_family) > 0:
        family_set = set(best_family)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in all_indices:
                if idx in family_set:
                    continue
                # Check if idx can be added
                # idx can be added if for all pairs (a,b) in best_family,
                # not (inter[a,b] == inter[a,idx] == inter[b,idx])
                # Equivalent: conflict_count for idx is 0 if we compute it
                # But we don't have the conflict_count from greedy_vectorized
                # Let's just check directly
                can_add = True
                for i in range(len(best_family)):
                    a = best_family[i]
                    ab = inter_matrix[a, idx]
                    for j in range(i+1, len(best_family)):
                        b = best_family[j]
                        if inter_matrix[a,b] == ab and inter_matrix[b,idx] == ab:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    best_family.append(idx)
                    family_set.add(idx)
                    improved = True
                    break
    
    return [sorted(all_sets[idx]) for idx in best_family]
# EVOLVE-BLOCK-END
