import numpy as np
import time
import itertools

def _count_comparable(family_masks, n):
    """Count unordered pairs where one is a strict subset of the other.
    family_masks: array of shape (m,) with integer masks.
    """
    m = len(family_masks)
    if m < 2:
        return 0
    # Build bit matrix: for each pair (i,j), check if i is subset of j or j is subset of i
    # i is subset of j iff (i & ~j) == 0, i.e., all bits in i are in j
    # Strict: also i != j (but since we pick distinct sets, i != j is guaranteed)
    # For distinct masks, i is strict subset of j iff (i & ~j) == 0
    # We need: for pair (i,j), either (i & ~j)==0 or (j & ~i)==0
    # Using numpy: create matrix of masks
    fm = family_masks.reshape(-1, 1)  # (m, 1)
    fmT = family_masks.reshape(1, -1)  # (1, m)
    # i is subset of j: (i & ~j) == 0  =>  i & ~j == 0
    # In numpy: (fm & ~fmT) == 0  gives element-wise
    # But we need integer ops. Let's use:
    # i subset j iff (i | j) == j iff i & ~j == 0
    # Using bitwise: i & ~j == 0
    # ~j in Python is infinite bits, but since we only care about n bits:
    # i & ~j == 0  <=>  i & ((1<<n)-1 ^ j) == 0
    full_mask = (1 << n) - 1
    # i is subset of j: (i & (full_mask ^ j)) == 0
    # j is subset of i: (j & (full_mask ^ i)) == 0
    # For pair (i,j): comparable if (i & (full_mask ^ j)) == 0 or (j & (full_mask ^ i)) == 0
    
    # Vectorized: 
    # A[i,j] = (fm[i] & (full_mask ^ fm[j])) == 0  => i subset j
    # B[i,j] = (fm[j] & (full_mask ^ fm[i])) == 0  => j subset i
    # comparable[i,j] = A[i,j] or B[i,j]
    # But for i==j, both are true (set is subset of itself), but we want strict.
    # Since all masks are distinct, for i!=j, if i subset j then j is NOT subset i (strict).
    # So for i!=j: comparable iff exactly one of A[i,j], B[i,j] is true (or we can just OR them).
    # Actually if i subset j and i != j, then j is not subset i. So A[i,j] XOR B[i,j] for i!=j.
    # But simpler: for i != j, comparable iff A[i,j] or B[i,j].
    
    # Let's compute:
    # A[i,j] = (fm[i] & (full_mask ^ fm[j])) == 0
    # This is equivalent to: (fm[i] | fm[j]) == fm[j]
    
    # Use numpy bitwise ops:
    # We need to compute for all pairs. This is O(m^2) which for m=40 is 1600, fine.
    # But we can do it with broadcasting.
    
    # For each i, j: check if i is subset of j
    # i & ~j == 0  where ~j means complement within n bits
    # i & (full_mask ^ j) == 0
    
    # Broadcasting: fm is (m,1), fmT is (1,m)
    # i_subset_j[i,j] = (fm[i] & (full_mask ^ fm[j])) == 0
    i_subset_j = (fm & (full_mask ^ fmT)) == 0  # (m, m)
    j_subset_i = (fmT & (full_mask ^ fm)) == 0  # (m, m) - this is transpose of above? 
    # Actually j_subset_i[i,j] = (fm[j] & (full_mask ^ fm[i])) == 0
    # Which is the same as i_subset_j[j,i]
    
    # comparable[i,j] for i != j: i_subset_j[i,j] or j_subset_i[i,j]
    # But j_subset_i[i,j] = i_subset_j[j,i]
    # So comparable[i,j] = i_subset_j[i,j] or i_subset_j[j,i]
    
    # For counting unordered pairs, we count for i < j:
    # comparable if i_subset_j[i,j] or i_subset_j[j,i]
    
    # Create upper triangle mask
    idx = np.triu_indices(m, k=1)
    comp_upper = i_subset_j[idx] | i_subset_j[idx[1], idx[0]]
    return int(np.sum(comp_upper))


def _greedy_build(n, m, seed=0, time_limit=50.0):
    """Greedy construction with numpy vectorization."""
    rng = np.random.RandomState(seed)
    
    # All possible masks: 0 to 2^n - 1
    all_masks = np.arange(1 << n, dtype=np.int64)
    full_mask = (1 << n) - 1
    
    # Start with empty set and full set
    family = [0, (1 << n) - 1]
    family_set = set(family)
    
    # Precompute: for each candidate, how many existing family members it's comparable with
    # We'll recompute each step
    
    t_start = time.time()
    
    while len(family) < m:
        if time.time() - t_start > time_limit:
            break
        
        # For each candidate not in family, compute gain
        # Gain = number of family members s.t. candidate is comparable with
        # comparable: (c & ~s) == 0 or (s & ~c) == 0  (within n bits)
        # (c & (full_mask ^ s)) == 0 or (s & (full_mask ^ c)) == 0
        
        cands = all_masks[~np.isin(all_masks, family_set)]
        
        if len(cands) == 0:
            break
        
        # Vectorize: for each candidate c, compute how many family members f are comparable
        # c is comparable with f iff (c & (full_mask ^ f)) == 0 or (f & (full_mask ^ c)) == 0
        
        # Build matrix: rows = candidates, cols = family
        # M[c_idx, f_idx] = (c & (full_mask ^ f)) == 0
        # N[c_idx, f_idx] = (f & (full_mask ^ c)) == 0
        # comparable = M | N
        
        c_mat = cands.reshape(-1, 1)  # (C, 1)
        f_mat = np.array(family, dtype=np.int64).reshape(1, -1)  # (1, F)
        
        M = (c_mat & (full_mask ^ f_mat)) == 0  # (C, F)
        N = (f_mat & (full_mask ^ c_mat)) == 0  # (C, F)
        
        comparable = M | N  # (C, F)
        gains = comparable.sum(axis=1)  # (C,)
        
        # Pick candidate with max gain; tie-break by size (smaller first for diversity?)
        # Actually, let's just pick max gain, then random among ties
        max_gain = gains.max()
        best_indices = np.where(gains == max_gain)[0]
        chosen_idx = best_indices[rng.randint(len(best_indices))]
        
        new_mask = cands[chosen_idx]
        family.append(int(new_mask))
        family_set.add(int(new_mask))
    
    return family


def _local_search(family, n, m, seed=0, time_limit=50.0):
    """Try to improve via swap: replace one set with another to increase comparable count."""
    rng = np.random.RandomState(seed)
    full_mask = (1 << n) - 1
    all_masks = np.arange(1 << n, dtype=np.int64)
    
    t_start = time.time()
    current_score = _count_comparable(np.array(family, dtype=np.int64), n)
    best_score = current_score
    best_family = list(family)
    family_set = set(family)
    
    while time.time() - t_start < time_limit:
        # Try random swaps
        improved = False
        for _ in range(50):
            if time.time() - t_start > time_limit:
                break
            i = rng.randint(m)
            # Pick a candidate not in family
            cands = all_masks[~np.isin(all_masks, family_set)]
            if len(cands) == 0:
                break
            j_idx = rng.randint(len(cands))
            new_mask = int(cands[j_idx])
            
            # Try swap
            old_mask = family[i]
            new_family = list(family)
            new_family[i] = new_mask
            new_set = set(new_family)
            if len(new_set) != m:
                continue
            
            new_score = _count_comparable(np.array(new_family, dtype=np.int64), n)
            if new_score > current_score:
                family = new_family
                family_set = set(family)
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
        
        if not improved:
            break
    
    return best_family, best_score


def construct():
    n = 12
    m = 40
    t_global_start = time.time()
    deadline = t_global_start + 80.0  # Safety margin before 90s kill
    
    best_family = None
    best_score = -1
    
    # Multiple restarts with greedy
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        
        time_per_greedy = min(1.0, (deadline - time.time() - 5.0) / max(1, 50 - trial))
        family = _greedy_build(n, m, seed=trial, time_limit=time_per_greedy)
        
        if len(family) < m:
            # Pad with random distinct sets
            existing = set(family)
            all_masks = np.arange(1 << n, dtype=np.int64)
            cands = all_masks[~np.isin(all_masks, existing)]
            rng = np.random.RandomState(trial + 1000)
            while len(family) < m and len(cands) > 0:
                idx = rng.randint(len(cands))
                family.append(int(cands[idx]))
                existing.add(int(cands[idx]))
                cands = all_masks[~np.isin(all_masks, existing)]
        
        score = _count_comparable(np.array(family, dtype=np.int64), n)
        
        if score > best_score:
            best_score = score
            best_family = list(family)
        
        # Local search polish
        if time.time() < deadline - 5.0:
            ls_time = min(2.0, deadline - time.time() - 3.0)
            polished, polished_score = _local_search(family, n, m, seed=trial + 5000, time_limit=ls_time)
            if polished_score > best_score:
                best_score = polished_score
                best_family = list(polished)
    
    # Final: ensure exactly m distinct sets, convert to list of lists of ints
    if best_family is None:
        # Fallback: first m masks
        best_family = list(range(m))
    
    # Ensure exactly m distinct
    seen = set()
    final = []
    for mask in best_family:
        if mask not in seen:
            seen.add(mask)
            final.append(mask)
    # If we have more than m, trim
    final = final[:m]
    # If we have less, pad
    if len(final) < m:
        all_masks = list(range(1 << n))
        for mask in all_masks:
            if mask not in seen:
                final.append(mask)
                seen.add(mask)
                if len(final) == m:
                    break
    
    # Convert to list of lists of ints
    result = []
    for mask in final:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
