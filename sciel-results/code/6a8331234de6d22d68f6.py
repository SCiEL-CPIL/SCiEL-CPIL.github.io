"""Rewritten program: greedy chain init + fast vectorized 2-opt with precomputed comparability matrix."""
import time
import random
import numpy as np

N = 12
M = 40
MAX_MASK = 1 << N  # 4096

# Precompute comparability matrix: comp[i, j] = 1 if i and j are comparable (one strict subset of other)
# i and j are masks 0..4095
# comp[i,j] = 1 iff (i & j) == i and i != j, or (i & j) == j and i != j
_COMP = np.zeros((MAX_MASK, MAX_MASK), dtype=np.int8)
# Vectorized: for each pair, check subset relation
# (i & j) == i means i is subset of j
# We'll do it in chunks to avoid memory issues
# Actually 4096x4096 = 16M entries, int8 = 16MB, fine
# Let's build it:
# For i in range(MAX_MASK): for j in range(MAX_MASK): check
# Vectorized approach:
# i_subset_j[i,j] = (i & j) == i
# j_subset_i[i,j] = (j & i) == j
# comparable = (i_subset_j OR j_subset_i) AND (i != j)

# Build using broadcasting in chunks to save memory
_chunk = 256
for i_start in range(0, MAX_MASK, _chunk):
    i_end = min(i_start + _chunk, MAX_MASK)
    i_arr = np.arange(i_start, i_end, dtype=np.int32)[:, np.newaxis]  # (chunk, 1)
    j_arr = np.arange(MAX_MASK, dtype=np.int32)[np.newaxis, :]        # (1, MAX_MASK)
    ij = i_arr & j_arr  # (chunk, MAX_MASK)
    i_subset_j = (ij == i_arr)  # (chunk, MAX_MASK)
    j_subset_i = (ij == j_arr)  # (chunk, MAX_MASK)
    not_eq = (i_arr != j_arr)   # (chunk, MAX_MASK)
    _COMP[i_start:i_end, :] = ((i_subset_j | j_subset_i) & not_eq).astype(np.int8)

def _calc_score(family_list):
    """Count comparable unordered pairs in a family of masks."""
    m = len(family_list)
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            if _COMP[family_list[i], family_list[j]]:
                count += 1
    return count

def _greedy_chain_init():
    """Build initial family by greedy selection maximizing comparability."""
    family = []
    used = set()
    
    # Start with empty set and full set for structure
    family.append(0)
    used.add(0)
    family.append(MAX_MASK - 1)
    used.add(MAX_MASK - 1)
    
    # Greedily add sets that maximize comparability with existing family
    for _ in range(M - 2):
        best_mask = -1
        best_gain = -1
        # Try all masks, but sample if too slow
        # With 4096 masks and up to 40 existing, each check is O(40), total ~160K ops per addition
        # 38 additions * 160K = ~6M ops, fine
        for s in range(MAX_MASK):
            if s in used:
                continue
            gain = 0
            for t in family:
                if _COMP[s, t]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        
        if best_mask >= 0:
            family.append(best_mask)
            used.add(best_mask)
    
    return family

def _local_search(family, deadline):
    """Fast 2-opt local search using precomputed comparability matrix."""
    m = M
    cur = np.array(family, dtype=np.int32)
    cur_set = set(cur.tolist())
    cur_score = _calc_score(cur.tolist())
    
    best_family = cur.copy()
    best_score = cur_score
    
    # Precompute for each mask in family: its comparability count with other family members
    # comp_count[i] = sum of _COMP[cur[i], cur[j]] for j != i
    def compute_comp_counts(arr):
        counts = np.zeros(m, dtype=np.int32)
        for i in range(m):
            counts[i] = _COMP[arr[i], arr].sum() - (1 if _COMP[arr[i], arr[i]] else 0)
            # Actually _COMP[i,i] = 0 always (not_eq is false), so just sum
            counts[i] = _COMP[arr[i], arr].sum()
        return counts
    
    comp_counts = compute_comp_counts(cur)
    
    no_improve = 0
    
    while time.time() < deadline:
        best_delta = 0
        best_idx = -1
        best_new = -1
        
        for idx in range(m):
            if time.time() >= deadline:
                break
            old = int(cur[idx])
            old_comp = int(comp_counts[idx])
            
            # For each candidate new mask, compute delta
            # delta = new_comp_with_others - old_comp
            # new_comp_with_others = sum of _COMP[new, cur[j]] for j != idx
            # We can compute this as: _COMP[new, cur].sum() - _COMP[new, old] (since j != idx excludes old)
            # But _COMP[new, old] might be 0 or 1
            # Actually: sum over j != idx of _COMP[new, cur[j]] = _COMP[new, cur].sum() - _COMP[new, cur[idx]]
            # = _COMP[new, cur].sum() - _COMP[new, old]
            
            # Vectorized: compute for all candidates at once
            # _COMP is (4096, 4096), cur is (40,)
            # We want _COMP[:, cur] which is (4096, 40) - sum along axis 1 gives (4096,)
            # This is 4096*40 = 163K lookups, very fast
            
            # Get comparability of all masks with each family member
            # comp_all = _COMP[:, cur]  # (4096, 40)
            # comp_sum = comp_all.sum(axis=1)  # (4096,)
            # For candidate s: new_comp = comp_sum[s] - _COMP[s, old]
            
            comp_with_family = _COMP[:, cur]  # (4096, 40)
            comp_sum = comp_with_family.sum(axis=1).astype(np.int32)  # (4096,)
            
            # For each candidate s, delta = (comp_sum[s] - _COMP[s, old]) - old_comp
            # But we need to exclude s == old (no change) and s in cur_set (not allowed)
            
            deltas = comp_sum - _COMP[:, old].astype(np.int32) - old_comp  # (4096,)
            
            # Mask out invalid candidates
            # Invalid: s in cur_set (including old)
            for s in cur_set:
                deltas[s] = -999999
            
            max_idx = np.argmax(deltas)
            max_delta = int(deltas[max_idx])
            
            if max_delta > best_delta:
                best_delta = max_delta
                best_idx = idx
                best_new = int(max_idx)
        
        if best_delta > 0 and best_idx >= 0:
            old = int(cur[best_idx])
            cur_set.discard(old)
            cur_set.add(best_new)
            cur[best_idx] = best_new
            cur_score += best_delta
            no_improve = 0
            
            # Recompute comp_counts
            comp_counts = compute_comp_counts(cur)
            
            if cur_score > best_score:
                best_score = cur_score
                best_family = cur.copy()
        else:
            no_improve += 1
            if no_improve >= 5:
                # Random perturbation: swap 2-3 random elements
                for _ in range(30):
                    if time.time() >= deadline:
                        break
                    idx = random.randint(0, m - 1)
                    new_s = random.randint(0, MAX_MASK - 1)
                    if new_s in cur_set:
                        continue
                    old = int(cur[idx])
                    
                    # Compute delta
                    old_comp = int(comp_counts[idx])
                    new_comp = int(_COMP[new_s, cur].sum() - _COMP[new_s, old])
                    delta = new_comp - old_comp
                    
                    if delta >= 0 or random.random() < 0.2:
                        cur_set.discard(old)
                        cur_set.add(new_s)
                        cur[idx] = new_s
                        cur_score += delta
                        comp_counts = compute_comp_counts(cur)
                        if cur_score > best_score:
                            best_score = cur_score
                            best_family = cur.copy()
                        break
                no_improve = 0
    
    return best_family.tolist(), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (evaluator kills at 90s)
    
    # Initial family
    family = _greedy_chain_init()
    
    # Local search
    best_family, best_score = _local_search(family, deadline)
    
    # Convert back to lists of ints
    result = []
    for s in best_family:
        lst = []
        for b in range(N):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    
    return result
