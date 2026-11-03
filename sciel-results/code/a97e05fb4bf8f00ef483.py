import numpy as np
import itertools
import time
from collections import defaultdict

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Convert to bitsets for faster operations
    # N is ~5300, so 5300/64 ~ 83 words.
    # We can use Python integers as bitsets for simplicity and speed in small lists,
    # or numpy arrays.
    # Given N=5300, using Python ints for bitsets is efficient.
    
    conflict_bits = [0] * N
    for i in range(N):
        # Find indices where Conflict[i] is 1
        # This is O(N^2) to build, but N=5300 is small enough.
        # 5300 * 5300 / 2 ~ 14M ops. Fast.
        neighbors = np.where(Conflict[i] == 1)[0]
        bits = 0
        for n in neighbors:
            bits |= (1 << n)
        conflict_bits[i] = bits
        
    best_set = []
    best_size = 0
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        selected_mask_bits = 0
        for i in selected_set:
            selected_mask_bits |= (1 << i)
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng()
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Find candidates j not in selected that conflict with i
                # and do NOT conflict with any other element in selected_set
                
                # Candidates must be neighbors of i
                # And not in selected_set
                # And compatible with selected_set \ {i}
                
                # Get neighbors of i
                # We can iterate through the bits of conflict_bits[i]
                # Or just use the precomputed neighbor list?
                # Let's use the bitset.
                
                # Mask of neighbors of i
                neighbors_i_bits = conflict_bits[i]
                
                # Mask of selected nodes
                # We want j such that:
                # 1. j is in neighbors_i_bits
                # 2. j is NOT in selected_mask_bits
                # 3. j is compatible with (selected_mask_bits ^ (1 << i))
                
                # Let's collect potential j's first
                potential_js = []
                
                # Iterate through set bits in neighbors_i_bits
                # This is efficient if degree is low.
                bits_to_check = neighbors_i_bits & ~selected_mask_bits
                while bits_to_check:
                    # Get lowest set bit
                    lsb = bits_to_check & -bits_to_check
                    j = lsb.bit_length() - 1
                    bits_to_check ^= lsb
                    
                    # Check compatibility with rest of selected
                    # rest_selected_bits = selected_mask_bits ^ (1 << i)
                    # conflict_mask_j = conflict_bits[j]
                    # if conflict_mask_j & rest_selected_bits == 0, then j is compatible
                    if (conflict_bits[j] & (selected_mask_bits ^ (1 << i))) == 0:
                        potential_js.append(j)
                
                if len(potential_js) >= 2:
                    # Look for a pair in potential_js that are compatible
                    found_pair = False
                    # potential_js is small (degree ~56, so at most 56 candidates)
                    for p in range(len(potential_js)):
                        if found_pair: break
                        j1 = potential_js[p]
                        for q in range(p+1, len(potential_js)):
                            j2 = potential_js[q]
                            if (conflict_bits[j1] & (1 << j2)) == 0:
                                # Swap i -> j1, j2
                                selected_set.remove(i)
                                selected_set.add(j1)
                                selected_set.add(j2)
                                
                                # Update selected_mask_bits
                                selected_mask_bits ^= (1 << i)
                                selected_mask_bits |= (1 << j1)
                                selected_mask_bits |= (1 << j2)
                                
                                improved = True
                                found_pair = True
                                break
                    if found_pair:
                        break # Restart loop
                
        return list(selected_set)

    for s in range(200):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
    if not best_set:
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
