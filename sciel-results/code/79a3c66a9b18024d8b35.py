import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    # Map t-subsets to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    full_mask = (1 << n_tsubs) - 1
    
    # Generate all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Precompute masks for each k-subset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask
        
    # Convert to numpy array for faster bitwise operations if possible? 
    # Python ints are arbitrary precision, so bitwise ops on large ints are fast.
    # numpy uint64 would limit us to 64 t-subsets, which is far too small (C(19,4)=3876).
    # So we stick with Python ints.
    
    def get_union_mask(indices):
        union = 0
        for idx in indices:
            union |= ksub_masks[idx]
        return union

    def is_valid_cover(selected_indices):
        return get_union_mask(selected_indices) == full_mask

    # 2. Greedy Initialization
    selected_indices = []
    uncovered = full_mask
    
    while uncovered and time.time() < deadline - 10.0:
        best_gain = 0
        best_idx = -1
        for j in range(n_ksubs):
            if j in selected_indices:
                continue
            gain = (uncovered & ksub_masks[j]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~ksub_masks[best_idx]
        
        if uncovered == 0:
            break
            
    if not selected_indices:
        return []
        
    # Ensure we have a valid cover. If greedy didn't finish, try to fix or return best effort.
    # But greedy should always finish for this problem size.
    
    best_solution = selected_indices[:]
    best_size = len(best_solution)
    
    # 3. Local Search: redundancy removal + 1-for-1 and 2-for-2 swaps
    # Use coverage counts to quickly detect redundant blocks.
    
    # Helper: compute coverage counts for a list of block indices
    def coverage_counts(indices):
        # counts[tsub] = number of selected blocks covering that t-subset
        counts = [0] * n_tsubs
        for idx in indices:
            mask = ksub_masks[idx]
            # iterate set bits
            while mask:
                lsb = mask & -mask
                bit = lsb.bit_length() - 1
                counts[bit] += 1
                mask ^= lsb
        return counts
    
    # Helper: check if any t-subset has count 0
    def is_covered(counts):
        # fast check: all counts > 0
        return all(c > 0 for c in counts)
    
    # Initialize counts
    counts = coverage_counts(selected_indices)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # 3a. Remove redundant blocks using counts
        # A block is redundant if for every t-subset it covers, count[tsub] > 1.
        # We do a single pass; if any removed, loop again.
        new_selected = []
        any_removed = False
        for idx in selected_indices:
            mask = ksub_masks[idx]
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                bit = lsb.bit_length() - 1
                if counts[bit] == 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                # Remove this block: decrement counts for all tsubs it covers
                m = mask
                while m:
                    lsb = m & -m
                    bit = lsb.bit_length() - 1
                    counts[bit] -= 1
                    m ^= lsb
                any_removed = True
            else:
                new_selected.append(idx)
        
        if any_removed:
            selected_indices = new_selected
            improved = True
            if len(selected_indices) < best_size:
                best_size = len(selected_indices)
                best_solution = selected_indices[:]
            continue  # restart loop to check for more redundancies
        
        # 3b. Try 1-for-1 swaps
        # For each block i, compute what is uniquely covered (count==1 for tsubs only in i)
        # Then find a block j not in selected that covers all those unique tsubs.
        selected_set = set(selected_indices)
        swapped = False
        
        for i, idx_out in enumerate(selected_indices):
            if time.time() > deadline - 5.0:
                break
            
            # Find tsubs uniquely covered by idx_out
            unique_mask = 0
            m = ksub_masks[idx_out]
            while m:
                lsb = m & -m
                bit = lsb.bit_length() - 1
                if counts[bit] == 1:
                    unique_mask |= lsb
                m ^= lsb
            
            if unique_mask == 0:
                # Should have been removed in 3a
                continue
            
            # Find replacement block j that covers all unique tsubs
            found_swap = False
            for j in range(n_ksubs):
                if j in selected_set:
                    continue
                if (ksub_masks[j] & unique_mask) == unique_mask:
                    # Perform swap: remove idx_out, add j
                    # Update counts: decrement for idx_out, increment for j
                    m_out = ksub_masks[idx_out]
                    while m_out:
                        lsb = m_out & -m_out
                        bit = lsb.bit_length() - 1
                        counts[bit] -= 1
                        m_out ^= lsb
                    
                    m_in = ksub_masks[j]
                    while m_in:
                        lsb = m_in & -m_in
                        bit = lsb.bit_length() - 1
                        counts[bit] += 1
                        m_in ^= lsb
                    
                    selected_indices[i] = j
                    selected_set.remove(idx_out)
                    selected_set.add(j)
                    swapped = True
                    found_swap = True
                    break
            
            if found_swap:
                improved = True
                break  # restart outer loop
        
        if swapped:
            continue
        
        # 3c. Try 2-for-2 swaps
        # For each pair (i, j) in selected, compute tsubs uniquely covered by the pair
        # (count==1 for tsubs in i or j, but not in any other block)
        # Then find two blocks (a, b) not in selected that together cover all those unique tsubs.
        # This is more expensive, so only try if time permits.
        
        n_sel = len(selected_indices)
        if n_sel >= 2 and time.time() < deadline - 10.0:
            # Precompute unique masks for each block
            unique_masks = []
            for idx in selected_indices:
                um = 0
                m = ksub_masks[idx]
                while m:
                    lsb = m & -m
                    bit = lsb.bit_length() - 1
                    if counts[bit] == 1:
                        um |= lsb
                    m ^= lsb
                unique_masks.append(um)
            
            # For each pair, compute combined unique mask
            # unique_pair = (unique_masks[i] | unique_masks[j]) & ~(unique of others)
            # Actually, a tsub is uniquely covered by the pair if:
            # - it is covered by i or j
            # - it is not covered by any other selected block
            # - count[tsub] == 1 or 2 (if 2, both i and j cover it)
            
            # Simpler: for pair (i,j), the tsubs that would be uncovered if we remove both i and j
            # are those with count==1 and covered by i or j, OR count==2 and covered only by i and j.
            # But we can compute: union_others = union of all blocks except i,j
            # missing = full_mask & ~union_others
            # Then we need two new blocks a,b such that (mask[a] | mask[b]) covers missing.
            
            # To save time, only try pairs where the missing set is small.
            
            found_2swap = False
            # Limit pairs to first few to avoid O(n^2) blowup
            max_pairs = min(n_sel * (n_sel - 1) // 2, 50)
            pair_count = 0
            
            for i in range(n_sel):
                if found_2swap or pair_count >= max_pairs:
                    break
                for j in range(i + 1, n_sel):
                    if pair_count >= max_pairs:
                        break
                    pair_count += 1
                    
                    if time.time() > deadline - 10.0:
                        break
                    
                    # Compute union of all blocks except i and j
                    union_others = 0
                    for idx in selected_indices:
                        # skip i and j
                        pass  # We need indices, not values
                    
                    # Better: use counts? If count[tsub] > 2, it's covered by others.
                    # If count[tsub] == 2, it might be only i and j.
                    # If count[tsub] == 1, it's only one of i or j (or neither, but then not in union).
                    
                    # Compute missing: tsubs with count <= 2 that are covered by i or j,
                    # and not covered by any other block.
                    # A tsub is covered by i or j if (mask[i] | mask[j]) has the bit.
                    # It is covered by others if count[tsub] > (number of {i,j} that cover it).
                    
                    # Simpler: compute union_others by OR-ing all masks except i,j.
                    # This is O(n_sel) per pair, with n_sel~150, max_pairs=50 -> 7500 OR ops, fine.
                    
                    union_others = 0
                    for idx in selected_indices:
                        if idx != selected_indices[i] and idx != selected_indices[j]:
                            union_others |= ksub_masks[idx]
                    
                    missing = full_mask & ~union_others
                    
                    if missing == 0:
                        # Both i and j are redundant together? Should have been caught.
                        continue
                    
                    # Find two blocks a,b not in selected that cover missing.
                    # We can iterate a, then for each a, find b that covers missing & ~mask[a].
                    # But this is O(n_ksubs^2) which is 5011^2 = 25M, too slow.
                    # Instead, for each a, compute remaining = missing & ~mask[a].
                    # Then find b such that (mask[b] & remaining) == remaining.
                    # We can precompute for each tsub the list of blocks covering it? Too much memory.
                    # Alternative: iterate a, and for each a, iterate b only if remaining is small.
                    
                    # Given time constraints, skip 2-for-2 for now.
                    # We could add a limited search: try first 100 blocks for a, and for each a, try first 100 for b.
                    
                    # Let's do a limited search:
                    found = False
                    # Sample some candidate blocks
                    candidates = list(range(n_ksubs))
                    random.shuffle(candidates)
                    candidates = candidates[:100]
                    
                    for a in candidates:
                        if a in selected_set:
                            continue
                        remaining = missing & ~ksub_masks[a]
                        if remaining == 0:
                            # a alone covers missing, so we can do 1-for-2? But we want 2-for-2.
                            # Actually, if a covers missing, we can replace i,j with a (1 block for 2), reducing size.
                            # But that's a 1-for-2 removal, which is even better.
                            # Perform: remove i and j, add a.
                            # Update counts.
                            m_i = ksub_masks[selected_indices[i]]
                            while m_i:
                                lsb = m_i & -m_i
                                bit = lsb.bit_length() - 1
                                counts[bit] -= 1
                                m_i ^= lsb
                            
                            m_j = ksub_masks[selected_indices[j]]
                            while m_j:
                                lsb = m_j & -m_j
                                bit = lsb.bit_length() - 1
                                counts[bit] -= 1
                                m_j ^= lsb
                            
                            m_a = ksub_masks[a]
                            while m_a:
                                lsb = m_a & -m_a
                                bit = lsb.bit_length() - 1
                                counts[bit] += 1
                                m_a ^= lsb
                            
                            # Remove i and j from selected_indices (in reverse order to preserve indices)
                            selected_indices.pop(j)
                            selected_indices.pop(i)
                            selected_indices.append(a)
                            
                            selected_set.remove(selected_indices[i] if i < len(selected_indices) else selected_indices[-1])
                            # This is getting messy with indices. Let's rebuild selected_set.
                            selected_set = set(selected_indices)
                            
                            improved = True
                            found_2swap = True
                            found = True
                            break
                        
                        # Find b that covers remaining
                        for b in candidates:
                            if b == a or b in selected_set:
                                continue
                            if (ksub_masks[b] & remaining) == remaining:
                                # Found a and b
                                # Remove i and j, add a and b
                                m_i = ksub_masks[selected_indices[i]]
                                while m_i:
                                    lsb = m_i & -m_i
                                    bit = lsb.bit_length() - 1
                                    counts[bit] -= 1
                                    m_i ^= lsb
                                
                                m_j = ksub_masks[selected_indices[j]]
                                while m_j:
                                    lsb = m_j & -m_j
                                    bit = lsb.bit_length() - 1
                                    counts[bit] -= 1
                                    m_j ^= lsb
                                
                                m_a = ksub_masks[a]
                                while m_a:
                                    lsb = m_a & -m_a
                                    bit = lsb.bit_length() - 1
                                    counts[bit] += 1
                                    m_a ^= lsb
                                
                                m_b = ksub_masks[b]
                                while m_b:
                                    lsb = m_b & -m_b
                                    bit = lsb.bit_length() - 1
                                    counts[bit] += 1
                                    m_b ^= lsb
                                
                                # Remove i and j, add a and b
                                selected_indices.pop(j)
                                selected_indices.pop(i)
                                selected_indices.append(a)
                                selected_indices.append(b)
                                
                                selected_set = set(selected_indices)
                                
                                improved = True
                                found_2swap = True
                                found = True
                                break
                        
                        if found:
                            break
                    
                    if found_2swap:
                        break
        
        if not improved:
            break

    # Final check
    if is_valid_cover(selected_indices) and len(selected_indices) < best_size:
        best_solution = selected_indices[:]
        
    return [list(all_ksubs[idx]) for idx in best_solution]
