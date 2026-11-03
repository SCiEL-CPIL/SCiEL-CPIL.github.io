import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    D = 11
    candidates_list = []
    
    # Generate candidates
    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
            
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
                
    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    
    # We assume R^2 = 4 (max norm squared)
    # If we only pick weight 3, R^2=3. But we mix, so R^2=4.
    # Constraint: ||u-v||^2 >= 4
    # u.u + v.v - 2 u.v >= 4
    # 2 u.v <= u.u + v.v - 4
    
    # Precompute dot products
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)
    
    # Conflict condition: 2 * dot[i,j] > norms_sq[i] + norms_sq[j] - 4
    # i.e., dot[i,j] > (norms_sq[i] + norms_sq[j] - 4) / 2
    # Since dot is integer, dot[i,j] >= floor((norms_sq[i] + norms_sq[j] - 4) / 2) + 1
    
    conflict_bitsets = [0] * n
    
    # Vectorized conflict check is hard with bitsets directly, so we do it in chunks or loop
    # N is ~7000, so N^2 is 49M. Looping in Python is slow.
    # We can use NumPy to compute the conflict matrix, then convert to bitsets.
    
    # threshold[i,j] = (norms_sq[i] + norms_sq[j] - 4) / 2
    # Conflict if dot[i,j] > threshold[i,j]
    
    # To save memory and time, we can compute row by row.
    for i in range(n):
        # dot[i, :]
        # threshold for row i
        thresh = (norms_sq[i] + norms_sq - 4) / 2.0
        # conflict_mask is boolean array
        conflict_mask = dot[i, :] > thresh
        conflict_mask[i] = False # No self conflict
        
        # Convert boolean array to integer bitset
        # This is the bottleneck if done naively.
        # We can use np.packbits? No, we need big integers.
        # We can construct the integer by summing 1 << j for each j in conflict_mask.
        # This is O(N) per row, so O(N^2) total. 49M operations.
        # In Python, this might take a few seconds.
        
        # Optimization: use chunks?
        # Or just accept the time if it's < 5s.
        
        # Let's try to speed up bitset construction.
        # indices = np.where(conflict_mask)[0]
        # bitset = 0
        # for j in indices: bitset |= (1 << j)
        
        # Even faster:
        # We can use a lookup table for 8-bit chunks?
        # Or just use the loop. 7000 rows * avg 3000 conflicts = 21M shifts/ors.
        # Python is slow for this.
        
        # Alternative: Keep conflict matrix as numpy array of uint8.
        # But bitsets are much faster for local search.
        
        # Let's try to do it efficiently.
        indices = np.where(conflict_mask)[0]
        if len(indices) > 0:
            # Use a trick: sum of powers of 2
            # But 1 << j for large j is big int.
            # We can build the big int from bytes.
            # However, for N=7000, we need ~875 bytes.
            # We can create a bytearray of size (N+7)//8, set bits, then convert to int.
            
            byte_len = (n + 7) // 8
            b = bytearray(byte_len)
            for j in indices:
                b[j // 8] |= (1 << (j % 8))
            
            # Convert bytearray to int (little endian)
            conflict_bitsets[i] = int.from_bytes(b, byteorder='little')
            
        if time.time() > deadline - 30:
            break

    # If we timed out building conflicts, we might have incomplete data.
    # But 7000*7000/8 is ~6MB. Should be fast enough in C-optimized numpy?
    # The loop above is Python.
    
    # Let's refine the bitset construction to be faster.
    # Actually, N=6820.
    # 6820 / 8 = 853 bytes.
    # 6820 iterations of setting bits.
    # 6820 * 6820 / 8 ~ 5.8M byte operations? No, only for conflicts.
    # Avg degree?
    # Let's assume it's fast enough.
    
    # Local Search with Bitsets
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_min_degree_vertex(mask, conflict_bitsets, n):
        # Find vertex in mask with min degree within mask
        # Iterate through set bits
        best_idx = -1
        best_deg = n + 1
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            # Degree of idx within mask
            deg = popcount(conflict_bitsets[idx] & mask)
            if deg < best_deg:
                best_deg = deg
                best_idx = idx
            m ^= lsb
            if time.time() > deadline - 10:
                return best_idx
        return best_idx

    # Greedy Initialization
    selected = []
    selected_mask = 0
    free_mask = (1 << n) - 1
    
    while free_mask:
        u = get_min_degree_vertex(free_mask, conflict_bitsets, n)
        if u == -1:
            break
        selected.append(u)
        selected_mask |= (1 << u)
        free_mask &= ~(conflict_bitsets[u] | (1 << u))
        
    # Local Search: 1-for-2 Swap
    # Remove u, add v, w
    # v, w must be in (free_mask | conflict_bitsets[u]) & ~selected_mask
    # And v, w must be compatible.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Shuffle selected to randomize
        random.shuffle(selected)
        
        for u in selected:
            if time.time() > deadline - 10:
                break
            # Potential candidates for v, w
            potential_mask = (free_mask | conflict_bitsets[u]) & ~selected_mask
            
            if potential_mask == 0:
                continue
            
            # Find a pair v, w in potential_mask that are compatible
            # Iterate through set bits in potential_mask
            m = potential_mask
            candidates_in_potential = []
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                candidates_in_potential.append(idx)
                m ^= lsb
                if len(candidates_in_potential) > 100: # Limit check
                    break
            
            # Check pairs
            found = False
            for i in range(len(candidates_in_potential)):
                v = candidates_in_potential[i]
                # w must be in potential_mask, not v, and compatible with v
                # compatible with v means conflict_bitsets[v] does not have bit w
                # So w is in potential_mask & ~conflict_bitsets[v] & ~(1<<v)
                w_mask = potential_mask & ~conflict_bitsets[v] & ~(1 << v)
                if w_mask:
                    # Pick any w
                    lsb_w = w_mask & -w_mask
                    w = lsb_w.bit_length() - 1
                    # Perform swap
                    selected.remove(u)
                    selected.append(v)
                    selected.append(w)
                    selected_mask &= ~(1 << u)
                    selected_mask |= (1 << v)
                    selected_mask |= (1 << w)
                    
                    # Update free_mask
                    # free_mask was compatible with all selected.
                    # Now we removed u, added v, w.
                    # New free_mask = (free_mask | conflict_bitsets[u]) & ~conflict_bitsets[v] & ~conflict_bitsets[w] & ~(1<<v) & ~(1<<w)
                    # Wait, free_mask was already ~conflict_bitsets[all selected].
                    # So free_mask & ~conflict_bitsets[v] & ~conflict_bitsets[w] is correct for the new constraints.
                    # And we add back neighbors of u.
                    new_free_mask = (free_mask | conflict_bitsets[u]) & ~conflict_bitsets[v] & ~conflict_bitsets[w]
                    new_free_mask &= ~(1 << v)
                    new_free_mask &= ~(1 << w)
                    # Also remove u from free? u was not in free.
                    # But v, w are now selected, so they are not in free.
                    
                    free_mask = new_free_mask
                    improved = True
                    found = True
                    break
            if found:
                break

    # Final check: try to add any remaining compatible vertices
    # This is handled by the greedy loop if we restart, but we can just scan free_mask
    m = free_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        selected.append(idx)
        selected_mask |= (1 << idx)
        free_mask &= ~(conflict_bitsets[idx] | (1 << idx))
        m ^= lsb
        if time.time() > deadline - 2:
            break

    if not selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(selected, dtype=np.int64)]
    return result.astype(np.int64)

