import time
import random
from itertools import combinations

def _precompute_conflicts(all_sets):
    n = len(all_sets)
    # Map frozenset to index
    set_to_idx = {frozenset(s): i for i, s in enumerate(all_sets)}
    
    # valid_k_mask[i][j] is a bitmask of all k such that {i, j, k} is a sunflower
    # We only need i < j for storage, but access is symmetric.
    # To save memory/time, we can store as a list of lists of ints.
    # 1820 x 1820 ints is ~3.3M ints. Python ints are 28 bytes? ~90MB. Acceptable.
    # But we can store only for i < j and use symmetry.
    # Or just a list of lists of size n x n.
    
    valid_k_mask = [[0] * n for _ in range(n)]
    
    # Precompute universe mask for [1, 16]
    # We work with 0-indexed elements in sets for bitmask ops, 
    # but the sets in all_sets are 1-indexed.
    # Let's convert all_sets to 0-indexed frozensets for faster intersection?
    # Or just use the existing sets.
    
    for i in range(n):
        A = all_sets[i]
        for j in range(i + 1, n):
            B = all_sets[j]
            
            # Calculate K = A & B
            K = A & B
            k_size = len(K)
            
            # V = Universe - (A | B)
            # Universe is 1..16
            # A | B size is 8 - k_size
            # V size is 16 - (8 - k_size) = 8 + k_size
            # We need to choose (4 - k_size) elements from V
            # V elements are {x in 1..16 | x not in A and x not in B}
            
            # Optimization: if k_size == 4, A == B, skip (but i!=j so A!=B)
            # if k_size == 3, need 1 from V.
            # if k_size == 2, need 2 from V.
            # if k_size == 1, need 3 from V.
            # if k_size == 0, need 4 from V.
            
            # Find V
            # Using set difference is slow.
            # We can iterate 1..16 and check.
            # Or precompute masks for each set?
            # Let's use bitmasks for sets A, B.
            
            # To make this fast, let's precompute bitmasks for all sets.
            # This will be done in construct().
            
            # For now, let's assume we have masks.
            # mask_A, mask_B
            # K_mask = mask_A & mask_B
            # V_mask = full_mask & ~(mask_A | mask_B)
            # Need to find all subsets of V_mask of size 4-k_size.
            # For each such subset W, C = K | W.
            # Check if C is in set_to_idx.
            
            # This inner loop is the bottleneck.
            pass
            
    return valid_k_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Generate all 4-subsets of [1, 16]
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n = len(all_sets)
    
    # Convert to bitmasks for faster intersection logic in precomputation
    # 0-indexed bits: 0..15
    set_masks = []
    set_to_idx = {}
    for s in all_sets:
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        set_masks.append(m)
        set_to_idx[frozenset(s)] = len(set_masks) - 1
        
    full_mask = (1 << 16) - 1
    
    # Precompute valid_k_mask
    # valid_k_mask[i][j] is a bitmask of indices k
    valid_k_mask = [[0] * n for _ in range(n)]
    
    # To speed up generation of subsets of V of specific size,
    # we can precompute all 4-subsets of [0..15] and their masks?
    # No, V varies.
    
    # We can iterate over all possible C (1820 of them) and check if it forms a sunflower with A, B.
    # This is O(N^3) = 1820^3 ~ 6e9. Too slow.
    
    # Back to: For each pair (i, j), find k's.
    # We need to enumerate subsets of V of size 4-k_size.
    # We can use a helper to get bits of a mask.
    
    def get_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        return bits

    # Precompute combinations of bits?
    # For size 1, 2, 3, 4.
    # We can just iterate.
    
    for i in range(n):
        mask_A = set_masks[i]
        for j in range(i + 1, n):
            mask_B = set_masks[j]
            
            K_mask = mask_A & mask_B
            k_size = K_mask.bit_count()
            
            # If k_size == 4, A == B, impossible for distinct i, j.
            # V_mask
            V_mask = full_mask & ~(mask_A | mask_B)
            v_size = V_mask.bit_count()
            
            need = 4 - k_size
            
            # Check if enough elements in V
            if v_size < need:
                continue
                
            # Enumerate all subsets of V_mask of size 'need'
            # And combine with K_mask to form C_mask
            # Then find index of C_mask
            
            # Optimization:
            # If need == 0: C = K. But C must be size 4.
            # If k_size == 4, need 0. C = K = A = B. But i!=j.
            # If k_size < 4, need > 0.
            
            # We can iterate through all sets C (0 to n-1) and check?
            # No, that's O(N) per pair. O(N^3).
            
            # We must enumerate subsets of V.
            # Since need is small (0..4), we can do this.
            
            # Get list of bits in V
            v_bits = get_bits(V_mask)
            
            # Generate combinations of v_bits of size 'need'
            # For each combination, form C_mask = K_mask | (bits of comb)
            # Look up C_mask in set_to_idx?
            # We need a mapping from mask to index.
            mask_to_idx = {set_masks[k]: k for k in range(n)}
            
            if need == 0:
                # C = K
                C_mask = K_mask
                if C_mask in mask_to_idx:
                    k_idx = mask_to_idx[C_mask]
                    if k_idx != i and k_idx != j:
                        valid_k_mask[i][j] |= (1 << k_idx)
                        valid_k_mask[j][i] |= (1 << k_idx)
            elif need == 1:
                for b in v_bits:
                    C_mask = K_mask | (1 << b)
                    if C_mask in mask_to_idx:
                        k_idx = mask_to_idx[C_mask]
                        if k_idx != i and k_idx != j:
                            valid_k_mask[i][j] |= (1 << k_idx)
                            valid_k_mask[j][i] |= (1 << k_idx)
            elif need == 2:
                for p in range(len(v_bits)):
                    for q in range(p + 1, len(v_bits)):
                        C_mask = K_mask | (1 << v_bits[p]) | (1 << v_bits[q])
                        if C_mask in mask_to_idx:
                            k_idx = mask_to_idx[C_mask]
                            if k_idx != i and k_idx != j:
                                valid_k_mask[i][j] |= (1 << k_idx)
                                valid_k_mask[j][i] |= (1 << k_idx)
            elif need == 3:
                for p in range(len(v_bits)):
                    for q in range(p + 1, len(v_bits)):
                        for r in range(q + 1, len(v_bits)):
                            C_mask = K_mask | (1 << v_bits[p]) | (1 << v_bits[q]) | (1 << v_bits[r])
                            if C_mask in mask_to_idx:
                                k_idx = mask_to_idx[C_mask]
                                if k_idx != i and k_idx != j:
                                    valid_k_mask[i][j] |= (1 << k_idx)
                                    valid_k_mask[j][i] |= (1 << k_idx)
            elif need == 4:
                for p in range(len(v_bits)):
                    for q in range(p + 1, len(v_bits)):
                        for r in range(q + 1, len(v_bits)):
                            for s in range(r + 1, len(v_bits)):
                                C_mask = K_mask | (1 << v_bits[p]) | (1 << v_bits[q]) | (1 << v_bits[r]) | (1 << v_bits[s])
                                if C_mask in mask_to_idx:
                                    k_idx = mask_to_idx[C_mask]
                                    if k_idx != i and k_idx != j:
                                        valid_k_mask[i][j] |= (1 << k_idx)
                                        valid_k_mask[j][i] |= (1 << k_idx)
                                        
    if time.time() > deadline:
        return []

    # Simulated Annealing
    best_mask = 0
    best_size = 0
    
    # Initialize with a random valid set? Or empty?
    # Start with empty.
    current_mask = 0
    current_conflicts = 0
    current_size = 0
    
    # We want to maximize size.
    # Energy = -size + M * conflicts.
    # We only care about conflicts == 0 for the final answer, but SA can use intermediate states.
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Number of iterations
    # We have ~60s. Precomp took some time.
    # Let's run until deadline.
    
    while time.time() < deadline - 5:
        
        # Pick random i
        i = random.randint(0, n - 1)
        
        if current_mask & (1 << i):
            # Remove i
            # Count conflicts involving i
            # Iterate j in current_mask
            # For each j, check k in current_mask & valid_k_mask[i][j]
            new_conflicts = 0
            temp_mask = current_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                j = lsb.bit_length() - 1
                temp_mask ^= lsb
                
                if j == i:
                    continue
                    
                k_candidates = current_mask & valid_k_mask[i][j]
                k_candidates &= ~(1 << i)
                k_candidates &= ~(1 << j)
                new_conflicts += k_candidates.bit_count()
            
            # new_conflicts is 2 * (number of conflict triples involving i)
            delta_conflicts = - (new_conflicts // 2)
            delta_size = -1
            
            # Accept if it reduces conflicts or with prob
            # If conflicts > 0, we want to reduce them.
            # If conflicts == 0, we don't want to increase them.
            
            # Simple rule:
            # If current_conflicts > 0:
            #   Accept if delta_conflicts < 0 (reduces conflicts)
            #   Or if delta_conflicts == 0 and delta_size > 0 (impossible here)
            #   Or with prob exp(-(delta_conflicts*M - delta_size)/T)
            
            # Let's use Energy E = conflicts * 100 - size
            # Delta E = delta_conflicts * 100 - delta_size
            delta_E = delta_conflicts * 100 - delta_size
            
            if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                current_mask &= ~(1 << i)
                current_conflicts += delta_conflicts
                current_size += delta_size
                
                if current_conflicts == 0 and current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                    
        else:
            # Add i
            new_conflicts = 0
            temp_mask = current_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                j = lsb.bit_length() - 1
                temp_mask ^= lsb
                
                k_candidates = current_mask & valid_k_mask[i][j]
                k_candidates &= ~(1 << i)
                k_candidates &= ~(1 << j)
                new_conflicts += k_candidates.bit_count()
            
            delta_conflicts = new_conflicts // 2
            delta_size = 1
            
            delta_E = delta_conflicts * 100 - delta_size
            
            if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                current_mask |= (1 << i)
                current_conflicts += delta_conflicts
                current_size += delta_size
                
                if current_conflicts == 0 and current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
        
        T *= cooling
        if T < min_T:
            T = min_T
            # Restart with higher T? Or just keep going.
            # Maybe restart from best_mask if stuck.
            if random.random() < 0.1:
                current_mask = best_mask
                # Recalculate conflicts and size for best_mask?
                # We can just set current_conflicts = 0 if best is valid.
                # And current_size = best_size.
                current_conflicts = 0
                current_size = best_size
                T = 0.5

    # Convert best_mask to list of sets
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(sorted(all_sets[i]))
            
    return result
