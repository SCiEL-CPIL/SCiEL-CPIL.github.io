import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # 1. Generate all 4-subsets of [1, 16]
    # Represent elements as 0..15
    all_sets = []
    index_map = {}
    for combo in itertools.combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        index_map[mask] = len(all_sets) - 1
        
    n_sets = len(all_sets)
    
    # Helper to get forbidden indices for a pair of masks
    def get_forbidden(mask_a, mask_c):
        K = mask_a & mask_c
        U = 0xFFFF ^ (mask_a | mask_c)
        need = 4 - bin(K).count('1')
        if need < 0 or need > bin(U).count('1'):
            return []
        
        # Generate combinations of 'need' bits from U
        # Extract indices of set bits in U
        u_bits = []
        u_temp = U
        while u_temp:
            lsb = u_temp & -u_temp
            idx = lsb.bit_length() - 1
            u_bits.append(idx)
            u_temp ^= lsb
            
        forbidden = []
        for combo in itertools.combinations(u_bits, need):
            p = 0
            for b in combo:
                p |= (1 << b)
            mask_k = K | p
            # mask_k must be a valid 4-set, which it is by construction
            forbidden.append(index_map[mask_k])
        return forbidden

    best_family_size = 0
    best_family = []
    
    while time.time() < deadline:
        # Shuffle
        indices = list(range(n_sets))
        random.shuffle(indices)
        
        F = []
        F_set = set() # For O(1) membership check if needed, but we use Blocked
        Blocked = set()
        
        for c in indices:
            if c in Blocked:
                continue
            
            # Add c to F
            F.append(c)
            F_set.add(c)
            
            # Update Blocked
            # For each a in F (a != c), add forbidden(a, c) to Blocked
            # Note: We should not block c itself? 
            # The forbidden set for (a, c) contains k such that {a, c, k} is sunflower.
            # k can be a or c?
            # If k=a, {a, c, a} is not 3 distinct sets. 
            # The problem says "No three distinct sets".
            # So if k==a or k==c, it doesn't form a valid sunflower triple of *distinct* sets.
            # However, if we block a, we might prevent adding a later? 
            # But a is already in F.
            # If we block a, it just means we can't add a again (which we can't anyway).
            # But what if blocking a prevents adding some other set d that is not in F?
            # No, Blocked is just a set of indices we can't add.
            # If a is in F, we don't need to add it.
            # But if we block a, and later we want to remove a and re-add? 
            # We are doing greedy, so we don't remove.
            # So it's safe to block a and c as well, as they are already in F.
            
            for a in F:
                if a == c:
                    continue
                forb = get_forbidden(all_sets[a], all_sets[c])
                Blocked.update(forb)
                
        if len(F) > best_family_size:
            best_family_size = len(F)
            best_family = F.copy()
            
    # Return the best family found
    # Convert indices back to sorted lists of elements 1..16
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
