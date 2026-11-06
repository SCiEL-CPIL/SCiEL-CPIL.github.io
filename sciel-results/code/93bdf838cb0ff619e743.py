import random
import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 50.0  # 50 seconds to be safe
    
    n = 16
    k = 4
    all_sets = [set(c) for c in combinations(range(1, n + 1), k)]
    
    # Precompute bitmasks for faster intersection?
    # 16 elements fit in an integer.
    # Map element 1..16 to bit 0..15
    # Set S -> mask
    # Intersection -> mask & mask
    # Size -> bit_count()
    
    masks = [0] * len(all_sets)
    for i, s in enumerate(all_sets):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        masks[i] = m
        
    best_family = []
    
    # Helper to check if a set can be added
    # family_indices: list of indices into all_sets
    # index_by_core: dict mapping core_mask (0..2^16-1, but only size 0..3) to list of indices in family
    # Actually, we can just iterate family if it's small enough.
    # Let's try the index approach for speed.
    
    def can_add(candidate_idx, family_indices, index_by_core):
        cand_mask = masks[candidate_idx]
        for a_idx in family_indices:
            a_mask = masks[a_idx]
            core_mask = cand_mask & a_mask
            # Petals for A and Candidate
            # P_A = A \ Core
            # P_C = C \ Core
            # Union P = P_A | P_C
            # B must contain Core and be disjoint from P
            # So B & P == 0 and (B & Core) == Core
            
            # Iterate B in index_by_core[core_mask]
            # If core_mask is not in index, no B contains it, so skip?
            # Wait, B must contain Core. So B must be in index_by_core[core_mask].
            
            if core_mask not in index_by_core:
                continue
                
            p_union = (a_mask | cand_mask) ^ core_mask # (A U C) \ Core
            # Actually (A \ Core) U (C \ Core) = (A U C) \ Core
            # Since Core = A & C, A U C = (A \ Core) U (C \ Core) U Core.
            # So (A U C) \ Core is exactly the union of petals.
            
            for b_idx in index_by_core[core_mask]:
                b_mask = masks[b_idx]
                if b_mask & p_union == 0:
                    # Check if B contains Core
                    if (b_mask & core_mask) == core_mask:
                        return False
        return True

    def add_to_index(idx, index_by_core):
        m = masks[idx]
        # Add to all cores? No, we only look up by core = A & C.
        # When checking candidate C against A, core = A & C.
        # We look up B's that contain this core.
        # So for each B in family, we need it to be findable by ANY core it contains?
        # No, the core is determined by A and C.
        # So we need index_by_core[core] to contain B if B contains core.
        # So when adding B, we should add B to index_by_core[K] for all K subset of B with |K| <= 3?
        # That's 1 + 4 + 6 + 4 = 15 entries per set.
        # This is efficient.
        
        # Generate all subsets of B of size 0, 1, 2, 3
        b_set = all_sets[idx]
        for r in range(0, 4):
            for sub in combinations(b_set, r):
                sm = 0
                for x in sub:
                    sm |= (1 << (x - 1))
                if sm not in index_by_core:
                    index_by_core[sm] = []
                index_by_core[sm].append(idx)

    def remove_from_index(idx, index_by_core):
        b_set = all_sets[idx]
        for r in range(0, 4):
            for sub in combinations(b_set, r):
                sm = 0
                for x in sub:
                    sm |= (1 << (x - 1))
                if sm in index_by_core:
                    if idx in index_by_core[sm]:
                        index_by_core[sm].remove(idx)

    # Random restarts
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        indices = list(range(len(all_sets)))
        random.shuffle(indices)
        
        family = []
        index_by_core = {}
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            if can_add(idx, family, index_by_core):
                family.append(idx)
                add_to_index(idx, index_by_core)
                
        if len(family) > len(best_family):
            best_family = family[:]
            
    # Convert best_family indices to sorted lists
    result = [sorted(all_sets[i]) for i in best_family]
    
    # Cap at 250 if necessary?
    # The prompt says "with at most 250 members".
    if len(result) > 250:
        result = result[:250]
        
    return result
