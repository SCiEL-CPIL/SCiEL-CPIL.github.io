import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Precompute masks for all triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples):
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks[i] = mask

    # Precompute popcounts for all 16-bit masks
    popcounts = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcounts[i] = popcounts[i >> 1] + (i & 1)

    # Precompute supersets for each triple? 
    # No, we will compute covered masks dynamically in calc_alpha.
    # But we can precompute the list of supersets for each triple index to speed up marking.
    # Supersets of a mask M: all masks S such that (S & M) == M.
    # We can generate these on the fly or precompute.
    # Precomputing 560 lists of 8192 ints is ~4.5M ints, ~36MB. Acceptable.
    # But iterating 30 * 8192 is fast enough. Let's just iterate.
    
    def get_supersets(mask):
        # Generate all masks that are supersets of mask
        # Free bits are those not in mask
        free_bits = [(1 << i) for i in range(16) if not (mask & (1 << i))]
        # This is 13 bits. 2^13 = 8192.
        # We can iterate submasks of the complement
        comp = ((1 << 16) - 1) ^ mask
        sub = comp
        while True:
            yield mask | sub
            if sub == 0:
                break
            sub = (sub - 1) & comp

    # Cache supersets? Might be memory heavy if stored as lists. 
    # Let's just use the generator or a precomputed list if needed.
    # Given the small size, precomputing lists is safer for speed.
    superset_lists = []
    for m in triple_masks:
        sups = []
        comp = ((1 << 16) - 1) ^ m
        sub = comp
        while True:
            sups.append(m | sub)
            if sub == 0:
                break
            sub = (sub - 1) & comp
        superset_lists.append(sups)

    def calc_alpha(packing_indices):
        if not packing_indices:
            return 16
            
        covered = [False] * (1 << 16)
        
        for idx in packing_indices:
            for sup in superset_lists[idx]:
                covered[sup] = True
        
        # Find largest mask not covered
        max_alpha = 0
        # Iterate masks in descending order of popcount?
        # Or just iterate all and track max?
        # Since we want the MAX, and we stop at the first one found if we iterate descending?
        # We don't have masks sorted by popcount precomputed globally to avoid sorting overhead each time.
        # But 65536 iterations is trivial.
        for mask in range(1 << 16):
            if not covered[mask]:
                pc = popcounts[mask]
                if pc > max_alpha:
                    max_alpha = pc
                    if max_alpha == 16: # Can't get better than 16
                        break
        return max_alpha

    def is_linear(packing_indices):
        # Check if any two triples share more than 1 point
        n = len(packing_indices)
        for i in range(n):
            m1 = triple_masks[packing_indices[i]]
            for j in range(i + 1, n):
                m2 = triple_masks[packing_indices[j]]
                inter = m1 & m2
                if inter & (inter - 1): # More than one bit set
                    return False
        return True

    def can_add(packing_indices, new_idx):
        new_mask = triple_masks[new_idx]
        for idx in packing_indices:
            inter = new_mask & triple_masks[idx]
            if inter & (inter - 1):
                return False
        return True

    # Local Search
    def local_search(packing_indices):
        current_alpha = calc_alpha(packing_indices)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try adding a random triple
            # Get all addable triples
            addable = [i for i in range(num_triples) if can_add(packing_indices, i)]
            if addable:
                random.shuffle(addable)
                # Try a few
                for i in addable[:5]:
                    new_packing = packing_indices + [i]
                    new_alpha = calc_alpha(new_packing)
                    if new_alpha < current_alpha:
                        packing_indices = new_packing
                        current_alpha = new_alpha
                        improved = True
                        break
                    elif new_alpha == current_alpha and random.random() < 0.1:
                        # Accept neutral move to escape local optima? 
                        # Risky if it leads to worse later, but helps exploration.
                        packing_indices = new_packing
                        improved = True
                        break

            # 2. Try swapping: remove one, add one
            if len(packing_indices) > 0:
                # Pick a random triple to remove
                remove_idx = random.choice(packing_indices)
                temp_packing = [x for x in packing_indices if x != remove_idx]
                
                # Now try to add a triple to temp_packing
                addable = [i for i in range(num_triples) if can_add(temp_packing, i)]
                if addable:
                    random.shuffle(addable)
                    for i in addable[:5]:
                        if i == remove_idx:
                            continue
                        new_packing = temp_packing + [i]
                        new_alpha = calc_alpha(new_packing)
                        if new_alpha < current_alpha:
                            packing_indices = new_packing
                            current_alpha = new_alpha
                            improved = True
                            break

            # 3. Try 2-opt: remove two, add two
            if len(packing_indices) >= 2:
                # Pick two random triples to remove
                r1, r2 = random.sample(packing_indices, 2)
                temp_packing = [x for x in packing_indices if x != r1 and x != r2]
                
                addable = [i for i in range(num_triples) if can_add(temp_packing, i)]
                if addable:
                    random.shuffle(addable)
                    # Try to add 2
                    # This is O(N^2) for the addition check if we do it naively.
                    # We can just try random pairs from addable
                    if len(addable) >= 2:
                        for _ in range(5):
                            a1, a2 = random.sample(addable, 2)
                            if a1 == a2: continue
                            # Check if a1 and a2 are compatible with each other and temp_packing
                            if can_add(temp_packing, a1) and can_add(temp_packing + [a1], a2):
                                new_packing = temp_packing + [a1, a2]
                                new_alpha = calc_alpha(new_packing)
                                if new_alpha < current_alpha:
                                    packing_indices = new_packing
                                    current_alpha = new_alpha
                                    improved = True
                                    break

        return packing_indices, current_alpha

    best_alpha = 16
    best_packing = []
    
    # Main Loop
    while time.time() < deadline:
        # Random Initial Packing
        packing = []
        available = list(range(num_triples))
        random.shuffle(available)
        for i in available:
            if can_add(packing, i):
                packing.append(i)
        
        # Local Search
        improved_packing, alpha = local_search(packing)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_packing = improved_packing
            if best_alpha <= 5:
                break

    if not best_packing:
        # Fallback: just a few disjoint triples
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        
    return [list(all_triples[i]) for i in best_packing]
