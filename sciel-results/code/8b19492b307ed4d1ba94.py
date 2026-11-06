import time
import random
from itertools import combinations

def construct():
    # Precompute all 4-subsets of [1, 16]
    n = 16
    k = 4
    all_sets = list(combinations(range(n), k))
    all_masks = [0] * len(all_sets)
    for i, s in enumerate(all_sets):
        mask = 0
        for x in s:
            mask |= (1 << x)
        all_masks[i] = mask
    
    # Map mask to index (for potential use, though we use masks in F)
    # Precompute sunflower_map
    # Key: (mask_a, mask_b) sorted
    # Value: list of masks c such that {a, b, c} is a sunflower
    
    # Helper to get list of b's for given a, i
    def get_sunflower_partners(mask_a, mask_i):
        H = mask_a & mask_i
        h_size = bin(H).count('1')
        U = (mask_a | mask_i) & ~H
        available = ((1 << n) - 1) & ~(H | U)
        partners = []
        
        if h_size == 3:
            # b = H U {c}
            # c in available
            avail_bits = [i for i in range(n) if (available >> i) & 1]
            for c in avail_bits:
                partners.append(H | (1 << c))
        elif h_size == 2:
            # b = H U {c1, c2}
            avail_bits = [i for i in range(n) if (available >> i) & 1]
            for c1, c2 in combinations(avail_bits, 2):
                partners.append(H | (1 << c1) | (1 << c2))
        elif h_size == 1:
            # b = H U {c1, c2, c3}
            avail_bits = [i for i in range(n) if (available >> i) & 1]
            for c1, c2, c3 in combinations(avail_bits, 3):
                partners.append(H | (1 << c1) | (1 << c2) | (1 << c3))
        elif h_size == 0:
            # b disjoint from a U i
            # b is 4-subset of available
            avail_bits = [i for i in range(n) if (available >> i) & 1]
            for c1, c2, c3, c4 in combinations(avail_bits, 4):
                partners.append((1 << c1) | (1 << c2) | (1 << c3) | (1 << c4))
        # h_size > 3 is impossible for distinct sets in this context?
        # If h_size == 4, a == i, but we assume distinct or handle separately.
        # In our check, a is from F, i is candidate. They can be same? No, i not in F.
        
        return partners

    sunflower_map = {}
    # Precompute for all pairs
    # To save time, we can compute on the fly in is_valid if precomputation is too slow
    # But 1820^2 is 3.3M. Let's try to precompute.
    
    # Optimization: Only compute if time permits?
    # 3.3M * 50 ops = 165M ops. Might take 5-10s in Python.
    # Let's do it.
    
    for i in range(len(all_masks)):
        for j in range(i + 1, len(all_masks)):
            m1 = all_masks[i]
            m2 = all_masks[j]
            partners = get_sunflower_partners(m1, m2)
            if partners:
                sunflower_map[(m1, m2)] = partners
                sunflower_map[(m2, m1)] = partners # Symmetric

    def is_valid(mask_idx, F_masks):
        # Check if mask_idx can be added to F_masks
        # F_masks is a set of masks
        for mask_a in F_masks:
            key = (mask_a, mask_idx)
            if key not in sunflower_map:
                # Try reversed
                key = (mask_idx, mask_a)
                if key not in sunflower_map:
                    continue
            partners = sunflower_map[key]
            for mask_b in partners:
                if mask_b in F_masks:
                    return False
        return True

    best_F = set()
    best_size = 0
    deadline = time.time() + 50
    all_masks_list = all_masks
    
    while time.time() < deadline:
        F_masks = set()
        indices = list(range(len(all_masks_list)))
        random.shuffle(indices)
        
        for idx in indices:
            m = all_masks_list[idx]
            if is_valid(m, F_masks):
                F_masks.add(m)
        
        if len(F_masks) > best_size:
            best_size = len(F_masks)
            best_F = set(F_masks)
        
        # Local Search
        # Try to remove one and refill
        while time.time() < deadline - 5:
            if not F_masks:
                break
            to_remove = random.choice(list(F_masks))
            F_masks.remove(to_remove)
            old_size = len(F_masks)
            
            candidates = [m for m in all_masks_list if m not in F_masks]
            random.shuffle(candidates)
            
            for m in candidates:
                if is_valid(m, F_masks):
                    F_masks.add(m)
            
            new_size = len(F_masks)
            
            if new_size > best_size:
                best_size = new_size
                best_F = set(F_masks)
            
            if new_size <= old_size:
                F_masks.add(to_remove)
                break
            
            # If improved, we could continue, but restart is often better
            # Let's restart the outer loop to diversify

    # Convert best_F (masks) to list of sorted lists
    result = []
    for m in best_F:
        s = [i for i in range(n) if (m >> i) & 1]
        result.append(s)
    return result

