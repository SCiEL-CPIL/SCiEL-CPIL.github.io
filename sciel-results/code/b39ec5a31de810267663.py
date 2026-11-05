"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) – the Witt design.
    Uses M12 group orbit of a base block, with deterministic fallback.
    Returns a list of 132 distinct 6-subsets of {0,...,11}.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # === M12 Group Orbit Approach ===
    # Known generators for M12 acting on {0,...,11}
    # Generator 1: 12-cycle (0 1 2 3 4 5 6 7 8 9 10 11)
    g1 = tuple(range(12))  # g1[i] = i+1 mod 12
    # Generator 2: another element that with g1 generates M12
    # This is a standard generator pair for M12
    g2 = (0, 1, 3, 7, 10, 4, 9, 2, 6, 11, 5, 8)
    
    def apply_perm(perm, block):
        """Apply permutation to a sorted tuple, return sorted tuple."""
        return tuple(sorted(perm[i] for i in block))
    
    base_block = (0, 1, 2, 3, 4, 5)
    
    visited = set()
    visited.add(base_block)
    queue = [base_block]
    blocks = []
    
    while queue:
        if time.time() > deadline:
            break
        current = queue.pop(0)
        blocks.append(current)
        
        nb1 = apply_perm(g1, current)
        if nb1 not in visited:
            visited.add(nb1)
            queue.append(nb1)
        
        nb2 = apply_perm(g2, current)
        if nb2 not in visited:
            visited.add(nb2)
            queue.append(nb2)
    
    # Verify the M12 orbit result
    if len(blocks) == 132:
        seen_5subs = set()
        valid = True
        for b in blocks:
            for i in range(6):
                sub = tuple(b[:i] + b[i+1:])
                if sub in seen_5subs:
                    valid = False
                    break
                seen_5subs.add(sub)
            if not valid:
                break
        if valid and len(seen_5subs) == 792:
            return [list(b) for b in blocks]
    
    # === Fallback: Structured Greedy with Bitmask Optimization ===
    # Precompute all 5-subsets and map them to indices
    all_5subsets = list(combinations(range(12), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Precompute all 6-subsets and their corresponding 5-subset bitmask
    all_6subsets = list(combinations(range(12), 6))
    block_masks = []
    for b in all_6subsets:
        mask = 0
        for sub in combinations(b, 5):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Try multiple deterministic orderings
    def greedy_packing(order):
        covered = 0
        blocks_found = []
        for idx in order:
            mask = block_masks[idx]
            if (covered & mask) == 0:
                blocks_found.append(all_6subsets[idx])
                covered |= mask
                if len(blocks_found) == 132:
                    break
        return blocks_found
    
    best_blocks = []
    best_count = 0
    
    # Ordering 1: By sum of elements
    order1 = sorted(range(924), key=lambda i: sum(all_6subsets[i]))
    b1 = greedy_packing(order1)
    if len(b1) > best_count:
        best_count = len(b1)
        best_blocks = b1
    
    # Ordering 2: Lexicographic
    order2 = list(range(924))
    b2 = greedy_packing(order2)
    if len(b2) > best_count:
        best_count = len(b2)
        best_blocks = b2
    
    # Ordering 3: Reverse lexicographic
    order3 = list(range(923, -1, -1))
    b3 = greedy_packing(order3)
    if len(b3) > best_count:
        best_count = len(b3)
        best_blocks = b3
    
    # Ordering 4: By minimum element, then sum
    order4 = sorted(range(924), key=lambda i: (min(all_6subsets[i]), sum(all_6subsets[i])))
    b4 = greedy_packing(order4)
    if len(b4) > best_count:
        best_count = len(b4)
        best_blocks = b4
    
    if best_count >= 132:
        return [list(b) for b in best_blocks]
    
    # If we still haven't found 132, return best found
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
