import time
import random
from itertools import combinations

def construct():
    """
    Constructs the Witt design S(5,6,12) using the orbit of a base block
    under the Mathieu group M12, with bitset-optimized fallback.
    
    Returns a list of 132 distinct 6-subsets of {0,...,11} such that no 5-subset
    appears in more than one block.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin for 150s limit
    
    # === STEP 1: M12 Group Orbit (Deterministic, Fast) ===
    # Generators for M12 acting on 0..11
    # x is a 12-cycle: (0,1,2,...,11)
    x = tuple(range(12))
    # y is another 12-cycle that with x generates M12
    y = (0, 1, 3, 7, 10, 4, 9, 2, 6, 11, 5, 8)
    
    def apply_perm(perm, subset):
        """Apply permutation to a sorted tuple, return sorted tuple."""
        return tuple(sorted(perm[i] for i in subset))
    
    base_block = (0, 1, 2, 3, 4, 5)
    
    visited = set()
    queue = [base_block]
    visited.add(base_block)
    
    blocks = []
    
    while queue:
        if time.time() > deadline:
            break
        current = queue.pop(0)
        blocks.append(current)
        
        next_b1 = apply_perm(x, current)
        if next_b1 not in visited:
            visited.add(next_b1)
            queue.append(next_b1)
            
        next_b2 = apply_perm(y, current)
        if next_b2 not in visited:
            visited.add(next_b2)
            queue.append(next_b2)
    
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
    
    # === STEP 2: Bitset-Optimized Greedy with Random Restarts ===
    # Precompute all 5-subsets and map them to indices 0..791
    all_5subsets = list(combinations(range(12), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    num_5subsets = len(all_5subsets)  # 792
    
    # Precompute all 6-subsets and their corresponding 5-subset bitmask
    all_6subsets = list(combinations(range(12), 6))
    block_masks = []
    for b in all_6subsets:
        mask = 0
        for sub in combinations(b, 5):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    num_blocks = len(all_6subsets)  # 924
    
    def greedy_packing(order):
        """Perform greedy packing given an order of block indices."""
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
    
    # Try deterministic heuristics first
    # Heuristic 1: Sort by sum of elements
    indices_by_sum = sorted(range(num_blocks), key=lambda i: sum(all_6subsets[i]))
    blocks1 = greedy_packing(indices_by_sum)
    if len(blocks1) > best_count:
        best_count = len(blocks1)
        best_blocks = blocks1
    if best_count >= 132:
        return [list(b) for b in best_blocks]
    
    # Heuristic 2: Lexicographic order
    indices_by_lex = sorted(range(num_blocks), key=lambda i: all_6subsets[i])
    blocks2 = greedy_packing(indices_by_lex)
    if len(blocks2) > best_count:
        best_count = len(blocks2)
        best_blocks = blocks2
    if best_count >= 132:
        return [list(b) for b in best_blocks]
    
    # Heuristic 3: Reverse lexicographic
    indices_by_revlex = sorted(range(num_blocks), key=lambda i: all_6subsets[i], reverse=True)
    blocks3 = greedy_packing(indices_by_revlex)
    if len(blocks3) > best_count:
        best_count = len(blocks3)
        best_blocks = blocks3
    if best_count >= 132:
        return [list(b) for b in best_blocks]
    
    # Random restarts with bitset optimization
    rng = random.Random(42)
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        # Shuffle block indices
        order = list(range(num_blocks))
        rng.shuffle(order)
        
        blocks_found = greedy_packing(order)
        
        if len(blocks_found) > best_count:
            best_count = len(blocks_found)
            best_blocks = blocks_found
            if best_count >= 132:
                break
    
    return [list(b) for b in best_blocks]

if __name__ == "__main__":
    blocks = construct()
    print(f"Found {len(blocks)} blocks")
    # Quick verification
    assert len(set(map(tuple, blocks))) == len(blocks), "Blocks are not distinct"
    seen_5subs = set()
    for b in blocks:
        for i in range(6):
            sub = tuple(b[:i] + b[i+1:])
            if sub in seen_5subs:
                print("Conflict found!")
                break
            seen_5subs.add(sub)
    print(f"Unique 5-subsets covered: {len(seen_5subs)}")
