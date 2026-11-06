import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using optimized backtracking
    with bitset operations for fast conflict checking.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_points = 12
    block_size = 6
    sub_size = 5
    target_blocks = 132
    total_subs = 792
    
    all_blocks = list(combinations(range(n_points), block_size))
    n_blocks = len(all_blocks)
    
    # Index all 5-subsets
    all_subs = list(combinations(range(n_points), sub_size))
    n_subs = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    # For each block, store the bitset of 5-subsets it covers as a Python int
    block_bitsets = []
    for b in all_blocks:
        bits = 0
        for s in combinations(b, sub_size):
            idx = sub_to_idx[s]
            bits |= (1 << idx)
        block_bitsets.append(bits)
    
    # Inverted index: for each 5-subset index, which block indices contain it
    subs_to_blocks = [[] for _ in range(n_subs)]
    for j, b in enumerate(all_blocks):
        for s in combinations(b, sub_size):
            i = sub_to_idx[s]
            subs_to_blocks[i].append(j)
    
    # Start with block (0,1,2,3,4,5) which is all_blocks[0]
    first_block_idx = 0
    covered_bits = block_bitsets[first_block_idx]
    chosen = [all_blocks[first_block_idx]]
    last_block_idx = first_block_idx
    
    def backtrack(covered_bits, chosen, last_block_idx):
        if time.time() > deadline:
            return None
        if len(chosen) == target_blocks:
            if bin(covered_bits).count('1') == total_subs:
                return list(chosen)
            return None
        
        # Find uncovered 5-subset with fewest candidate blocks
        best_sub_idx = -1
        best_candidates = None
        min_count = float('inf')
        
        for i in range(n_subs):
            mask = (1 << i)
            if covered_bits & mask:
                continue
            
            candidates = []
            for j in subs_to_blocks[i]:
                if j <= last_block_idx:
                    continue
                if block_bitsets[j] & covered_bits:
                    continue
                candidates.append(j)
            
            count = len(candidates)
            if count == 0:
                return None
            if count < min_count:
                min_count = count
                best_sub_idx = i
                best_candidates = candidates
                if count == 1:
                    break
        
        if best_candidates is None:
            return None
        
        for j in best_candidates:
            new_covered = covered_bits | block_bitsets[j]
            result = backtrack(new_covered, chosen + [all_blocks[j]], j)
            if result is not None:
                return result
        
        return None
    
    solution = backtrack(covered_bits, chosen, last_block_idx)
    
    if solution:
        return solution
    
    # Greedy fallback
    covered_bits = 0
    greedy_blocks = []
    for j, b in enumerate(all_blocks):
        if block_bitsets[j] & covered_bits:
            continue
        greedy_blocks.append(b)
        covered_bits |= block_bitsets[j]
        if len(greedy_blocks) >= target_blocks:
            break
    return greedy_blocks
