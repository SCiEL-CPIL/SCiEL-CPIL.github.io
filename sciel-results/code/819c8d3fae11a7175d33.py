import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Precomputation
    # Generate all t-subsets and assign each a compact index 0..N-1
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {}
    t_idx_to_mask = []
    for i, sub in enumerate(all_t_subsets):
        t_sub_to_idx[sub] = i
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_idx_to_mask.append(mask)

    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Precompute for each k-block:
    #   block_cover_mask: a Python int bitmask over t-subset indices (bit i set iff t-sub i is in this block)
    #   block_t_indices: list of t-subset indices covered by this block
    block_cover_mask = [0] * n_k
    block_t_indices = [[] for _ in range(n_k)]
    for bi, block in enumerate(all_k_subsets):
        cm = 0
        tlist = []
        for t_sub in itertools.combinations(block, t):
            ti = t_sub_to_idx[t_sub]
            cm |= (1 << ti)
            tlist.append(ti)
        block_cover_mask[bi] = cm
        block_t_indices[bi] = tlist

    # For each t-subset index, list of block indices that cover it
    t_idx_to_blocks = [[] for _ in range(n_t)]
    for bi in range(n_k):
        for ti in block_t_indices[bi]:
            t_idx_to_blocks[ti].append(bi)

    # 2. Greedy with restarts
    best_blocks = None
    best_count = float('inf')

    # full_mask has all n_t bits set
    full_mask = (1 << n_t) - 1

    while time.time() < deadline:
        uncovered_mask = full_mask
        blocks = []

        while uncovered_mask:
            # Pick a random uncovered t-subset index
            # Find all set bits, pick one randomly
            # For speed, we can just pick the lowest set bit (deterministic) 
            # but random is better for diversity. 
            # Since n_t = C(19,4) = 3876, finding a random set bit is cheap.
            # Use: get lowest set bit, then randomize by shifting? 
            # Simpler: convert to list of set bit positions and pick random.
            # But that's O(n_t) worst case. 
            # Alternative: use int.bit_length() tricks or just min.
            # Let's use: find one set bit via (uncovered_mask & -uncovered_mask) to get lowest,
            # then to randomize, we can rotate or just accept lowest for speed.
            # Given the directive says random target selection improved score, let's do random.
            # Efficient random set bit: 
            #   count = uncovered_mask.bit_count()  # Python 3.10+
            #   r = random.randrange(count)
            #   then find the r-th set bit.
            # This is O(n_t) in worst case for finding r-th set bit.
            # For n_t=3876, this is fine.
            
            # Quick approach: just pick lowest set bit for now (fast), 
            # and rely on block tie-breaking randomization.
            # Actually, let's do a simple random: 
            #   Get a list of candidate positions by scanning? No.
            #   Use: temp = uncovered_mask; r = random.randrange(uncovered_mask.bit_count()); 
            #   while r > 0: temp &= temp - 1; r -= 1; then lowest set bit of temp.
            
            n_uncovered = uncovered_mask.bit_count()
            r = random.randrange(n_uncovered)
            temp = uncovered_mask
            for _ in range(r):
                temp &= temp - 1
            # Now temp has exactly one bit? No, temp has (n_uncovered - r) bits set.
            # We want the r-th set bit. After clearing r bits, the lowest set bit of temp is the r-th.
            target_bit = temp & (-temp)
            # Get index
            target_idx = target_bit.bit_length() - 1

            candidates = t_idx_to_blocks[target_idx]
            if not candidates:
                break

            # Calculate gain for each candidate: count of uncovered bits in block's cover
            max_gain = 0
            best_candidates = []

            for bi in candidates:
                gain = (uncovered_mask & block_cover_mask[bi]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [bi]
                elif gain == max_gain:
                    best_candidates.append(bi)

            if not best_candidates:
                break

            chosen_bi = random.choice(best_candidates)
            blocks.append(all_k_subsets[chosen_bi])
            uncovered_mask &= ~block_cover_mask[chosen_bi]

        if uncovered_mask:
            continue

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered_mask = full_mask
        blocks = []
        while uncovered_mask:
            target_bit = uncovered_mask & (-uncovered_mask)
            target_idx = target_bit.bit_length() - 1
            candidates = t_idx_to_blocks[target_idx]
            max_gain = 0
            best_bi = candidates[0]
            for bi in candidates:
                gain = (uncovered_mask & block_cover_mask[bi]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_bi = bi
            blocks.append(all_k_subsets[best_bi])
            uncovered_mask &= ~block_cover_mask[best_bi]
        return [list(b) for b in blocks]

    return best_blocks
