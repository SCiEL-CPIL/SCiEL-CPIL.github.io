import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the bitmask of t-subsets it covers
    # and the list of t-subset indices
    k_sub_t_idxs = []
    block_masks = []
    for k_sub in all_k_subs:
        t_idxs = []
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[t_sub]
            t_idxs.append(idx)
            mask |= (1 << idx)
        k_sub_t_idxs.append(t_idxs)
        block_masks.append(mask)

    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed):
        """Greedy set cover with bitmask acceleration."""
        rng = random.Random(seed)
        # Initialize uncovered mask: all bits set to 1
        uncovered = (1 << n_t) - 1
        blocks_idx = []
        used_blocks = set()
        
        # To speed up selection of the "most constrained" uncovered t-subset,
        # we can just pick a random one from the uncovered set, or the first one.
        # Finding the first set bit in a large int is fast.
        # But picking a random uncovered t-subset might lead to better diversification.
        
        while uncovered:
            # Pick an uncovered t-subset
            # Method: Get list of set bits? Too slow.
            # Method: Pick a random index and check if it's set?
            # Method: Find the index of the least significant set bit?
            # Let's find a random uncovered t-subset efficiently.
            # We can maintain a list of uncovered indices?
            # Or just pick a random t_sub_idx and check if (uncovered >> idx) & 1.
            # If n_t is 11628, and we have ~300 blocks, the number of iterations is ~300.
            # Checking a random bit is O(1).
            
            # Heuristic: Pick the uncovered t-subset with the fewest candidates?
            # That requires scanning all uncovered bits.
            # Let's stick to: Pick a random uncovered t-subset.
            # To ensure we don't loop forever if we pick a covered one, we retry.
            # Or better: Maintain a list of uncovered indices.
            
            # For simplicity and speed in Python, let's use the "first uncovered" or "random" approach.
            # Given the bitmask, finding the index of the first set bit is:
            # lsb = uncovered & -uncovered
            # idx = lsb.bit_length() - 1
            
            # However, "most constrained" is better.
            # Let's iterate through a few random candidates and pick the one with min candidates?
            # No, let's just pick the first uncovered t-subset (LSB) for determinism in seed 0,
            # and random for others.
            
            if seed == 0:
                lsb = uncovered & -uncovered
                s_idx = lsb.bit_length() - 1
            else:
                # Pick a random t-subset index, ensure it's uncovered
                while True:
                    s_idx = rng.randint(0, n_t - 1)
                    if (uncovered >> s_idx) & 1:
                        break

            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1

            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                # Calculate gain using bitmask
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx

            if best_block_idx == -1:
                # This should not happen if the problem is solvable and we haven't used all blocks
                break

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)
            
            # Update uncovered
            uncovered &= ~block_masks[best_block_idx]

        return blocks_idx

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass using NumPy counts."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            # Vectorized increment
            cover_count[k_sub_t_idxs[b_idx]] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                # Check if any t-subset in this block has cover_count == 1
                # We need to check the counts for the t-subsets covered by this block
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if np.any(counts_for_block == 1):
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    # Decrement counts
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search(blocks_idx, max_iter=50):
        """
        Local search: Try to remove blocks.
        Since reduce_blocks already removes all immediately redundant blocks,
        this function is essentially the same as reduce_blocks if it just removes.
        To do better, we could try swapping.
        However, given the time constraints, a strong reduce_blocks is often sufficient
        when combined with multiple random restarts.
        """
        return reduce_blocks(blocks_idx)

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 2. Randomized Greedy Restarts
    # With bitmask speedup, we can afford many more restarts
    seeds = list(range(1, 101)) # Try 100 seeds
    for seed in seeds:
        if time.time() >= deadline - 10.0:
            break
        blocks_idx = greedy_cover(seed)
        blocks_idx = reduce_blocks(blocks_idx)
        if blocks_idx and len(blocks_idx) < best_len:
            best_blocks = blocks_idx
            best_len = len(blocks_idx)

    # 3. Final Reduction Pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
