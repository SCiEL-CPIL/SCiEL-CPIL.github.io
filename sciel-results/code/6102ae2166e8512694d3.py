# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using bitset-accelerated greedy
    with randomized restarts and redundant-block removal.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute a Python int bitmask of which t-subsets it covers
    # C(24,3) = 2024 bits, so each mask is a ~253-byte int
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_index[ts])
        block_masks.append(mask)

    # For each t-subset, list of block indices that contain it
    blocks_containing = [[] for _ in range(n_t)]
    for b_idx, mask in enumerate(block_masks):
        # Extract set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            blocks_containing[idx].append(b_idx)
            m ^= lsb

    def greedy_bitset(rng, deadline):
        """
        Bitset-accelerated greedy set cover.
        uncovered is a Python int bitmask of uncovered t-subsets.
        """
        # Initialize: all t-subsets uncovered
        uncovered = (1 << n_t) - 1
        blocks = []
        remaining = n_t

        while remaining > 0:
            if time.time() > deadline:
                break

            # Find the block that maximizes popcount(block_mask & uncovered)
            best_block = -1
            best_gain = 0
            best_mask = 0

            # Randomize order to break ties
            order = list(range(n_blocks))
            rng.shuffle(order)

            for b_idx in order:
                if time.time() > deadline:
                    break
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_mask = block_masks[b_idx]
                elif gain == best_gain and gain > 0 and rng.random() < 0.3:
                    best_block = b_idx
                    best_mask = block_masks[b_idx]

            if best_block == -1 or best_gain <= 0:
                break

            blocks.append(all_blocks[best_block])
            uncovered &= ~best_mask
            remaining = uncovered.bit_count()

        if remaining > 0:
            return None
        return blocks

    def remove_redundant(blocks, deadline):
        """
        Remove redundant blocks: a block is redundant if every t-subset it
        covers is also covered by at least one other block.
        """
        if not blocks:
            return blocks

        # Map block tuples to their index in all_blocks
        block_set = set(blocks)
        n = len(blocks)

        # Compute coverage count per t-subset using bitmasks
        # cov_count[i] = number of blocks covering t-subset i
        cov_count = [0] * n_t
        for b in blocks:
            b_idx = all_blocks.index(b)
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cov_count[idx] += 1
                m ^= lsb

        changed = True
        while changed:
            changed = False
            # Try removing blocks in random order
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return blocks

                b = blocks[i]
                b_idx = all_blocks.index(b)
                mask = block_masks[b_idx]

                # Check if any t-subset is uniquely covered by this block
                m = mask
                redundant = True
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb

                if redundant:
                    blocks.pop(i)
                    # Update coverage counts
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                    changed = True

        return blocks

    # Main loop: multiple randomized restarts
    best_blocks = None
    best_len = float('inf')

    restart_count = 0
    while time.time() < deadline - 15.0:
        restart_count += 1
        rng = random.Random(42 + restart_count)

        blocks = greedy_bitset(rng, deadline)

        if blocks is not None:
            blocks = remove_redundant(blocks, deadline)

            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks

        # Allow more restarts if we're doing well time-wise
        if time.time() > deadline - 20.0:
            break

    # Fallback: deterministic greedy if nothing found
    if best_blocks is None:
        uncovered = (1 << n_t) - 1
        blocks = []
        remaining = n_t

        while remaining > 0:
            if time.time() > deadline:
                break

            best_block = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx

            if best_block == -1 or best_gain <= 0:
                break

            blocks.append(all_blocks[best_block])
            uncovered &= ~block_masks[best_block]
            remaining = uncovered.bit_count()

        if remaining > 0:
            # Should not happen
            return blocks

        best_blocks = blocks

    return best_blocks
# EVOLVE-BLOCK-END
