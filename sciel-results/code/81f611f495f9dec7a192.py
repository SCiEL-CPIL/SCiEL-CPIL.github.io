import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Pre-compute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}

    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Pre-compute bitmask for each block
    block_masks = [0] * num_blocks
    for bi, b in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        block_masks[bi] = mask

    # Pre-compute for each triple, the list of blocks that cover it
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi in range(num_blocks):
        mask = block_masks[bi]
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            triple_to_blocks[idx].append(bi)
            m ^= lb

    def popcount(x):
        return x.bit_count()

    def find_uncovered(uncovered_mask):
        if uncovered_mask == 0:
            return -1
        lb = uncovered_mask & -uncovered_mask
        return lb.bit_length() - 1

    # 1. Initial Construction: Greedy Set Cover
    def greedy_construct(seed):
        random.seed(seed)
        uncovered_mask = (1 << num_triples) - 1
        current_blocks = []

        while uncovered_mask != 0:
            target_idx = find_uncovered(uncovered_mask)
            if target_idx == -1:
                break

            candidates = triple_to_blocks[target_idx]
            best_gain = -1
            best_block_idx = -1
            best_block_mask = 0

            for bi in candidates:
                bm = block_masks[bi]
                gain = popcount(uncovered_mask & bm)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_block_mask = bm
                elif gain == best_gain and random.random() < 0.5:
                    best_block_idx = bi
                    best_block_mask = bm

            if best_block_idx != -1:
                current_blocks.append(best_block_idx)
                uncovered_mask &= ~best_block_mask

        return current_blocks

    # 2. Local Optimization: Remove Redundant Blocks
    def remove_redundant(state, counts):
        if not state:
            return state, counts

        changed = True
        while changed:
            changed = False
            for i in range(len(state) - 1, -1, -1):
                bi = state[i]
                mask = block_masks[bi]
                m = mask
                redundant = True
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lb

                if redundant:
                    m = mask
                    while m:
                        lb = m & -m
                        idx = lb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lb
                    state.pop(i)
                    changed = True
                    break

        return state, counts

    # Helper to compute counts from a state
    def compute_counts(state):
        counts = [0] * num_triples
        for bi in state:
            mask = block_masks[bi]
            m = mask
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                counts[idx] += 1
                m ^= lb
        return counts

    # 3. Iterated Local Search (ILS) with a different kick strategy
    best_blocks = None
    best_len = float('inf')

    # Run several initial greedy restarts to find a good starting point
    for restart in range(15):
        if time.time() > deadline - 30:
            break
        state = greedy_construct(42 + restart)
        counts = compute_counts(state)
        state, counts = remove_redundant(state, counts)

        if len(state) < best_len:
            best_len = len(state)
            best_blocks = state[:]

    # Main ILS Loop
    current_state = best_blocks[:]
    current_counts = compute_counts(current_state)
    current_state, current_counts = remove_redundant(current_state, current_counts)

    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Kick: Replace 1-3 random blocks with new random blocks that share elements
        num_swaps = random.randint(1, 3)
        for _ in range(num_swaps):
            if not current_state:
                break
            idx = random.randint(0, len(current_state) - 1)
            bi_old = current_state[idx]
            old_block = all_blocks[bi_old]

            # Pick a random element from the old block and find a new block containing it
            elem = random.choice(old_block)
            new_block = random.choice(all_blocks)

            # Ensure new block is different
            if new_block == bi_old:
                continue

            current_state[idx] = new_block

        # Recompute counts
        current_counts = compute_counts(current_state)

        # Repair: Add blocks to cover any uncovered triples
        uncovered_mask = 0
        for i in range(num_triples):
            if current_counts[i] == 0:
                uncovered_mask |= (1 << i)

        while uncovered_mask != 0:
            if time.time() > deadline - 5:
                break
            target_idx = find_uncovered(uncovered_mask)
            if target_idx == -1:
                break
            candidates = triple_to_blocks[target_idx]
            best_gain = -1
            best_block_idx = -1
            best_block_mask = 0
            for bi in candidates:
                bm = block_masks[bi]
                gain = popcount(uncovered_mask & bm)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_block_mask = bm
            if best_block_idx != -1:
                current_state.append(best_block_idx)
                m = best_block_mask
                while m:
                    lb = m & -m
                    idx_tri = lb.bit_length() - 1
                    current_counts[idx_tri] += 1
                    m ^= lb
                uncovered_mask &= ~best_block_mask
            else:
                break

        # Optimize: Remove redundancies
        current_state, current_counts = remove_redundant(current_state, current_counts)

        if len(current_state) < best_len:
            best_len = len(current_state)
            best_blocks = current_state[:]
            no_improve = 0
        else:
            no_improve += 1

    # Final cleanup of best_blocks
    final_counts = compute_counts(best_blocks)
    final_blocks, _ = remove_redundant(best_blocks[:], final_counts)

    # Convert indices to actual blocks
    return [list(all_blocks[bi]) for bi in final_blocks]
