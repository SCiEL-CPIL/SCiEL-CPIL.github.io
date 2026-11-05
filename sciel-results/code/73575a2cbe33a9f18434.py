import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Precompute all t-subsets and map to IDs
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-subsets (blocks) as tuples
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build bitmask for each block: bit i set if block covers triplet i
    block_masks = [0] * n_blocks
    for idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_masks[idx] = mask

    # ---------- Fast Greedy Construction ----------
    def greedy_fast(seed=0):
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_list = list(range(n_triplets))
        pos = np.arange(n_triplets, dtype=np.int32)
        blocks = []

        while uncovered_list:
            if time.time() > deadline - 20:
                break

            # Pick a random uncovered triplet
            idx = rng.randrange(len(uncovered_list))
            start_tid = uncovered_list[idx]
            start_triplet = all_triplets[start_tid]

            block = list(start_triplet)
            block_set = set(block)

            # Greedily add points to maximize coverage of uncovered triplets
            while len(block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]

                # Precompute (t-1)-subsets of current block
                pairs_in_block = list(itertools.combinations(block, t - 1))

                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = triplet_to_id[triplet]
                        if is_uncovered[tid]:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p
                        if max_gain == len(pairs_in_block):
                            break

                if max_gain <= 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(tuple(block))

            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    cur_idx = pos[tid]
                    last_idx = len(uncovered_list) - 1
                    if cur_idx != last_idx:
                        last_tid = uncovered_list[last_idx]
                        uncovered_list[cur_idx] = last_tid
                        pos[last_tid] = cur_idx
                    uncovered_list.pop()

        return blocks

    # ---------- Local Search ----------
    def local_search(blocks, time_limit=10):
        """Redundancy removal, 1-for-1 swaps (cost-aware), and 2-for-1 merges."""
        ls_start = time.time()
        ls_deadline = ls_start + time_limit

        # Precompute triplet-id lists for every possible block (speeds up local search)
        block_triplet_ids = [None] * n_blocks
        for bi, b in enumerate(all_blocks):
            block_triplet_ids[bi] = [triplet_to_id[tp] for tp in itertools.combinations(b, t)]

        # Build count array
        count = np.zeros(n_triplets, dtype=np.int32)

        def add_block_count(b, delta):
            for tid in block_triplet_ids[all_blocks.index(b)]:
                count[tid] += delta

        # Fallback O(1) lookup via dict for current blocks
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}

        def update_counts(block_list, delta):
            for b in block_list:
                for tid in block_triplet_ids[block_to_idx[b]]:
                    count[tid] += delta

        # Initialize counts
        update_counts(blocks, 1)

        # Map from block tuple -> its index in all_blocks (for O(1) access)
        # (already built above as block_to_idx)

        progress = True
        while progress and time.time() < ls_deadline:
            progress = False

            # Pass 1: Remove redundant blocks (using precomputed triplet ids)
            new_blocks = []
            for b in blocks:
                bidx = block_to_idx[b]
                redundant = True
                for tid in block_triplet_ids[bidx]:
                    if count[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    for tid in block_triplet_ids[bidx]:
                        count[tid] -= 1
                    progress = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks

            n = len(blocks)

            # Pass 2: 2-for-1 merge (remove two blocks whose unique triplets are covered by one other block)
            if n >= 3 and time.time() < ls_deadline:
                # Precompute unique triplet ids per block
                uniques = []
                for b in blocks:
                    bidx = block_to_idx[b]
                    uniques.append([tid for tid in block_triplet_ids[bidx] if count[tid] == 1])

                merged = False
                for i in range(n):
                    if time.time() > ls_deadline:
                        break
                    if not uniques[i]:
                        continue
                    for j in range(i + 1, n):
                        if time.time() > ls_deadline:
                            break
                        if not uniques[j]:
                            continue
                        # Union of unique triplets of i and j
                        req = set(uniques[i]) | set(uniques[j])
                        # Find a block k (k != i, k != j) that covers all req
                        found_k = -1
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            kidx = block_to_idx[blocks[k]]
                            kmask = block_masks[kidx]
                            ok = True
                            for tid in req:
                                if not (kmask >> tid) & 1:
                                    ok = False
                                    break
                            if ok:
                                found_k = k
                                break
                        if found_k != -1:
                            # Remove blocks i and j
                            bidx_i = block_to_idx[blocks[i]]
                            bidx_j = block_to_idx[blocks[j]]
                            for tid in block_triplet_ids[bidx_i]:
                                count[tid] -= 1
                            for tid in block_triplet_ids[bidx_j]:
                                count[tid] -= 1
                            # Remove from list (higher index first to keep indices valid)
                            blocks.pop(max(i, j))
                            blocks.pop(min(i, j))
                            progress = True
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    continue

            # Pass 3: 1-for-1 swap with cost-aware candidate selection
            if n >= 1 and time.time() < ls_deadline:
                uniques = []
                for b in blocks:
                    bidx = block_to_idx[b]
                    uniques.append([tid for tid in block_triplet_ids[bidx] if count[tid] == 1])

                swapped = False
                for i in range(n):
                    if time.time() > ls_deadline:
                        break
                    if not uniques[i]:
                        continue

                    req_mask = 0
                    for tid in uniques[i]:
                        req_mask |= (1 << tid)

                    # Score candidates: must cover req_mask; prefer fewer new unique triplets
                    best_idx = -1
                    best_cost = float('inf')
                    for bi in range(n_blocks):
                        if all_blocks[bi] == blocks[i]:
                            continue
                        if (block_masks[bi] & req_mask) != req_mask:
                            continue
                        # Compute cost: number of triplets in candidate that are currently count==0
                        cost = 0
                        for tid in block_triplet_ids[bi]:
                            if count[tid] == 0:
                                cost += 1
                                if cost >= best_cost:
                                    break
                        if cost < best_cost:
                            best_cost = cost
                            best_idx = bi
                            if cost == 0:
                                break

                    if best_idx != -1:
                        old_b = blocks[i]
                        new_b = all_blocks[best_idx]
                        old_idx = block_to_idx[old_b]
                        new_idx = block_to_idx[new_b]
                        for tid in block_triplet_ids[old_idx]:
                            count[tid] -= 1
                        for tid in block_triplet_ids[new_idx]:
                            count[tid] += 1
                        blocks[i] = new_b
                        progress = True
                        swapped = True
                        break
                if swapped:
                    continue

        return blocks

    # ---------- Main ----------
    best_blocks = None
    best_count = float('inf')

    # Run greedy with a few seeds
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        blocks = greedy_fast(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    # Local search on best (with a generous time budget)
    if best_blocks is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            best_blocks = local_search(best_blocks, time_limit=min(remaining, 80))

    return best_blocks if best_blocks is not None else []

if __name__ == "__main__":
    pass
