"""Covering design C(19,7,4): fast bitmask greedy + NumPy local search."""
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets and their coverage masks
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Build masks as numpy arrays of indices (sparse representation)
    # For each k-subset, list of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        idxs = []
        for ts in itertools.combinations(block, t):
            idxs.append(t_subset_index[ts])
        block_t_indices.append(np.array(idxs, dtype=np.int32))

    # Also build dense bitmasks for fast intersection in greedy
    block_masks = []
    for idxs in block_t_indices:
        mask = 0
        for idx in idxs:
            mask |= (1 << idx)
        block_masks.append(mask)

    def run_greedy(seed):
        rng = random.Random(seed)
        unc = (1 << n_t) - 1
        sel = []
        sel_set = set()
        remaining = set(range(n_k))

        while unc:
            if time.time() > deadline - 20.0:
                return None

            best_gain = -1
            best_indices = []

            for i in remaining:
                x = unc & block_masks[i]
                if x == 0:
                    continue
                gain = x.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain:
                    best_indices.append(i)

            if not best_indices:
                break

            chosen = rng.choice(best_indices)
            sel.append(chosen)
            sel_set.add(chosen)
            remaining.discard(chosen)
            unc &= ~block_masks[chosen]

        return sel

    def local_search(block_indices, time_budget):
        """Remove redundant blocks, then try swaps to enable more removals."""
        blocks = list(block_indices)
        n = len(blocks)

        # Coverage count: how many blocks cover each t-subset
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            cover_count[block_t_indices[b]] += 1

        improved = True
        while improved and time.time() < time_budget:
            improved = False

            # Removal pass
            i = 0
            while i < len(blocks) and time.time() < time_budget:
                b = blocks[i]
                t_idx = block_t_indices[b]
                # Check if all t-subsets in this block are covered by >= 2 blocks
                if np.all(cover_count[t_idx] > 1):
                    cover_count[t_idx] -= 1
                    blocks.pop(i)
                    improved = True
                else:
                    i += 1

            # Swap pass: try replacing one block with another to enable removal
            if time.time() < time_budget - 2.0:
                n_blocks_now = len(blocks)
                if n_blocks_now > 0:
                    for _ in range(min(50, n_blocks_now * 3)):
                        if time.time() > time_budget:
                            break
                        i = random.randrange(n_blocks_now)
                        # Pick a candidate block not already in set
                        j = random.randrange(n_k)
                        if j in blocks:
                            continue

                        old = blocks[i]
                        # Tentative swap
                        blocks[i] = j
                        # Update coverage
                        cover_count[block_t_indices[old]] -= 1
                        cover_count[block_t_indices[j]] += 1

                        # Try to remove any block
                        removed = False
                        for r in range(len(blocks)):
                            t_idx = block_t_indices[blocks[r]]
                            if np.all(cover_count[t_idx] > 1):
                                cover_count[t_idx] -= 1
                                blocks.pop(r)
                                removed = True
                                break

                        if removed:
                            improved = True
                            n_blocks_now = len(blocks)
                        else:
                            # Revert
                            cover_count[block_t_indices[j]] -= 1
                            cover_count[block_t_indices[old]] += 1
                            blocks[i] = old

        return blocks

    best_blocks = None
    best_count = float('inf')

    # Run greedy with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 30.0:
            break
        res = run_greedy(seed)
        if res is not None:
            if len(res) < best_count:
                best_blocks = res
                best_count = len(res)

    if best_blocks is None:
        return []

    # Local search on best
    if time.time() < deadline - 5.0:
        improved = local_search(best_blocks, deadline - 2.0)
        if len(improved) < best_count:
            best_blocks = improved
            best_count = len(improved)

    # Convert to lists of ints
    result = [list(all_k_subsets[b]) for b in best_blocks]

    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))

    return dedup
