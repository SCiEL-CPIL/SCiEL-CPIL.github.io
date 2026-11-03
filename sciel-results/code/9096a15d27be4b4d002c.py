import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    # Deadline with safety margin
    deadline = time.time() + 112

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute for each point x, the list of t-subsets containing x
    # This helps with gain calculation
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_tsubs[x].append(i)

    # Precompute t-subsets for each possible (t-1)-subset
    # For fast gain: gain of adding x to block B = number of uncovered t-subsets
    # of form R U {x} where R is a (t-1)-subset of B
    # We can precompute for each (t-1)-subset R, the list of t-subsets R U {x}
    # Actually, let's just compute gain directly but efficiently

    # For each (t-1)-subset, store the t-subsets it can form with each additional point
    tminus1_to_tsubs = {}
    for ts in all_tsubs:
        for i in range(t):
            r = tuple(sorted(ts[:i] + ts[i+1:]))
            if r not in tminus1_to_tsubs:
                tminus1_to_tsubs[r] = {}
            tminus1_to_tsubs[r][ts[i]] = ts

    def _compute_gain(block, uncovered_set):
        """Compute the gain of each candidate point for extending block"""
        block_set = set(block)
        gains = {}
        # For each (t-1)-subset of block, for each point x not in block,
        # check if R U {x} is a t-subset and if it's uncovered
        for r in itertools.combinations(block, t - 1):
            r_tuple = tuple(sorted(r))
            if r_tuple in tminus1_to_tsubs:
                for x, ts in tminus1_to_tsubs[r_tuple].items():
                    if x not in block_set:
                        if ts in uncovered_set:
                            gains[x] = gains.get(x, 0) + 1
        return gains

    def _remove_redundant(blocks, deadline):
        if not blocks:
            return blocks

        # Map block index to the set of t-subset indices it covers
        block_tsub_indices = []
        cover_count = [0] * n_tsubs

        for block in blocks:
            tsubs = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    tsubs.append(idx)
                    cover_count[idx] += 1
            block_tsub_indices.append(tsubs)

        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                # Check if any t-subset in this block is covered ONLY by this block
                is_redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    # Remove block i
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsub_indices.pop(i)
                    changed = True
                    break

        return blocks

    def _local_search(blocks, deadline):
        if not blocks:
            return blocks

        # Precompute tsub indices for each block
        block_tsub_indices = []
        cover_count = [0] * n_tsubs
        for block in blocks:
            tsubs = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    tsubs.append(idx)
                    cover_count[idx] += 1
            block_tsub_indices.append(tsubs)

        improved = True
        max_iterations = 30
        iter_count = 0

        while improved and time.time() < deadline - 10 and iter_count < max_iterations:
            improved = False
            iter_count += 1

            # Try to remove a block
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break

                # Find critical t-subsets for block i
                criticals = [idx for idx in block_tsub_indices[i] if cover_count[idx] == 1]

                if not criticals:
                    # Block i is redundant, remove it
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsub_indices.pop(i)
                    improved = True
                    break

                # If not redundant, try to find a swap in another block that covers one critical
                crit_idx = criticals[0]
                crit_tsub = all_tsubs[crit_idx]

                found_swap = False
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if time.time() > deadline - 10:
                        break

                    block_j = blocks[j]
                    block_j_set = set(block_j)

                    common = block_j_set.intersection(crit_tsub)

                    if len(common) >= t - 1:
                        missing = [p for p in crit_tsub if p not in block_j_set]
                        if len(missing) == 1:
                            extra = [p for p in block_j if p not in common]
                            if extra:
                                for rem_p in extra:
                                    new_block_j = [p if p != rem_p else missing[0] for p in block_j]
                                    new_block_j_set = set(new_block_j)

                                    old_tsubs = set(block_tsub_indices[j])
                                    new_tsubs = set()
                                    for ts in itertools.combinations(new_block_j, t):
                                        idx = tsub_to_idx.get(ts)
                                        if idx is not None:
                                            new_tsubs.add(idx)

                                    can_swap = True
                                    for idx in old_tsubs:
                                        if idx not in new_tsubs and cover_count[idx] == 1:
                                            can_swap = False
                                            break

                                    if can_swap:
                                        for idx in old_tsubs:
                                            cover_count[idx] -= 1
                                        for idx in new_tsubs:
                                            cover_count[idx] += 1

                                        blocks[j] = new_block_j
                                        block_tsub_indices[j] = list(new_tsubs)

                                        improved = True
                                        found_swap = True
                                        break

                                    if found_swap:
                                        break
                    if found_swap:
                        break

                if found_swap:
                    break

        return blocks

    def _greedy_construct(uncovered, deadline, rng, seed_order=None):
        """Greedy construction with randomization"""
        blocks = []
        if seed_order is None:
            order = list(range(n_tsubs))
            rng.shuffle(order)
        else:
            order = seed_order

        while uncovered and time.time() < deadline - 15:
            # Pick the first uncovered t-subset in our order
            seed_idx = None
            for idx in order:
                if idx in uncovered:
                    seed_idx = idx
                    break
            if seed_idx is None:
                break

            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)

            while len(block) < k:
                if time.time() > deadline - 15:
                    break

                gains = _compute_gain(block, uncovered)

                if not gains:
                    # No gain, pick random remaining point
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(rng.choice(remaining))
                        block_set.add(block[-1])
                    else:
                        break
                    continue

                max_gain = max(gains.values())
                candidates = [x for x, g in gains.items() if g == max_gain]
                best_point = rng.choice(candidates)
                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)

        return blocks

    def _dual_guided_construct(deadline, rng):
        """
        Dual-guided construction: prioritize t-subsets that are "hard" to cover
        (i.e., those with fewer potential blocks). This gives a better initial
        covering for local search.
        """
        uncovered = set(range(n_tsubs))
        blocks = []

        # Compute a "price" for each t-subset based on how many blocks could cover it
        # A t-subset S can be covered by any k-block containing S.
        # Number of such blocks = C(v-t, k-t)
        # But we want to prioritize subsets that are harder to cover in practice.
        # Use a heuristic: subsets with points that appear in fewer "popular" positions.

        # Alternative: use the number of uncovered (t-1)-subsets that are subsets
        # of this t-subset as a priority indicator.
        # Actually, let's use a simpler approach: process in random order but
        # weight by how "central" the t-subset is.

        # Let's just do a greedy with a priority queue based on uncovered count
        # of the t-subset's (t-1)-subsets.

        # For simplicity, use random order but with a twist:
        # Start with a random t-subset, then greedily extend.
        # After each block, re-evaluate priorities.

        order = list(range(n_tsubs))
        rng.shuffle(order)

        while uncovered and time.time() < deadline - 15:
            seed_idx = None
            for idx in order:
                if idx in uncovered:
                    seed_idx = idx
                    break
            if seed_idx is None:
                break

            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)

            while len(block) < k:
                if time.time() > deadline - 15:
                    break

                gains = _compute_gain(block, uncovered)

                if not gains:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(rng.choice(remaining))
                        block_set.add(block[-1])
                    else:
                        break
                    continue

                max_gain = max(gains.values())
                candidates = [x for x, g in gains.items() if g == max_gain]
                best_point = rng.choice(candidates)
                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)

        return blocks

    best_blocks = None
    best_count = float('inf')

    # Multiple strategies
    strategies = [
        ("random_greedy", lambda rng: _greedy_construct(set(range(n_tsubs)), deadline, rng)),
        ("dual_guided", lambda rng: _dual_guided_construct(deadline, rng)),
    ]

    for strategy_name, strategy_fn in strategies:
        for restart in range(50):
            if time.time() > deadline - 20:
                break

            rng = random.Random(hash((strategy_name, restart)) % (2**32))
            blocks = strategy_fn(rng)

            if time.time() > deadline - 10 or not blocks:
                continue

            # Check if all covered
            all_covered = set()
            for block in blocks:
                for ts in itertools.combinations(block, t):
                    all_covered.add(ts)
            if len(all_covered) < n_tsubs:
                continue

            # Post-processing
            blocks = _remove_redundant(blocks, deadline)

            if time.time() < deadline - 15:
                blocks = _local_search(blocks, deadline)
                blocks = _remove_redundant(blocks, deadline)

            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
                if best_count <= 330:
                    break

        if time.time() > deadline - 20 or best_count <= 330:
            break

    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered and time.time() < deadline - 5:
            seed_idx = min(uncovered)
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                gains = _compute_gain(block, uncovered)
                if not gains:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(remaining[0])
                        block_set.add(block[-1])
                    else:
                        break
                    continue
                best_point = max(gains.keys(), key=lambda x: gains[x])
                block.append(best_point)
                block_set.add(best_point)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)
        best_blocks = blocks

    return best_blocks
