"""Covering design C(24,7,3): bitmask greedy with pair-guided construction and swap-based local search."""
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets (triples) as sorted tuples
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsub_set = set(all_tsubs)
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index for fast array lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets (7-blocks) as frozensets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, precompute the list of t-subset indices it covers
    # and a set of the t-subsets themselves
    ksub_tsubs = []  # list of list of t-subset indices
    ksub_tsub_sets = []  # list of set of t-subset tuples
    for ks in all_ksubs:
        tsubs = list(itertools.combinations(ks, t))
        idxs = [tsub_to_idx[ts] for ts in tsubs]
        ksub_tsubs.append(idxs)
        ksub_tsub_sets.append(set(tsubs))

    # Also store the k-subsets as tuples for quick reference
    ksub_tuples = [tuple(ks) for ks in all_ksubs]

    # Index k-subsets by their first element for faster search in local search
    # For local search, we need to find k-subsets containing a given t-subset
    # Build: for each t-subset, list of k-subset indices that contain it
    tsub_to_ksubs = [[] for _ in range(n_tsubs)]
    for ki, ks in enumerate(all_ksubs):
        for ts in itertools.combinations(ks, t):
            tsub_to_ksubs[tsub_to_idx[ts]].append(ki)

    best_blocks = None
    best_count = float('inf')

    rng = random.Random(12345)

    def _greedy_construct(pair_weights, rng_seed, deadline):
        """Greedy construction with pair-coverage weighting."""
        local_rng = random.Random(rng_seed)
        uncovered = set(all_tsub_set)
        blocks = []

        # Track pair coverage counts for weighting
        pair_count = {}  # frozenset of 2 points -> count

        while uncovered:
            if time.time() > deadline:
                break

            # Pick a random uncovered triple to start
            start_triple = local_rng.choice(list(uncovered))
            block = list(start_triple)
            block_set = set(block)

            # Update pair counts for the initial triple
            for a, b in itertools.combinations(block, 2):
                key = frozenset((a, b))
                pair_count[key] = pair_count.get(key, 0) + 1

            while len(block) < k:
                if time.time() > deadline:
                    break

                best_x = -1
                best_score = -1

                for x in range(v):
                    if x in block_set:
                        continue

                    # Count how many uncovered t-subsets this point would cover
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1

                    if gain == 0:
                        continue

                    # Compute pair-weighted score: prefer points that form under-covered pairs
                    pair_weight_sum = 0.0
                    for y in block:
                        key = frozenset((x, y))
                        pc = pair_count.get(key, 0)
                        # Lower pair count -> higher weight
                        pair_weight_sum += 1.0 / (1.0 + pc)

                    score = gain * 10.0 + pair_weight_sum

                    if score > best_score:
                        best_score = score
                        best_x = x

                if best_x == -1:
                    break

                block.append(best_x)
                block_set.add(best_x)

                # Update pair counts
                for y in block[:-1]:
                    key = frozenset((best_x, y))
                    pair_count[key] = pair_count.get(key, 0) + 1

            if len(block) < k:
                break

            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))

        if uncovered:
            return None

        return blocks

    def _remove_redundant(blocks, deadline):
        """Remove blocks whose t-subsets are all covered by other blocks."""
        if not blocks:
            return blocks

        # Build coverage map
        cov = {}
        for i, blk in enumerate(blocks):
            for tsub in itertools.combinations(blk, t):
                if tsub not in cov:
                    cov[tsub] = []
                cov[tsub].append(i)

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_blocks = []
            for i, blk in enumerate(blocks):
                redundant = True
                for tsub in itertools.combinations(blk, t):
                    covering = cov.get(tsub, [])
                    # Need at least one other block
                    if len(covering) <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    for tsub in itertools.combinations(blk, t):
                        cov[tsub].remove(i)
                    changed = True
                else:
                    new_blocks.append(blk)
            blocks = new_blocks
            # Re-index cov
            cov = {}
            for i, blk in enumerate(blocks):
                for tsub in itertools.combinations(blk, t):
                    if tsub not in cov:
                        cov[tsub] = []
                    cov[tsub].append(i)

        return blocks

    def _swap_local_search(blocks, deadline):
        """Try to replace each block with a better one that creates redundancy."""
        if not blocks:
            return blocks

        n = len(blocks)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            # Build coverage map
            cov = {}
            for i, blk in enumerate(blocks):
                for tsub in itertools.combinations(blk, t):
                    if tsub not in cov:
                        cov[tsub] = []
                    cov[tsub].append(i)

            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks

                old_block = blocks[i]
                old_tsubs = list(itertools.combinations(old_block, t))

                # Find t-subsets uniquely covered by this block
                unique_tsubs = []
                for tsub in old_tsubs:
                    covering = cov.get(tsub, [])
                    if len(covering) == 1:
                        unique_tsubs.append(tsub)

                if not unique_tsubs:
                    # Block is redundant, remove it
                    for tsub in old_tsubs:
                        cov[tsub].remove(i)
                    blocks.pop(i)
                    # Re-index
                    cov = {}
                    for j, blk in enumerate(blocks):
                        for tsub in itertools.combinations(blk, t):
                            if tsub not in cov:
                                cov[tsub] = []
                            cov[tsub].append(j)
                    improved = True
                    break

                # Find candidate k-subsets that cover all unique_tsubs
                # The new block must contain all points in each unique t-subset
                # Since t=3, each unique t-subset is 3 points; the new block must contain their union
                # But the union might be larger than k, so we need to check

                # Find the minimal set of points that must be in the new block
                required_points = set()
                for tsub in unique_tsubs:
                    required_points.update(tsub)

                if len(required_points) > k:
                    continue

                # Find k-subsets that contain all required_points
                candidates = []
                for ki, ks in enumerate(ksub_tuples):
                    if all(p in ks for p in required_points):
                        candidates.append(ki)

                if not candidates:
                    continue

                # For each candidate, check if it covers all unique_tsubs
                # (it should, since it contains required_points which is the union of unique_tsubs)
                # But we also need to check it doesn't create new uncovered t-subsets
                # Actually, we need to ensure that after removing old_block and adding new_block,
                # all t-subsets are still covered.

                best_candidate = None
                best_redundancy_gain = -1

                for ki in candidates:
                    new_block = ksub_tuples[ki]
                    new_tsubs = ksub_tsub_sets[ki]

                    # Check: does new_block cover all unique_tsubs?
                    # Since required_points is the union of unique_tsubs, and new_block contains
                    # required_points, it should cover each unique_tsub. But let's verify.
                    covers_all = True
                    for tsub in unique_tsubs:
                        if tsub not in new_tsubs:
                            covers_all = False
                            break
                    if not covers_all:
                        continue

                    # Count how many blocks become redundant after this swap
                    # A block j (j != i) becomes redundant if all its t-subsets are covered
                    # by blocks other than j AND (new_block or blocks other than j and i)
                    # Simplified: for each block j != i, check if all its t-subsets are covered
                    # by (blocks \ {j, i}) ∪ {new_block}

                    redundancy_gain = 0
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        blk_j = blocks[j]
                        redundant = True
                        for tsub in itertools.combinations(blk_j, t):
                            # Is tsub covered by some block other than j?
                            # Check blocks \ {j, i} and new_block
                            covered = False
                            # Check new_block
                            if tsub in new_tsubs:
                                covered = True
                            else:
                                # Check other blocks
                                for m in range(len(blocks)):
                                    if m == j or m == i:
                                        continue
                                    if tsub in itertools.combinations(blocks[m], t):
                                        covered = True
                                        break
                            if not covered:
                                redundant = False
                                break
                        if redundant:
                            redundancy_gain += 1

                    if redundancy_gain > best_redundancy_gain:
                        best_redundancy_gain = redundancy_gain
                        best_candidate = ki

                if best_candidate is not None and best_redundancy_gain > 0:
                    new_block = list(ksub_tuples[best_candidate])
                    new_block.sort()
                    blocks[i] = new_block
                    improved = True
                    break

        return blocks

    # Main search loop
    for restart in range(500):
        if time.time() > deadline:
            break

        # Vary pair weights slightly for diversity
        pair_weight_scale = 1.0 + 0.1 * (restart % 5)

        blocks = _greedy_construct(pair_weight_scale, 42 + restart, deadline)

        if blocks is not None:
            # Remove redundant blocks
            blocks = _remove_redundant(blocks, deadline)

            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)

            # Try swap-based local search
            if best_blocks is not None and len(best_blocks) <= 85:
                improved = _swap_local_search(list(best_blocks), deadline)
                improved = _remove_redundant(improved, deadline)
                if len(improved) < best_count:
                    best_blocks = improved
                    best_count = len(improved)

    # Fallback: simple greedy
    if best_blocks is None:
        uncovered = set(all_tsub_set)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks

    return best_blocks
