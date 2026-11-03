# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using randomized greedy with
    redundancy removal and local search polish.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 120s limit

    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)

    # Map each triple to its index for fast lookup
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}

    def greedy_cover(seed_triple_idx, rng, max_blocks=None):
        """
        Greedy covering starting from a seed triple.
        Returns list of blocks (each a sorted tuple of points).
        """
        uncovered = bytearray([1] * n_triples)
        uncovered_count = n_triples
        blocks = []

        if max_blocks is None:
            max_blocks = n_triples // (t * (t - 1) // 2) + 100

        uncovered[seed_triple_idx] = 0
        uncovered_count -= 1

        while uncovered_count > 0 and len(blocks) < max_blocks:
            if time.time() > deadline - 5:
                break

            start_idx = -1
            for i in range(n_triples):
                if uncovered[i]:
                    start_idx = i
                    break
            if start_idx == -1:
                break

            seed_triple = all_triples[start_idx]
            block = list(seed_triple)
            uncovered[start_idx] = 0
            uncovered_count -= 1

            while len(block) < k:
                if time.time() > deadline - 5:
                    break

                best_point = -1
                best_gain = 0

                for p in range(v):
                    if p in block:
                        continue
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(combo + (p,)))
                        idx = triple_to_idx.get(triple)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p

                if best_point == -1 or best_gain == 0:
                    candidates = [p for p in range(v) if p not in block]
                    if not candidates:
                        break
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block.sort()

                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(triple)
                    if idx is not None and uncovered[idx]:
                        uncovered[idx] = 0
                        uncovered_count -= 1

            if len(block) == k:
                blocks.append(tuple(block))

        return blocks

    def remove_redundant_blocks(blocks):
        """Remove blocks that don't cover any unique triple."""
        if not blocks:
            return []

        n = len(blocks)
        triple_blocks = defaultdict(list)

        for bi, block in enumerate(blocks):
            for triple in itertools.combinations(block, t):
                triple_blocks[triple].append(bi)

        active = [True] * n
        unique_triple_count = [0] * n
        for triple, blocks_list in triple_blocks.items():
            if len(blocks_list) == 1:
                unique_triple_count[blocks_list[0]] += 1

        changed = True
        while changed:
            changed = False
            for bi in range(n):
                if not active[bi]:
                    continue
                if unique_triple_count[bi] == 0:
                    active[bi] = False
                    changed = True
                    for triple in itertools.combinations(blocks[bi], t):
                        bl = triple_blocks[triple]
                        if bi in bl:
                            bl.remove(bi)
                        if len(bl) == 1:
                            unique_triple_count[bl[0]] += 1

        result = [blocks[bi] for bi in range(n) if active[bi]]
        return result

    def verify_covering(blocks):
        """Verify that every t-subset is covered by at least one block."""
        covered = set()
        for block in blocks:
            for triple in itertools.combinations(block, t):
                covered.add(triple)
        return len(covered) == n_triples

    def local_search_polish(blocks, rng, max_time=60):
        """
        Local search to reduce block count:
        1. Try removing blocks (if still covered)
        2. Try swapping one point in a block for another
        3. Try replacing a block with a different block that covers its unique triples
        """
        if not blocks:
            return blocks

        blocks = list(blocks)
        n_blocks = len(blocks)
        polish_deadline = time.time() + max_time

        # Build coverage count for each triple
        triple_coverage = defaultdict(int)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                triple_coverage[triple] += 1

        # Step 1: Remove redundant blocks iteratively
        changed = True
        while changed and time.time() < polish_deadline - 5:
            changed = False
            for bi in range(len(blocks)):
                # Check if removing block bi keeps all triples covered
                can_remove = True
                for triple in itertools.combinations(blocks[bi], t):
                    if triple_coverage[triple] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for triple in itertools.combinations(blocks[bi], t):
                        triple_coverage[triple] -= 1
                    blocks.pop(bi)
                    changed = True
                    break  # Restart scan after removal

        if not blocks:
            return blocks

        # Step 2: Try point swaps within blocks
        improved = True
        while improved and time.time() < polish_deadline - 10:
            improved = False
            n_blocks = len(blocks)
            for bi in range(n_blocks):
                if time.time() > polish_deadline - 10:
                    break
                block = list(blocks[bi])
                block_set = set(block)

                for pi in range(k):
                    if time.time() > polish_deadline - 10:
                        break
                    old_point = block[pi]
                    for new_point in range(v):
                        if new_point in block_set:
                            continue

                        # Simulate swap: remove old_point, add new_point
                        new_block = block[:]
                        new_block[pi] = new_point
                        new_block.sort()
                        new_block_set = set(new_block)

                        # Check if swap is beneficial: does it cover any triple
                        # that was only covered by the old block?
                        # And does it not uncover any triple?
                        # Compute triples lost and gained
                        old_triples = set(itertools.combinations(block, t))
                        new_triples = set(itertools.combinations(new_block, t))

                        lost = old_triples - new_triples
                        gained = new_triples - old_triples

                        # Swap is valid if no lost triple has coverage <= 1
                        swap_valid = True
                        for triple in lost:
                            if triple_coverage[triple] <= 1:
                                swap_valid = False
                                break

                        if not swap_valid:
                            continue

                        # Check if swap reduces total "excess" coverage
                        # or at least doesn't increase block count
                        # We prefer swaps that cover new triples or reduce redundancy
                        gain = len(gained)
                        loss = len(lost)

                        # Accept if it covers new triples or is neutral
                        if gain > 0 or (gain == 0 and loss == 0):
                            # Accept the swap
                            for triple in lost:
                                triple_coverage[triple] -= 1
                            for triple in gained:
                                triple_coverage[triple] += 1
                            blocks[bi] = tuple(new_block)
                            improved = True
                            break

                    if improved:
                        break

        # Step 3: Try replacing blocks with better ones
        # For each block, try to find an alternative block that covers all its
        # uniquely-covered triples
        for bi in range(len(blocks)):
            if time.time() > polish_deadline - 5:
                break

            block = list(blocks[bi])
            block_set = set(block)

            # Find triples that are only covered by this block
            unique_triples = []
            for triple in itertools.combinations(block, t):
                if triple_coverage[triple] == 1:
                    unique_triples.append(triple)

            if not unique_triples:
                continue

            # Try to find a replacement block that covers all unique triples
            # and has more total coverage
            best_replacement = None
            best_score = 0

            # Generate candidate blocks that contain all points of some unique triple
            # and try to extend them
            for ut in unique_triples[:3]:  # Limit to first 3 unique triples for speed
                # Try blocks that contain all points of ut
                ut_set = set(ut)
                remaining_points = [p for p in range(v) if p not in ut_set]

                # Try a few random extensions
                for _ in range(5):
                    if time.time() > polish_deadline - 5:
                        break
                    candidate = list(ut)
                    remaining = [p for p in remaining_points]
                    rng.shuffle(remaining)
                    candidate.extend(remaining[:k - t])
                    candidate.sort()
                    candidate_set = set(candidate)

                    # Check if candidate covers all unique triples
                    covers_all = True
                    for ut2 in unique_triples:
                        if not set(ut2).issubset(candidate_set):
                            covers_all = False
                            break

                    if not covers_all:
                        continue

                    # Score: number of new triples covered
                    cand_triples = set(itertools.combinations(candidate, t))
                    score = len(cand_triples)
                    if score > best_score:
                        best_score = score
                        best_replacement = tuple(candidate)

            if best_replacement is not None:
                old_block = blocks[bi]
                old_triples = set(itertools.combinations(old_block, t))
                new_triples = set(itertools.combinations(best_replacement, t))

                # Verify swap is valid
                lost = old_triples - new_triples
                for triple in lost:
                    if triple_coverage[triple] <= 1:
                        best_replacement = None
                        break

                if best_replacement is not None:
                    for triple in lost:
                        triple_coverage[triple] -= 1
                    gained = new_triples - old_triples
                    for triple in gained:
                        triple_coverage[triple] += 1
                    blocks[bi] = best_replacement

        return blocks

    # Multi-start randomized greedy
    best_blocks = None
    best_count = float('inf')

    rng = random.Random(42)

    # First, try a deterministic greedy
    blocks = greedy_cover(0, rng)
    if blocks:
        cleaned = remove_redundant_blocks(blocks)
        if verify_covering(cleaned):
            best_blocks = cleaned
            best_count = len(cleaned)
        elif verify_covering(blocks):
            best_blocks = blocks
            best_count = len(blocks)

    # Now try randomized starts
    num_attempts = 15
    for attempt in range(num_attempts):
        if time.time() > deadline - 30:
            break

        seed_idx = rng.randint(0, n_triples - 1)
        blocks = greedy_cover(seed_idx, rng)
        if not blocks:
            continue

        cleaned = remove_redundant_blocks(blocks)
        if verify_covering(cleaned):
            if len(cleaned) < best_count:
                best_blocks = cleaned
                best_count = len(cleaned)
        elif verify_covering(blocks):
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)

    # If we didn't find a valid covering, fall back
    if best_blocks is None or not verify_covering(best_blocks):
        uncovered = bytearray([1] * n_triples)
        uncovered_count = n_triples
        blocks = []

        while uncovered_count > 0:
            if time.time() > deadline - 2:
                break
            start_idx = -1
            for i in range(n_triples):
                if uncovered[i]:
                    start_idx = i
                    break
            if start_idx == -1:
                break

            seed_triple = all_triples[start_idx]
            block = list(seed_triple)
            uncovered[start_idx] = 0
            uncovered_count -= 1

            while len(block) < k:
                if time.time() > deadline - 2:
                    break
                best_point = -1
                best_gain = 0
                for p in range(v):
                    if p in block:
                        continue
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(combo + (p,)))
                        idx = triple_to_idx.get(triple)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                if best_point == -1:
                    candidates = [p for p in range(v) if p not in block]
                    best_point = rng.choice(candidates) if candidates else 0
                block.append(best_point)
                block.sort()
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(triple)
                    if idx is not None and uncovered[idx]:
                        uncovered[idx] = 0
                        uncovered_count -= 1

            if len(block) == k:
                blocks.append(tuple(block))

        if verify_covering(blocks):
            best_blocks = blocks
            best_count = len(blocks)

    # Apply local search polish to the best solution
    if best_blocks and time.time() < deadline - 15:
        polish_time = min(45, deadline - time.time() - 10)
        if polish_time > 5:
            polished = local_search_polish(best_blocks, rng, max_time=polish_time)
            if verify_covering(polished) and len(polished) < best_count:
                best_blocks = polished
                best_count = len(polished)

    # Final cleanup
    if best_blocks:
        cleaned = remove_redundant_blocks(best_blocks)
        if verify_covering(cleaned) and len(cleaned) < best_count:
            best_blocks = cleaned
            best_count = len(cleaned)

    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
# EVOLVE-BLOCK-END
