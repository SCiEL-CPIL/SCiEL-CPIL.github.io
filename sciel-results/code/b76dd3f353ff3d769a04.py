import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute: for each pair (a,b), list of triples containing both
    pair_to_triples = {}
    for i, tr in enumerate(triples):
        for j in range(t):
            for m in range(j+1, t):
                pair = (tr[j], tr[m])
                if pair not in pair_to_triples:
                    pair_to_triples[pair] = []
                pair_to_triples[pair].append(i)

    def greedy_cover(uncovered_mask=None, random_state=0):
        rng = random.Random(random_state)
        if uncovered_mask is None:
            uncovered = np.ones(num_triples, dtype=np.uint8)
            num_uncovered = num_triples
        else:
            uncovered = uncovered_mask.copy()
            num_uncovered = int(np.sum(uncovered))

        blocks = []
        points = list(range(v))

        while num_uncovered > 0:
            # Pick the uncovered triple with the most "pressure" (arbitrary: first uncovered)
            start_idx = int(np.argmax(uncovered))
            if uncovered[start_idx] == 0:
                break

            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)

            # Mark target triple as covered
            uncovered[start_idx] = 0
            num_uncovered -= 1

            while len(block) < k and num_uncovered > 0:
                candidates = [x for x in points if x not in block_set]
                if not candidates:
                    break

                # Compute gain for each candidate: number of currently-uncovered triples
                # that would be covered by adding x to the block
                best_gain = -1
                best_point = -1

                # For each candidate x, gain = count of uncovered triples containing x and 2 points from block
                for x in candidates:
                    gain = 0
                    for p1 in block:
                        for p2 in block:
                            if p1 == p2:
                                continue
                            pair = (min(p1,p2), max(p1,p2))
                            if pair in pair_to_triples:
                                for idx in pair_to_triples[pair]:
                                    if uncovered[idx] and triples[idx] != (min(x,p1,p2),) and x in triples[idx]:
                                        gain += 1
                                        break
                    # Simpler: check all triples containing x and 2 block points
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point == -1 or best_gain <= 0:
                    # No point adds coverage; add random point to fill
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

                # Update uncovered: mark all triples containing best_point and 2 block points
                for p1 in block:
                    for p2 in block:
                        if p1 == p2 or p1 == best_point or p2 == best_point:
                            continue
                        pair = (min(p1,p2), max(p1,p2))
                        if pair in pair_to_triples:
                            for idx in pair_to_triples[pair]:
                                if best_point in triples[idx] and uncovered[idx]:
                                    uncovered[idx] = 0
                                    num_uncovered -= 1

            blocks.append(tuple(sorted(block)))

        return blocks, uncovered, num_uncovered

    def trim_blocks(blocks):
        if not blocks:
            return blocks

        counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []

        for block in blocks:
            b_triples = []
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                b_triples.append(idx)
            block_triples.append(b_triples)

        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                essential = False
                for idx in block_triples[i]:
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for idx in block_triples[i]:
                        counts[idx] -= 1

        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def local_search_trim(blocks, max_iters=30):
        """Try removing blocks; if coverage breaks, try swapping with another block."""
        if not blocks:
            return blocks

        counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []
        for block in blocks:
            b_triples = []
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                b_triples.append(idx)
            block_triples.append(b_triples)

        active = [True] * len(blocks)
        best_len = len(blocks)

        for _ in range(max_iters):
            improved = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                # Try removing block i
                temp_counts = counts.copy()
                for idx in block_triples[i]:
                    temp_counts[idx] -= 1
                if np.any(temp_counts < 0):
                    continue
                if np.all(temp_counts >= 0):
                    uncovered_after = np.where(temp_counts == 0)[0]
                    if len(uncovered_after) == 0:
                        active[i] = False
                        counts = temp_counts
                        best_len -= 1
                        improved = True
                        break
                    else:
                        # Look for a block that covers all uncovered_after
                        for j in range(len(blocks)):
                            if not active[j] or j == i:
                                continue
                            covers_all = True
                            for uidx in uncovered_after:
                                if uidx not in block_triples[j]:
                                    covers_all = False
                                    break
                            if covers_all:
                                active[i] = False
                                counts = temp_counts
                                best_len -= 1
                                improved = True
                                break
                        if improved:
                            break
            if not improved:
                break

        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def is_valid(blocks):
        covered = np.zeros(num_triples, dtype=np.uint8)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                covered[idx] = 1
        return np.all(covered == 1)

    best_blocks = None
    best_len = float('inf')

    seed_counter = 0

    while time.time() < deadline - 10:
        # Phase 1: Fresh greedy
        blocks, uncovered, num_uncovered = greedy_cover(random_state=seed_counter)
        trimmed = trim_blocks(blocks)

        if len(trimmed) < best_len:
            best_len = len(trimmed)
            best_blocks = list(trimmed)

        if best_len <= 69:
            break

        # Phase 2: Perturbation and repair
        if time.time() < deadline - 5:
            rng_pert = random.Random(seed_counter + 1000)

            strategies = []

            # Strategy 1: Remove 1-3 random blocks
            if len(trimmed) > 5:
                num_kicks = rng_pert.randint(1, min(3, len(trimmed) - 1))
                kick_indices = rng_pert.sample(range(len(trimmed)), num_kicks)
                remaining = [b for i, b in enumerate(trimmed) if i not in kick_indices]
                strategies.append(remaining)

            # Strategy 2: Remove most redundant blocks
            if len(trimmed) > 5:
                counts = np.zeros(num_triples, dtype=np.int32)
                for block in trimmed:
                    for triple in itertools.combinations(block, t):
                        counts[triple_to_idx[triple]] += 1
                block_scores = []
                for i, block in enumerate(trimmed):
                    score = 0
                    for triple in itertools.combinations(block, t):
                        if counts[triple_to_idx[triple]] == 1:
                            score += 1
                    block_scores.append(score)
                sorted_idx = sorted(range(len(trimmed)), key=lambda x: block_scores[x])
                num_remove = min(2, len(trimmed) - 1)
                remove_idx = set(sorted_idx[:num_remove])
                remaining = [b for i, b in enumerate(trimmed) if i not in remove_idx]
                strategies.append(remaining)

            for remaining_blocks in strategies:
                if time.time() >= deadline - 5:
                    break

                covered_mask = np.zeros(num_triples, dtype=np.uint8)
                for block in remaining_blocks:
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        covered_mask[idx] = 1

                uncovered_mask = 1 - covered_mask
                num_uncovered = int(np.sum(uncovered_mask))

                if num_uncovered > 0:
                    repaired_blocks, _, _ = greedy_cover(uncovered_mask=uncovered_mask, random_state=seed_counter + 200)
                    combined = remaining_blocks + repaired_blocks
                else:
                    combined = remaining_blocks

                trimmed_combined = trim_blocks(combined)
                trimmed_combined = local_search_trim(trimmed_combined, max_iters=15)

                if len(trimmed_combined) < best_len:
                    best_len = len(trimmed_combined)
                    best_blocks = list(trimmed_combined)

        seed_counter += 1

    # Final polish
    if best_blocks and time.time() < deadline - 2:
        best_blocks = local_search_trim(best_blocks, max_iters=50)

    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
