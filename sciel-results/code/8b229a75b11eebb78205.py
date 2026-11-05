import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit

    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)

    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Map block index to list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[triple] for triple in itertools.combinations(b, t)]
        block_triples.append(ts)

    # Map triple index to list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)

    def greedy_randomized(seed):
        random.seed(seed)
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)

        selected_blocks = []

        while is_uncovered.any():
            max_gain = np.max(gain)
            if max_gain == 0:
                break

            candidates = np.where(gain == max_gain)[0]
            b_idx = random.choice(candidates.tolist())

            selected_blocks.append(b_idx)

            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        if not selected_blocks:
            return []
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        active = [True] * len(selected_blocks)

        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True

        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

    def local_search(cover, seed, time_budget=10.0):
        random.seed(seed)
        if not cover:
            return cover

        cover_set = set(cover)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        # First, remove redundant blocks
        cover = reduce_cover(cover)
        cover_set = set(cover)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        best_cover = list(cover)
        best_len = len(best_cover)

        # Try swaps: remove one block, add another
        max_iterations = 2000
        for _ in range(max_iterations):
            if time.time() - start_time > timeout - time_budget:
                break

            if not cover:
                break

            # Pick a random block to try removing
            idx = random.randint(0, len(cover) - 1)
            b_remove = cover[idx]

            # Remove it
            cover.remove(b_remove)

            # Check coverage
            cover_count = np.zeros(num_triples, dtype=np.int32)
            for b_idx in cover:
                for t_idx in block_triples[b_idx]:
                    cover_count[t_idx] += 1

            uncovered = np.where(cover_count == 0)[0]

            if len(uncovered) == 0:
                # Coverage is still complete, keep the removal
                best_cover = list(cover)
                best_len = len(cover)
                continue

            # Coverage is broken. We need to find a block that covers ALL uncovered_triples.
            # Use intersection of triple_blocks for efficiency.
            # Start with the blocks covering the first uncovered triple
            candidate_set = set(triple_blocks[uncovered[0]])

            # Intersect with blocks covering other uncovered triples
            for t_idx in uncovered[1:]:
                candidate_set &= set(triple_blocks[t_idx])
                if not candidate_set:
                    break

            found_swap = False
            if candidate_set:
                # Try to find a candidate not in current cover
                # Randomize order to avoid bias
                candidates = list(candidate_set)
                random.shuffle(candidates)

                for b_add in candidates:
                    if b_add in cover:
                        continue

                    # Valid swap found
                    # Update cover_count
                    for t_idx in block_triples[b_add]:
                        cover_count[t_idx] += 1

                    cover.append(b_add)
                    found_swap = True
                    break

            if not found_swap:
                # Restore b_remove
                for t_idx in block_triples[b_remove]:
                    cover_count[t_idx] += 1
                cover.append(b_remove)

        return best_cover

    best_cover = None
    seed = 0
    num_restarts = 0

    while True:
        if time.time() - start_time > timeout:
            break

        # Run greedy
        cover = greedy_randomized(seed)
        # Reduce
        cover = reduce_cover(cover)

        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover

        # Occasionally do local search on the best cover
        if num_restarts > 0 and num_restarts % 3 == 0 and best_cover is not None:
            ls_cover = local_search(best_cover, seed + 1000)
            if len(ls_cover) < len(best_cover):
                best_cover = ls_cover

        seed += 1
        num_restarts += 1

        # If we've been running for a while and haven't improved, increase local search effort
        if num_restarts > 50:
            break

    # Final local search on the best cover
    if best_cover is not None:
        ls_cover = local_search(best_cover, seed + 2000, time_budget=5.0)
        if len(ls_cover) < len(best_cover):
            best_cover = ls_cover

    # Convert indices back to blocks
    if best_cover is None:
        cover = greedy_randomized(0)
        return [list(all_blocks[b_idx]) for b_idx in cover]

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
