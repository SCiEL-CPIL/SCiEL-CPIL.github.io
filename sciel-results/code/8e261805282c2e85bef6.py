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
        """
        Local search: try to swap one block for another to reduce size or improve coverage.
        Uses greedy selection of best replacement block when coverage is broken.
        """
        random.seed(seed)
        if not cover:
            return cover

        # First, remove redundant blocks
        cover = reduce_cover(cover)
        
        if not cover:
            return cover

        # Precompute: for each triple, list of blocks that cover it (already have triple_blocks)
        # Also precompute: for each block, the set of triples it covers as a set for fast intersection
        block_triple_sets = [set(ts) for ts in block_triples]

        best_cover = list(cover)
        best_len = len(best_cover)

        # Main local search loop
        max_iterations = 300
        for it in range(max_iterations):
            if time.time() - start_time > timeout - time_budget:
                break

            if not cover:
                break

            # Pick a random block to try removing
            idx = random.randint(0, len(cover) - 1)
            b_remove = cover[idx]
            
            # Remove it
            cover.pop(idx)

            # Compute uncovered triples
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

            # Coverage is broken. Find the best replacement block.
            # We want a block that covers the maximum number of uncovered triples.
            # For efficiency, we only consider blocks that are not already in cover.
            
            uncovered_set = set(uncovered.tolist())
            best_block = -1
            best_overlap = -1
            
            # Try all blocks not in cover
            cover_set = set(cover)
            for b_candidate in range(num_blocks):
                if b_candidate in cover_set:
                    continue
                
                # Count how many uncovered triples this block covers
                overlap = len(block_triple_sets[b_candidate] & uncovered_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_block = b_candidate
                    if best_overlap == len(uncovered_set):
                        break  # Can't do better than covering all uncovered

            if best_block != -1 and best_overlap > 0:
                cover.append(best_block)
                # Update best if we didn't increase size (shouldn't happen since we removed one and added one)
                if len(cover) < best_len:
                    best_cover = list(cover)
                    best_len = len(cover)
            else:
                # No good replacement, restore the removed block
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
