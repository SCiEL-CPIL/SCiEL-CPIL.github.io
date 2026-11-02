"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)

    # Map each t-subset to an index
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}

    # Coverage count for each t-subset
    coverage = np.zeros(num_t_subs, dtype=np.int32)

    # List of blocks (each block is a sorted tuple)
    blocks = []

    # For each block, store the set of t-subset indices it covers
    block_triples = []  # list of sets of t-subset indices

    def add_block(block):
        """Add a block and update coverage."""
        blocks.append(tuple(sorted(block)))
        triple_set = set()
        for tri in itertools.combinations(block, t):
            idx = t_sub_index[tri]
            triple_set.add(idx)
            coverage[idx] += 1
        block_triples.append(triple_set)
        return len(blocks) - 1

    def remove_block(block_idx):
        """Remove a block and update coverage."""
        triple_set = block_triples[block_idx]
        for idx in triple_set:
            coverage[idx] -= 1
        del blocks[block_idx]
        del block_triples[block_idx]

    def is_redundant(block_idx):
        """Check if removing block_idx still leaves all t-subsets covered."""
        triple_set = block_triples[block_idx]
        for idx in triple_set:
            if coverage[idx] == 1:
                return False
        return True

    # Phase 1: Greedy construction using potential-based scoring
    # Use sum of 1/coverage^2 for uncovered triples, plus bonus for covering uncovered triples
    def greedy_build():
        """Build initial covering using greedy with potential scoring."""
        nonlocal blocks, block_triples, coverage

        uncovered_set = set(range(num_t_subs))

        while uncovered_set:
            if time.time() > deadline - 5.0:
                break

            # Pick the t-subset with lowest coverage (should be 0 for uncovered)
            # For speed, just pick the first uncovered one
            target = min(uncovered_set)
            target_sub = all_t_subs[target]

            # Build a block containing target_sub, choosing remaining points
            # to maximize potential gain
            block = list(target_sub)
            remaining_points = [p for p in range(v) if p not in block]

            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break

                best_gain = -1
                best_point = None
                for x in remaining_points:
                    # Compute gain: sum of 1/max(coverage[idx],1)^2 for triples
                    # that would be newly covered or have low coverage
                    gain = 0.0
                    for rest in itertools.combinations(block, t - 1):
                        tri = tuple(sorted(rest + (x,)))
                        idx = t_sub_index[tri]
                        if idx in uncovered_set:
                            gain += 1.0
                        else:
                            gain += 1.0 / (coverage[idx] + 1) ** 2
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point is None:
                    best_point = remaining_points[0]

                block.append(best_point)
                remaining_points.remove(best_point)

            # Add the block
            add_block(block)

            # Update uncovered set
            for idx in block_triples[-1]:
                if coverage[idx] == 1:
                    uncovered_set.discard(idx)

        return len(blocks)

    greedy_build()

    # Phase 2: Local search - try to remove redundant blocks
    if time.time() < deadline - 10.0:
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                if is_redundant(i):
                    remove_block(i)
                    changed = True

    # Phase 3: Local search - try to replace blocks with smaller/better ones
    if time.time() < deadline - 10.0:
        # Try swapping: for each block, see if we can find a different block
        # that covers all its uniquely-covered triples
        for i in range(len(blocks)):
            if time.time() > deadline - 5.0:
                break
            if not is_redundant(i):
                continue
            # This block is redundant, remove it
            remove_block(i)
            # We don't re-add; just let the coverage stand
            # But we need to make sure all are still covered
            # Actually, is_redundant already checks this

    # Phase 4: Try to merge or find better blocks
    # For each pair of blocks, check if their union of triples can be covered by fewer blocks
    if time.time() < deadline - 15.0:
        # Simple: try removing each block and see if we can re-cover with a new single block
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            if is_redundant(i):
                remove_block(i)

    # Final check: ensure all t-subsets are covered
    # If not, add blocks to cover remaining uncovered
    for idx in range(num_t_subs):
        if time.time() > deadline - 3.0:
            break
        if coverage[idx] == 0:
            sub = all_t_subs[idx]
            block = list(sub)
            remaining = [p for p in range(v) if p not in block]
            random.shuffle(remaining)
            while len(block) < k and remaining:
                block.append(remaining.pop())
            add_block(block)

    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
