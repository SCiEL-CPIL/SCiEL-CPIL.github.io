"""Covering design C(19,7,4) via simulated annealing over block populations."""
import time
import random
import itertools
import numpy as np

_DEADLINE_MARGIN = 8.0  # seconds to leave before the 150s kill


def _encode(v, k, t):
    """Return a function to encode a t-subset (as a sorted tuple of ints in 0..v-1)
    into a unique integer index."""
    # Use combinatorial number system: index of C(v, t)-th combination
    # Precompute binomial coefficients for fast lookup
    # For v=19, t=4: C(19,4) = 3876, manageable
    # We'll map each t-subset to an index via a precomputed dict for speed
    pass


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN

    # Total number of t-subsets
    from math import comb
    total_t = comb(v, t)

    # Generate all t-subsets and assign each an index
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map from t-subset tuple to index
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # State: coverage count for each t-subset (how many blocks cover it)
    # covered[i] = number of blocks containing t-subset i
    covered = np.zeros(total_t, dtype=np.int32)

    # A block covers a set of t-subsets: all t-subsets of its k elements
    # Precompute for a given block (tuple of k sorted ints) the list of t-subset indices it covers
    def block_cover_indices(block):
        """block: sorted tuple of k ints; return list of indices of t-subsets covered"""
        return [t_to_idx[ts] for ts in itertools.combinations(block, t)]

    # We will maintain a list of blocks, each as a sorted tuple of k ints
    # And for each block, the list of t-subset indices it covers (for fast add/remove)

    # ---- Initial solution: greedy set cover (fast, gives a good starting point) ----
    def greedy_initial():
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            # Pick the block that covers the most uncovered t-subsets
            # For efficiency, start from the first uncovered t-subset and extend greedily
            seed = min(uncovered)
            block = list(seed)
            while len(block) < k:
                # Find point x not in block that maximizes newly covered t-subsets
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets are in block ∪ {x}
                    # A t-subset is newly covered if it contains x and (t-1)-subset of block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            block_t = tuple(block)
            blocks.append(block_t)
            # Remove all t-subsets of this block from uncovered
            for ts in itertools.combinations(block_t, t):
                uncovered.discard(ts)
        return blocks

    blocks = greedy_initial()
    # Build coverage array and per-block cover lists
    block_covers = []
    for b in blocks:
        ci = block_cover_indices(b)
        block_covers.append(ci)
        for idx in ci:
            covered[idx] += 1

    n_uncov = int(np.sum(covered == 0))
    best_n_blocks = len(blocks)
    best_blocks = list(blocks)
    best_block_covers = list(block_covers)
    best_covered = covered.copy()

    # ---- Simulated Annealing ----
    # Moves:
    #   1. Swap a point in a block with another point
    #   2. Remove a block (if it doesn't cause any t-subset to become uncovered)
    #   3. Add a new block (greedy seed from an uncovered t-subset)
    #   4. Replace a block with a better one (local search on a single block)

    def evaluate_delta_cover(covered):
        """Return number of uncovered t-subsets."""
        return int(np.sum(covered == 0))

    rng = random.Random(42)

    # Temperature schedule
    T_start = 2.0
    T_end = 0.01
    # We'll run until deadline

    while time.time() < deadline:
        # Pick a move type
        r = rng.random()
        if r < 0.6:
            # Swap a point in a random block
            if not blocks:
                continue
            bi = rng.randrange(len(blocks))
            old_block = blocks[bi]
            old_covers = block_covers[bi]
            pos = rng.randrange(k)
            old_point = old_block[pos]
            # Choose a new point not in the block
            candidates = [x for x in range(v) if x != old_point and x not in old_block]
            if not candidates:
                continue
            new_point = rng.choice(candidates)
            new_block_list = list(old_block)
            new_block_list[pos] = new_point
            new_block_list.sort()
            new_block = tuple(new_block_list)
            if new_block == old_block:
                continue
            new_covers = block_cover_indices(new_block)

            # Compute delta in uncovered count
            # Remove old block's coverage, add new block's coverage
            old_set = set(old_covers)
            new_set = set(new_covers)
            # t-subsets only in old: may become uncovered
            only_old = old_set - new_set
            # t-subsets only in new: may become covered
            only_new = new_set - old_set

            delta = 0
            for idx in only_old:
                if covered[idx] == 1:
                    delta += 1  # becomes uncovered
            for idx in only_new:
                if covered[idx] == 0:
                    delta -= 1  # becomes covered

            # Accept?
            if delta <= 0 or rng.random() < np.exp(-delta / max(T_start, 1e-9)):
                # Apply move
                for idx in old_covers:
                    covered[idx] -= 1
                for idx in new_covers:
                    covered[idx] += 1
                blocks[bi] = new_block
                block_covers[bi] = new_covers
        elif r < 0.8:
            # Try to remove a block that is redundant
            if len(blocks) <= 1:
                continue
            bi = rng.randrange(len(blocks))
            old_block = blocks[bi]
            old_covers = block_covers[bi]
            # Check if any t-subset has coverage == 1 and is in old_covers
            redundant = True
            for idx in old_covers:
                if covered[idx] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block
                for idx in old_covers:
                    covered[idx] -= 1
                blocks.pop(bi)
                block_covers.pop(bi)
                n_uncov = evaluate_delta_cover(covered)
                if n_uncov == 0 and len(blocks) < best_n_blocks:
                    best_n_blocks = len(blocks)
                    best_blocks = list(blocks)
                    best_block_covers = list(block_covers)
                    best_covered = covered.copy()
        elif r < 0.9:
            # Add a new block seeded from an uncovered t-subset
            # Find an uncovered t-subset
            uncov_indices = np.where(covered == 0)[0]
            if len(uncov_indices) == 0:
                continue
            seed_idx = uncov_indices[rng.randrange(len(uncov_indices))]
            seed = all_t_subsets[seed_idx]
            block = list(seed)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_to_idx[ts]
                        if covered[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            new_block = tuple(block)
            new_covers = block_cover_indices(new_block)
            blocks.append(new_block)
            block_covers.append(new_covers)
            for idx in new_covers:
                covered[idx] += 1
        else:
            # Local improvement: for a random block, try all single-point swaps to minimize uncovered
            if not blocks:
                continue
            bi = rng.randrange(len(blocks))
            old_block = blocks[bi]
            old_covers = block_covers[bi]
            best_swap = None
            best_delta = 0
            for pos in range(k):
                old_point = old_block[pos]
                for x in range(v):
                    if x == old_point or x in old_block:
                        continue
                    new_block_list = list(old_block)
                    new_block_list[pos] = x
                    new_block_list.sort()
                    new_block = tuple(new_block_list)
                    if new_block == old_block:
                        continue
                    new_covers = block_cover_indices(new_block)
                    old_set = set(old_covers)
                    new_set = set(new_covers)
                    only_old = old_set - new_set
                    only_new = new_set - old_set
                    delta = 0
                    for idx in only_old:
                        if covered[idx] == 1:
                            delta += 1
                    for idx in only_new:
                        if covered[idx] == 0:
                            delta -= 1
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (new_block, new_covers)
            if best_swap is not None and best_delta < 0:
                new_block, new_covers = best_swap
                for idx in old_covers:
                    covered[idx] -= 1
                for idx in new_covers:
                    covered[idx] += 1
                blocks[bi] = new_block
                block_covers[bi] = new_covers

        # Update best if fully covered and fewer blocks
        n_uncov = evaluate_delta_cover(covered)
        if n_uncov == 0 and len(blocks) < best_n_blocks:
            best_n_blocks = len(blocks)
            best_blocks = list(blocks)
            best_block_covers = list(block_covers)
            best_covered = covered.copy()

        # Cool down
        T_start *= 0.999

    # Final cleanup: remove redundant blocks from best solution
    # Work on a copy
    final_blocks = list(best_blocks)
    final_covered = best_covered.copy()
    final_covers = [block_cover_indices(b) for b in final_blocks]

    changed = True
    while changed:
        changed = False
        for i in range(len(final_blocks)):
            redundant = True
            for idx in final_covers[i]:
                if final_covered[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for idx in final_covers[i]:
                    final_covered[idx] -= 1
                final_blocks.pop(i)
                final_covers.pop(i)
                changed = True
                break

    # Verify validity
    # Check all t-subsets are covered
    assert int(np.sum(final_covered == 0)) == 0, "Invalid covering: some t-subsets uncovered"

    # Return as list of lists
    return [list(b) for b in final_blocks]
