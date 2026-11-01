"""Covering design C(24,7,3): randomized greedy + local search with redundancy-aware deletion."""
import random
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START


def _block_triples(block, t):
    """Yield all t-subsets of a block."""
    return itertools.combinations(block, t)


def _is_valid(blocks, v, k, t):
    """Check that blocks form a valid covering of all t-subsets."""
    covered = set()
    for b in blocks:
        if len(b) != k or len(set(b)) != k:
            return False
        for x in b:
            if x < 0 or x >= v:
                return False
        covered.update(itertools.combinations(b, t))
    total = 1
    for i in range(v - t + 1, v + 1):
        total *= i
    for i in range(1, t + 1):
        total //= i
    return len(covered) == total


def _local_search_remove_redundant(blocks, v, k, t, all_tsub_set):
    """Iteratively remove blocks that are fully redundant (all their t-subsets covered by others).
    Uses an index map for efficiency. Returns the reduced list of blocks."""
    if not blocks:
        return blocks

    # Build mapping from each t-subset to the set of block indices covering it
    def build_index(blist):
        idx = defaultdict(set)
        for i, b in enumerate(blist):
            for tr in itertools.combinations(b, t):
                idx[tr].add(i)
        return idx

    idx = build_index(blocks)
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for i, b in enumerate(blocks):
            # Check if every t-subset in this block is covered by at least one other block
            redundant = True
            for tr in itertools.combinations(b, t):
                if len(idx[tr]) <= 1:
                    redundant = False
                    break
            if redundant:
                to_remove.add(i)

        if to_remove:
            # Remove all redundant blocks at once
            new_blocks = [b for i, b in enumerate(blocks) if i not in to_remove]
            blocks = new_blocks
            idx = build_index(blocks)
            changed = True

    return blocks


def _try_swap_reduce(blocks, v, k, t, rng, all_tsub_set):
    """Try to swap a point in a block to create redundancy, then delete.
    Returns new blocks list if an improving swap is found, else None."""
    if len(blocks) < 2:
        return None

    # Compute cover counts
    cover_count = defaultdict(int)
    for b in blocks:
        for tr in itertools.combinations(b, t):
            cover_count[tr] += 1

    n = len(blocks)
    # Try random point swaps: replace point p in block i with point q not in block i
    for _ in range(80):
        bi = rng.randrange(n)
        block = list(blocks[bi])
        p_idx = rng.randrange(k)
        p = block[p_idx]
        q = rng.randrange(v)
        if q == p or q in block:
            continue
        new_block = block[:p_idx] + [q] + block[p_idx + 1:]
        new_block.sort()

        # Check: does the new block still cover all t-subsets that were ONLY covered by old block?
        old_triples = set(itertools.combinations(block, t))
        new_triples = set(itertools.combinations(new_block, t))
        lost = old_triples - new_triples
        can_swap = True
        for tr in lost:
            if cover_count[tr] < 2:
                can_swap = False
                break
        if not can_swap:
            continue

        # Perform swap and try to delete a block
        new_blocks = list(blocks)
        new_blocks[bi] = new_block

        # Recompute cover counts quickly
        new_cover_count = defaultdict(int)
        for b in new_blocks:
            for tr in itertools.combinations(b, t):
                new_cover_count[tr] += 1

        # Try to find a removable block
        for i in range(len(new_blocks)):
            removable = True
            for tr in itertools.combinations(new_blocks[i], t):
                if new_cover_count[tr] < 2:
                    removable = False
                    break
            if removable:
                return new_blocks[:i] + new_blocks[i + 1:]

    return None


def _randomized_greedy(v, k, t, rng, deadline, all_tsub_set):
    """Build a covering using randomized greedy with randomization in point selection."""
    all_points = list(range(v))
    uncovered = set(all_tsub_set)
    blocks = []

    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset to start the block
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)

        while len(block) < k:
            # For each candidate point, compute how many new t-subsets it covers
            best_gain = -1
            best_candidates = []
            block_set = set(block)
            for x in all_points:
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tr = tuple(sorted(rest + (x,)))
                    if tr in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(x)
            if not best_candidates:
                # No point helps; pick random
                remaining = [x for x in all_points if x not in block_set]
                if not remaining:
                    break
                block.append(rng.choice(remaining))
            else:
                # Randomize among top candidates
                block.append(rng.choice(best_candidates))
            block_set = set(block)

        block.sort()
        blocks.append(block)
        # Update uncovered
        for tr in itertools.combinations(block, t):
            uncovered.discard(tr)

    return blocks


def _deterministic_greedy(v, k, t, all_tsub_set):
    """Deterministic greedy baseline."""
    uncovered = set(all_tsub_set)
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        block_set = set(block)
        while len(block) < k:
            best_gain = -1
            best_x = None
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tr = tuple(sorted(rest + (x,)))
                    if tr in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set = set(block)
        block.sort()
        blocks.append(block)
        for tr in itertools.combinations(block, t):
            uncovered.discard(tr)
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    all_tsub_set = set(itertools.combinations(range(v), t))

    # Phase 1: Deterministic greedy baseline
    best_blocks = _deterministic_greedy(v, k, t, all_tsub_set)
    best_count = len(best_blocks)

    # Phase 2: Randomized greedy with multiple seeds
    for seed in range(300):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed * 7919 + 1)
        blocks = _randomized_greedy(v, k, t, rng, deadline - 10, all_tsub_set)
        if blocks:
            # Apply local search to remove redundant blocks
            blocks = _local_search_remove_redundant(blocks, v, k, t, all_tsub_set)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks

    # Phase 3: Local search on the best solution found
    current = list(best_blocks)
    current_count = len(current)
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False

        # Try local search removal (batch)
        reduced = _local_search_remove_redundant(current, v, k, t, all_tsub_set)
        if len(reduced) < len(current):
            current = reduced
            current_count = len(current)
            improved = True
            no_improve_count = 0
            continue

        # Try point swaps to enable deletion
        rng_ls = random.Random(int(time.time() * 1000000))
        new_blocks = _try_swap_reduce(current, v, k, t, rng_ls, all_tsub_set)
        if new_blocks is not None:
            current = new_blocks
            current_count = len(current)
            improved = True
            no_improve_count = 0
            continue

        # Try a fresh randomized greedy restart if stuck
        if no_improve_count % 5 == 0:
            rng_restart = random.Random(int(time.time() * 1000000) + 999)
            blocks_restart = _randomized_greedy(v, k, t, rng_restart, deadline, all_tsub_set)
            if blocks_restart:
                blocks_restart = _local_search_remove_redundant(blocks_restart, v, k, t, all_tsub_set)
                if len(blocks_restart) < current_count:
                    current = blocks_restart
                    current_count = len(current)
                    improved = True
                    no_improve_count = 0
                    continue

        no_improve_count += 1

    # Final validation
    if not _is_valid(current, v, k, t):
        return best_blocks

    return current
# EVOLVE-BLOCK-END
