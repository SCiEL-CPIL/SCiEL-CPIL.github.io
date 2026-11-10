"""Covering design C(v, k, t) – fixed-budget max-coverage stochastic local search."""
import itertools
import time
import random
from collections import defaultdict

_DEADLINE_S = 115.0  # Safety margin from 150s limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _compute_coverage(blocks, v, k, t):
    """Return (covered_count, tsub_to_blocks_dict) for the given blocks."""
    tsub_to_blocks = defaultdict(set)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            tsub_to_blocks[tsub].add(i)
    return len(tsub_to_blocks), tsub_to_blocks


def _prune_redundant(blocks, v, k, t):
    """Remove blocks that are redundant (all their t-subsets covered by others)."""
    if not blocks:
        return blocks

    # Build tsub -> set of block indices
    tsub_to_blocks = defaultdict(set)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            tsub_to_blocks[tsub].add(i)

    n = len(blocks)
    to_remove = set()

    # A block i is redundant if for every tsub in block i, there exists j != i with j in tsub_to_blocks[tsub]
    for i in range(n):
        block = blocks[i]
        redundant = True
        for tsub in itertools.combinations(block, t):
            if len(tsub_to_blocks[tsub]) == 1:
                redundant = False
                break
        if redundant:
            to_remove.add(i)

    if not to_remove:
        return blocks

    # Remove all redundant blocks at once
    new_blocks = [blocks[i] for i in range(n) if i not in to_remove]
    return new_blocks


def _local_search_max_coverage(blocks, v, k, t, max_iter=50000):
    """
    Stochastic local search: fix the number of blocks, mutate individual points
    within blocks to maximize the number of distinct t-subsets covered.
    """
    n = len(blocks)
    if n == 0:
        return blocks

    # Convert to lists of lists for mutability
    blocks = [list(b) for b in blocks]

    # Initial coverage
    covered, tsub_to_blocks = _compute_coverage(blocks, v, k, t)
    total_tsubs = math_comb(v, t)

    # If already complete, just prune
    if covered >= total_tsubs:
        return _prune_redundant(blocks, v, k, t)

    # Precompute all t-subsets for each block index
    block_tsubs = [set(itertools.combinations(b, t)) for b in blocks]

    # Build tsub -> set of block indices
    tsub_to_blocks = defaultdict(set)
    for i, bts in enumerate(block_tsubs):
        for tsub in bts:
            tsub_to_blocks[tsub].add(i)

    best_covered = covered
    best_blocks = [list(b) for b in blocks]

    rng = random.Random(12345)

    for it in range(max_iter):
        if _time_left() < 3.0:
            break

        # Pick a random block to mutate
        i = rng.randrange(n)
        # Pick a random position in the block
        pos = rng.randrange(k)
        old_point = blocks[i][pos]

        # Pick a new point not already in the block
        block_set = set(blocks[i])
        candidates = [p for p in range(v) if p not in block_set]
        if not candidates:
            continue
        new_point = rng.choice(candidates)

        # Compute delta coverage
        # Old t-subsets of block i that contain old_point will be removed
        # New t-subsets of block i that contain new_point will be added
        # Other t-subsets of block i remain unchanged

        # T-subsets of old block i that contain old_point:
        # C(k-1, t-1) of them
        # T-subsets of new block i that contain new_point:
        # C(k-1, t-1) of them

        # For each such tsub, check if it's covered by any other block
        # If yes, removing/adding it doesn't change global coverage
        # If no, it does

        # Efficient delta:
        # For old tsubs containing old_point:
        #   If only block i covers it, we lose 1 coverage
        # For new tsubs containing new_point:
        #   If no other block covers it, we gain 1 coverage

        old_block = blocks[i][:]
        new_block = old_block[:]
        new_block[pos] = new_point

        delta = 0

        # Old tsubs that contain old_point
        # These are t-1 other points from old_block minus old_point, combined with old_point
        old_remaining = [p for j, p in enumerate(old_block) if j != pos]
        for tsub in itertools.combinations(old_remaining, t - 1):
            full_tsub = tuple(sorted(tsub + (old_point,)))
            # Check if any block other than i covers this tsub
            other_blocks = tsub_to_blocks[full_tsub]
            other_blocks.discard(i)
            if not other_blocks:
                delta -= 1

        # New tsubs that contain new_point
        new_remaining = [p for j, p in enumerate(new_block) if j != pos]
        for tsub in itertools.combinations(new_remaining, t - 1):
            full_tsub = tuple(sorted(tsub + (new_point,)))
            # Check if any block other than i covers this tsub
            other_blocks = tsub_to_blocks[full_tsub]
            other_blocks.discard(i)
            if not other_blocks:
                delta += 1

        if delta >= 0:
            # Accept the move
            blocks[i][pos] = new_point

            # Update data structures
            # Remove old tsubs from tsub_to_blocks
            old_remaining = [p for j, p in enumerate(old_block) if j != pos]
            for tsub in itertools.combinations(old_remaining, t - 1):
                full_tsub = tuple(sorted(tsub + (old_point,)))
                tsub_to_blocks[full_tsub].discard(i)

            # Add new tsubs to tsub_to_blocks
            new_remaining = [p for j, p in enumerate(new_block) if j != pos]
            for tsub in itertools.combinations(new_remaining, t - 1):
                full_tsub = tuple(sorted(tsub + (new_point,)))
                tsub_to_blocks[full_tsub].add(i)

            covered += delta
            if covered > best_covered:
                best_covered = covered
                best_blocks = [list(b) for b in blocks]

            # If we've achieved full coverage, we can break and prune
            if covered >= total_tsubs:
                break
        # else: reject the move (no change)

    return best_blocks


def math_comb(n, r):
    if r < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
    return result


def construct(v, k, t):
    random.seed(42)
    rng = random.Random(42)

    # Known best is 151. Try a range of block budgets.
    # The minimum possible is at least ceil(C(v,t) / C(k,t)) = ceil(3876/35) = 111
    # But the repository lower bound is 131, and best known is 151.
    # We'll try budgets from 131 to 160.

    total_tsubs = math_comb(v, t)
    min_blocks = (total_tsubs + math_comb(k, t) - 1) // math_comb(k, t)

    # Try multiple budgets, keep the best (fewest blocks that achieve full coverage)
    best_result = None

    # Start from higher budgets and go down, or try a range
    # For efficiency, try a few key budgets
    budgets_to_try = list(range(131, 161))
    # Shuffle slightly for diversity but keep deterministic
    rng.shuffle(budgets_to_try)

    for B in budgets_to_try:
        if _time_left() < 5.0:
            break

        # Generate random initial blocks
        all_points = list(range(v))
        initial_blocks = []
        used_blocks = set()

        for _ in range(B):
            while True:
                block = tuple(sorted(rng.sample(all_points, k)))
                if block not in used_blocks:
                    used_blocks.add(block)
                    initial_blocks.append(list(block))
                    break

        # Run local search
        refined = _local_search_max_coverage(initial_blocks, v, k, t, max_iter=20000)

        # Prune redundant blocks
        pruned = _prune_redundant(refined, v, k, t)

        # Verify coverage
        if pruned:
            covered, _ = _compute_coverage(pruned, v, k, t)
            if covered >= total_tsubs:
                if best_result is None or len(pruned) < len(best_result):
                    best_result = pruned
                    # If we hit the theoretical lower bound or a very good number, we can stop early
                    if len(pruned) <= 135:
                        break
        else:
            # If pruning removed everything (shouldn't happen if B is large enough), skip
            pass

    # Fallback: if no full coverage found, use a greedy construction as backup
    if best_result is None:
        # Simple greedy fallback
        all_tsubs = set(itertools.combinations(range(v), t))
        uncovered = set(all_tsubs)
        blocks = []
        all_ksubs = list(itertools.combinations(range(v), k))

        while uncovered:
            if _time_left() < 2.0:
                break
            best_block = None
            best_gain = -1
            for ksub in all_ksubs:
                gain = sum(1 for tsub in itertools.combinations(ksub, t) if tsub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = ksub
            if best_block is None:
                break
            blocks.append(list(best_block))
            for tsub in itertools.combinations(best_block, t):
                uncovered.discard(tsub)

        if blocks:
            best_result = blocks

    return best_result if best_result else []
