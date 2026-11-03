# EVOLVE-BLOCK-START
import itertools
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    total_tsubs = len(all_tsubs)

    # coverage[tsub] = number of k-blocks (among those already chosen) that contain tsub
    coverage = defaultdict(int)
    uncovered = set(all_tsubs)  # t-subsets not yet covered

    # Precompute: for each t-subset, the list of k-subsets that contain it? Too big.
    # Instead, for a given t-subset S, the k-subsets containing it are:
    #   S union (k-t) subset of (points \ S)
    # Count = C(v - t, k - t)
    from math import comb
    remaining_options = comb(v - t, k - t)

    blocks = []
    block_set = set()

    def time_left():
        return deadline - time.time()

    # Greedy phase
    while uncovered and time_left() > 5.0:
        # Pick the uncovered t-subset with the fewest remaining covering k-subsets
        # (i.e., the one that is hardest to cover). Approximate by: pick min uncovered.
        # For speed, just pick the lexicographically smallest uncovered (seed behavior)
        # but we can do a small random sample to find a "harder" one.
        # Use deterministic: min uncovered for reproducibility and speed.
        seed_tsub = min(uncovered)

        # Build a block containing seed_tsub, greedily adding points that maximize
        # the number of newly-covered uncovered t-subsets.
        block = list(seed_tsub)
        block_set_local = set(block)

        # For each candidate point x not in block, compute how many uncovered t-subsets
        # would be covered by adding x. A t-subset is covered by block if it's a subset of block.
        # When we add x, the new t-subsets covered are those t-subsets that:
        #   (a) are in uncovered,
        #   (b) are subsets of (block + [x]),
        #   (c) were NOT subsets of block (i.e., they include x).
        # So: t-subsets T in uncovered such that T ⊆ (block ∪ {x}) and x ∈ T.
        # Equivalently: T = (T ∩ block) ∪ {x} where T ∩ block has size t-1 and T ∩ block ⊆ block.

        # Precompute for the current block: all (t-1)-subsets of block
        block_t1subs = list(itertools.combinations(block, t - 1))

        # For each candidate x, count uncovered t-subsets that contain x and whose other t-1 elements are in block
        # i.e., for each (t-1)-subset S of block, the t-subset S ∪ {x} is covered iff it's in uncovered.
        # So gain(x) = count of S in C(block, t-1) such that tuple(S + (x,)) in uncovered (sorted)

        # To speed up, we can precompute a dict: for each (t-1)-subset, which x's make it uncovered?
        # That's expensive. Instead, iterate candidates.

        candidates = [x for x in points if x not in block_set_local]

        # Compute gains
        best_x = None
        best_gain = -1
        for x in candidates:
            gain = 0
            for s in block_t1subs:
                # s is a (t-1)-tuple, x is int
                # form t-tuple: sorted(s + (x,))
                tsub = tuple(sorted(s + (x,)))
                if tsub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x

        if best_x is None:
            break

        block.append(best_x)
        block_set_local.add(best_x)

        # Update block_t1subs: add new (t-1)-subsets that include best_x
        # New (t-1)-subs containing best_x: (t-2)-subsets of old block + best_x
        old_block = block[:-1]
        for s2 in itertools.combinations(old_block, t - 2):
            block_t1subs.append(tuple(sorted(s2 + (best_x,))))

        # Once block is full, we don't need to add more
        if len(block) >= k:
            break

    # If we broke early (block not full), fill with arbitrary points
    while len(block) < k:
        for x in points:
            if x not in block_set_local:
                block.append(x)
                block_set_local.add(x)
                break

    block.sort()
    blocks.append(block)
    block_set.add(tuple(block))

    # Update uncovered: remove all t-subsets of this block
    for tsub in itertools.combinations(block, t):
        if tsub in uncovered:
            uncovered.discard(tsub)
            coverage[tsub] += 1  # not strictly needed but for tracking

    # Continue greedy for remaining uncovered
    while uncovered and time_left() > 10.0:
        # Pick hardest uncovered t-subset: approximate by min
        seed_tsub = min(uncovered)
        block = list(seed_tsub)
        block_set_local = set(block)
        block_t1subs = list(itertools.combinations(block, t - 1))

        while len(block) < k:
            candidates = [x for x in points if x not in block_set_local]
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for s in block_t1subs:
                    tsub = tuple(sorted(s + (x,)))
                    if tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                # No beneficial point; pick arbitrary to fill
                best_x = candidates[0] if candidates else None
                if best_x is None:
                    break
            block.append(best_x)
            block_set_local.add(best_x)
            old_block = block[:-1]
            for s2 in itertools.combinations(old_block, t - 2):
                block_t1subs.append(tuple(sorted(s2 + (best_x,))))

        while len(block) < k:
            for x in points:
                if x not in block_set_local:
                    block.append(x)
                    block_set_local.add(x)
                    break

        block.sort()
        bt = tuple(block)
        if bt in block_set:
            # Duplicate; try to modify? For now, skip and continue
            continue
        blocks.append(block)
        block_set.add(bt)

        for tsub in itertools.combinations(block, t):
            if tsub in uncovered:
                uncovered.discard(tsub)

        if time_left() < 10.0:
            break

    # Local search: try to improve by swapping points in blocks to cover more t-subsets
    # We track coverage counts to know which t-subsets are covered exactly once (fragile)
    # and which are uncovered (shouldn't be after greedy, but just in case)

    # Rebuild coverage from blocks
    coverage = defaultdict(int)
    for b in blocks:
        for tsub in itertools.combinations(b, t):
            coverage[tsub] += 1

    # Local search: for each block, try replacing one point with another to increase total coverage
    # A swap (remove r from block, add a not in block) is beneficial if it covers new t-subsets
    # without uncovering any (i.e., all t-subsets of new block that were only covered by old block
    # are still covered by other blocks, and new t-subsets add coverage).

    # For speed, limit local search iterations
    ls_budget = time_left()
    if ls_budget > 2.0:
        # Sort blocks by size of their unique contribution (t-subsets covered only by this block)
        # to prioritize blocks that might be improvable
        # For each block, compute unique_tsubs = t-subsets of block with coverage == 1
        # This is expensive; do a few random swaps instead.

        import random
        random.seed(42)
        n_ls_iters = min(200, int(ls_budget * 50))  # up to ~50 per second

        for _ in range(n_ls_iters):
            if time_left() < 2.0:
                break
            bi = random.randrange(len(blocks))
            b = blocks[bi]
            # Pick a random point to remove
            ri = random.randrange(k)
            r = b[ri]
            # Pick a random point to add (not in b)
            candidates = [x for x in points if x not in b]
            if not candidates:
                continue
            a = random.choice(candidates)

            # Compute delta coverage
            # Old block covers: all t-subsets of b
            # New block covers: all t-subsets of (b - {r}) + {a}
            # Delta = new_covered - old_covered (counting only uncovered gains and coverage losses)

            # A t-subset T is lost if T ⊆ b, T ⊇ {r}, and coverage[T] == 1 (only this block covers it)
            # A t-subset T is gained if T ⊆ new_block, T ⊇ {a}, and T was not in b (i.e., T not ⊆ b)
            # and coverage[T] == 0 (uncovered)

            # Compute lost: t-subsets containing r, subset of b, with coverage 1
            lost = 0
            rest = [x for i, x in enumerate(b) if i != ri]
            for s in itertools.combinations(rest, t - 1):
                T = tuple(sorted(s + (r,)))
                if coverage[T] == 1:
                    lost += 1

            # Compute gained: t-subsets containing a, subset of new_block, not subset of b, coverage 0
            new_block = set(b)
            new_block.discard(r)
            new_block.add(a)
            new_block_list = list(new_block)

            gained = 0
            for s in itertools.combinations(new_block_list, t - 1):
                T = tuple(sorted(s + (a,)))
                # T must contain a (it does by construction)
                # T must NOT be subset of b (i.e., T contains a, and a not in b, so automatically not subset of b)
                # Check if T is covered: coverage[T] == 0
                if coverage[T] == 0:
                    gained += 1

            if gained > lost:
                # Apply swap
                blocks[bi] = sorted(new_block_list)
                # Update coverage
                # Remove old block's contributions
                for T in itertools.combinations(b, t):
                    coverage[T] -= 1
                # Add new block's contributions
                for T in itertools.combinations(blocks[bi], t):
                    coverage[T] += 1

    # Final: ensure all t-subsets are covered (greedy should have done this, but verify)
    # If any uncovered, add blocks for them (should be rare)
    final_uncovered = set()
    for T in all_tsubs:
        if coverage[T] == 0:
            final_uncovered.add(T)

    # If there are uncovered t-subsets, add minimal blocks to cover them
    while final_uncovered and time_left() > 1.0:
        T = min(final_uncovered)
        # Find a k-subset containing T that covers the most remaining uncovered
        best_block = None
        best_count = -1
        rest_points = [x for x in points if x not in T]
        # Try all (k-t) subsets of rest_points? C(19-5, 8-5) = C(14,3) = 364, feasible
        for add in itertools.combinations(rest_points, k - t):
            block = list(T) + list(add)
            block.sort()
            count = 0
            for T2 in itertools.combinations(block, t):
                if T2 in final_uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = block
        if best_block is None:
            break
        blocks.append(best_block)
        for T2 in itertools.combinations(best_block, t):
            if T2 in final_uncovered:
                final_uncovered.discard(T2)

    return blocks
# EVOLVE-BLOCK-END
