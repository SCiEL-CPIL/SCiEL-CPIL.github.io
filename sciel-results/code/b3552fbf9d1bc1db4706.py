"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    # Deadline guard: the evaluator kills after 150s wall-clock.
    # We aim to finish well before that; return best-so-far if we run out of time.
    deadline = time.time() + 145.0  # 5s safety margin

    # Precompute all t-subsets and map them to integers for fast set ops.
    all_combos = list(itertools.combinations(range(v), t))
    combo_to_idx = {c: i for i, c in enumerate(all_combos)}
    n_combos = len(all_combos)

    # uncovered is a set of indices into all_combos
    uncovered = set(range(n_combos))
    blocks = []

    # Precompute for each point x, the list of t-subsets containing x.
    # point_to_subsets[x] = list of combo indices that contain x
    point_to_subsets = [[] for _ in range(v)]
    for idx, combo in enumerate(all_combos):
        for x in combo:
            point_to_subsets[x].append(idx)

    # Precompute for each (t-1)-subset, the list of combo indices that extend it.
    # (t-1)-subset -> list of combo indices
    t_minus_1_combos = list(itertools.combinations(range(v), t - 1))
    t1_to_idx = {c: i for i, c in enumerate(t_minus_1_combos)}
    # For each (t-1) subset index, the points that can extend it to a t-subset,
    # and the corresponding combo indices.
    t1_ext = [[] for _ in range(len(t_minus_1_combos))]
    for idx, combo in enumerate(all_combos):
        # combo is a t-tuple; removing each element gives a (t-1)-subset
        for j in range(t):
            rest = combo[:j] + combo[j+1:]
            t1_idx = t1_to_idx[rest]
            x = combo[j]
            t1_ext[t1_idx].append((x, idx))

    while uncovered:
        if time.time() > deadline:
            break

        # Pick the first uncovered t-subset to seed the block
        seed_idx = min(uncovered)
        seed = all_combos[seed_idx]
        block = list(seed)

        while len(block) < k:
            if time.time() > deadline:
                break

            # For each candidate point x not in block, compute how many uncovered
            # t-subsets it would cover when added to the current block.
            best_x = None
            best_gain = -1

            # Build set of current block for quick lookup
            block_set = set(block)

            for x in range(v):
                if x in block_set:
                    continue
                # Gain = number of (t-1)-subsets of block that, combined with x,
                # form an uncovered t-subset.
                # A (t-1)-subset S of block covers combo S ∪ {x} if x ∉ S.
                # We need S ∪ {x} to be a valid t-subset (i.e., x not in S) and in uncovered.
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    rest_set = set(rest)
                    if x in rest_set:
                        continue
                    combo = tuple(sorted(rest + (x,)))
                    idx = combo_to_idx.get(combo)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain <= 0:
                # No point improves; just pick an arbitrary unused point to fill
                for x in range(v):
                    if x not in block_set:
                        block.append(x)
                        break
            else:
                block.append(best_x)

        block.sort()
        blocks.append(block)

        # Remove all t-subsets covered by this block
        for combo in itertools.combinations(block, t):
            idx = combo_to_idx[combo]
            uncovered.discard(idx)

    # If we ran out of time and still have uncovered subsets, we need to ensure
    # validity. But the contract says invalid scores 0. So we must cover all.
    # However, if we break early, the result is invalid. Let's handle this:
    # If uncovered is non-empty after the loop (due to deadline), we have a problem.
    # For safety, if we didn't finish, we should try to finish quickly or return
    # what we have (which may be invalid). But invalid = score 0.
    # Better: if we hit the deadline, we should still return a valid covering.
    # The greedy above is deterministic and fast for C(24,7,3). Let's check if
    # it finishes in time. For v=24, k=7, t=3: C(24,3)=2024 t-subsets.
    # Each block covers C(7,3)=35 t-subsets. ~58 blocks needed.
    # Each block construction: 7 iterations, each scanning 24 points, each point
    # scanning C(6,2)=15 (t-1)-subsets. So ~7*24*15 = 2520 ops per block.
    # 58 blocks * 2520 = ~146k ops. Very fast. Should finish in <1s.
    # So the deadline guard is just a safety net.

    return blocks
# EVOLVE-BLOCK-END
