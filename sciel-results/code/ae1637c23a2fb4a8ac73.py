"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    # Precompute all t-subsets as tuples
    all_tsubs = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # For each point, precompute the list of t-subset indices that contain it
    point_to_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_to_tsubs[p].append(i)

    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(n_tsubs))
    blocks = []

    while uncovered:
        if time.time() > deadline:
            # If we run out of time, we must still return a valid covering.
            # But we can't return an invalid one.
            # However, the problem says "Valid when every t-subset ... lies in at least one block".
            # If we break early, the covering is invalid.
            # So we must ensure we finish.
            # But the deadline is a guard against the evaluator killing us.
            # If we are close to the deadline, we should finish the current block and stop if possible?
            # No, if we stop, it's invalid.
            # The only safe action is to continue until done, but we risk timeout.
            # However, the greedy algorithm is usually fast enough for v=19, k=7, t=4.
            # Let's assume it finishes. If not, the evaluator kills it and score is 0.
            pass

        # Pick the first uncovered t-subset to start a new block
        start_ts = min(uncovered)
        start_tuple = all_tsubs[start_ts]
        block = list(start_tuple)

        # Greedily add points to the block
        while len(block) < k:
            if time.time() > deadline:
                # Can't safely abort. Just proceed.
                pass

            # Find the point that covers the most uncovered t-subsets
            best_point = -1
            best_gain = -1
            # Precompute the (t-1)-subsets of the current block
            block_tminus1 = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in block_tminus1:
                    # Check if rest + (x,) is an uncovered t-subset
                    ts = tuple(sorted(rest + (x,)))
                    if ts in tsub_index:
                        idx = tsub_index[ts]
                        if idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # Should not happen if v >= k and uncovered is not empty
                break
            block.append(best_point)

        block.sort()
        blocks.append(block)

        # Remove all t-subsets covered by this block
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            uncovered.discard(idx)

    return blocks
# EVOLVE-BLOCK-END
