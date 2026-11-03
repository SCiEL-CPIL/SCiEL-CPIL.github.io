"""Improved greedy covering with time guard and local repair."""
# EVOLVE-BLOCK-START
import itertools
import time


def _time_left(start, limit=110.0):
    return limit - (time.time() - start)


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for fast membership
    tsub_index = {sub: i for i, sub in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # covered[i] = True if t-subset i is covered
    covered = bytearray(n_tsubs)

    # For each point, list of t-subsets containing it (for fast gain computation)
    # point_tsubs[x] = list of indices of t-subsets that contain x
    point_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_tsubs):
        for x in sub:
            point_tsubs[x].append(i)

    blocks = []

    def covered_count_in_block(block):
        """Count how many t-subsets of block are covered."""
        cnt = 0
        for sub in itertools.combinations(block, t):
            idx = tsub_index[sub]
            if covered[idx]:
                cnt += 1
        return cnt

    def uncovered_in_block(block):
        """Count how many t-subsets of block are NOT covered."""
        total = len(list(itertools.combinations(block, t)))
        cnt = 0
        for sub in itertools.combinations(block, t):
            idx = tsub_index[sub]
            if not covered[idx]:
                cnt += 1
        return cnt

    # Greedy: repeatedly pick the block that covers the most uncovered t-subsets
    # Use a two-phase approach:
    #   Phase 1: Greedy set cover with good block selection
    #   Phase 2: Local repair to fix any remaining uncovered t-subsets

    while _time_left(start) > 5.0:
        # Find the first uncovered t-subset
        first_uncovered = None
        for i in range(n_tsubs):
            if not covered[i]:
                first_uncovered = i
                break
        if first_uncovered is None:
            break

        # Build a block starting from this uncovered t-subset
        seed = list(all_tsubs[first_uncovered])

        # Greedily add points to maximize coverage of uncovered t-subsets
        block = list(seed)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: how many new uncovered t-subsets would be covered
                # by adding x to block
                gain = 0
                # For each (t-1)-subset of current block, check if adding x creates
                # a t-subset that is uncovered
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = tsub_index[sub]
                    if not covered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()

        # Check if this block covers anything new
        new_covered = 0
        for sub in itertools.combinations(block, t):
            idx = tsub_index[sub]
            if not covered[idx]:
                new_covered += 1
                covered[idx] = 1

        if new_covered > 0:
            blocks.append(block)

    # Local repair: for any remaining uncovered t-subsets, create blocks covering them
    if _time_left(start) > 2.0:
        still_uncovered = []
        for i in range(n_tsubs):
            if not covered[i]:
                still_uncovered.append(i)

        for i in still_uncovered:
            if _time_left(start) < 1.0:
                break
            sub = list(all_tsubs[i])
            # Try to extend this t-subset to a k-block that covers it
            # Use the existing greedy extension
            block = list(sub)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        idx = tsub_index[s]
                        if not covered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            # Mark coverage
            for s in itertools.combinations(block, t):
                idx = tsub_index[s]
                if not covered[idx]:
                    covered[idx] = 1
            if block not in blocks:
                blocks.append(block)

    return blocks
# EVOLVE-BLOCK-END
