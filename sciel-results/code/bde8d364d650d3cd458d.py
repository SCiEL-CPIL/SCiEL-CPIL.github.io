"""Combined covering design: fast greedy with index-based coverage tracking,
plus redundancy removal and swap-based local optimization."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    all_points = range(v)
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Track coverage: covered[i] is True if t-subset i is covered
    covered = bytearray(n_t)  # memory-efficient
    uncovered_count = n_t
    blocks = []

    def time_ok():
        return time.time() < deadline

    def count_uncovered_in_block(block):
        """Count how many t-subsets of block are currently uncovered."""
        cnt = 0
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            if not covered[idx]:
                cnt += 1
        return cnt

    # Phase 1: Greedy construction
    while uncovered_count > 0 and time_ok():
        # Find the first uncovered t-subset
        first_uncovered = None
        for i in range(n_t):
            if not covered[i]:
                first_uncovered = i
                break
        if first_uncovered is None:
            break

        base = list(all_t_subs[first_uncovered])
        block = list(base)
        block_set = set(block)

        while len(block) < k and time_ok():
            best_x = -1
            best_gain = -1
            # Precompute t-1 subsets of current block
            t_minus_1_list = list(itertools.combinations(block, t - 1))

            for x in all_points:
                if x in block_set:
                    continue
                gain = 0
                for rest in t_minus_1_list:
                    cand = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx.get(cand)
                    if idx is not None and not covered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                # Fallback: fill with arbitrary points
                for x in all_points:
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                break

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Mark newly covered subsets
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            if not covered[idx]:
                covered[idx] = 1
                uncovered_count -= 1

    if uncovered_count > 0:
        # Can't finish in time; return what we have (may be invalid)
        return blocks

    # Phase 2: Remove redundant blocks
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(blocks)):
            if not time_ok():
                break
            redundant = True
            for sub in itertools.combinations(blocks[i], t):
                idx = t_sub_to_idx[sub]
                found = False
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if all(pt in blocks[j] for pt in sub):
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                break

    # Phase 3: Swap-based local optimization
    # Try to reduce block count by finding blocks whose removal can be compensated
    # by modifying other blocks, or simply remove if truly redundant after swaps.
    # Simpler: try swapping one point in a block for another to increase coverage
    # of remaining uncovered subsets, then re-check redundancy.
    if time_ok():
        improved = True
        while improved and time_ok():
            improved = False
            for i in range(len(blocks)):
                if not time_ok():
                    break
                block_i = list(blocks[i])
                for pos in range(k):
                    old_pt = block_i[pos]
                    for new_pt in all_points:
                        if new_pt == old_pt or new_pt in block_i:
                            continue
                        candidate = block_i[:]
                        candidate[pos] = new_pt
                        candidate.sort()
                        # Check if this swap doesn't lose any currently-covered
                        # subset that isn't covered elsewhere
                        # For simplicity, just check if the new block covers
                        # at least as many uncovered subsets as the old
                        old_uncovered = 0
                        for sub in itertools.combinations(block_i, t):
                            idx = t_sub_to_idx[sub]
                            if not covered[idx]:
                                old_uncovered += 1
                        new_uncovered = 0
                        for sub in itertools.combinations(candidate, t):
                            idx = t_sub_to_idx[sub]
                            if not covered[idx]:
                                new_uncovered += 1
                        if new_uncovered > old_uncovered:
                            blocks[i] = candidate
                            # Update coverage
                            for sub in itertools.combinations(block_i, t):
                                idx = t_sub_to_idx[sub]
                                if not covered[idx]:
                                    # Check if still covered by another block
                                    still_covered = False
                                    for j in range(len(blocks)):
                                        if j == i:
                                            continue
                                        if all(pt in blocks[j] for pt in sub):
                                            still_covered = True
                                            break
                                    if not still_covered:
                                        covered[idx] = 0
                                        uncovered_count += 1
                            for sub in itertools.combinations(candidate, t):
                                idx = t_sub_to_idx[sub]
                                if not covered[idx]:
                                    covered[idx] = 1
                                    uncovered_count -= 1
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return blocks
# EVOLVE-BLOCK-END
