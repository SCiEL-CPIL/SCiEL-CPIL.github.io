"""Fast bitmask-based greedy covering with deadline guard for C(v,k,t)."""
# EVOLVE-BLOCK-START
import time
import itertools


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # uncovered as a set of indices
    uncovered = set(range(n_t))

    # Precompute: for each point x, the list of t-subset indices containing x
    point_t_subsets = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for x in s:
            point_t_subsets[x].append(i)

    blocks = []
    seen_blocks = set()

    def time_ok():
        return time.time() < deadline

    def compute_gain(x, block_set):
        """Count uncovered t-subsets that contain x and whose other t-1 points are in block_set."""
        count = 0
        for rest in itertools.combinations(block_set, t - 1):
            s = tuple(sorted(rest + (x,)))
            idx = t_subset_to_idx.get(s)
            if idx is not None and idx in uncovered:
                count += 1
        return count

    def update_uncovered(block):
        """Remove from uncovered all t-subsets that are subsets of block."""
        block_set = set(block)
        to_remove = []
        for i in uncovered:
            if all(x in block_set for x in all_t_subsets[i]):
                to_remove.append(i)
        for i in to_remove:
            uncovered.discard(i)

    while uncovered and time_ok():
        first_idx = min(uncovered)
        first_subset = all_t_subsets[first_idx]
        block = list(first_subset)
        block_set = set(block)

        while len(block) < k and time_ok():
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                g = compute_gain(x, block_set)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1 or best_gain <= 0:
                break
            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        block_tuple = tuple(block)
        if block_tuple not in seen_blocks:
            seen_blocks.add(block_tuple)
            blocks.append(block)
            update_uncovered(block)

    # Repair: cover any remaining uncovered t-subsets
    if uncovered:
        remaining = sorted(uncovered)
        for idx in remaining:
            if not time_ok():
                break
            if idx not in uncovered:
                continue
            s = all_t_subsets[idx]
            block = list(s)
            block_set = set(block)
            while len(block) < k and time_ok():
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    g = compute_gain(x, block_set)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x == -1 or best_gain <= 0:
                    for x in range(v):
                        if x not in block_set:
                            best_x = x
                            break
                    if best_x == -1:
                        break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            block_tuple = tuple(block)
            if block_tuple not in seen_blocks:
                seen_blocks.add(block_tuple)
                blocks.append(block)
                update_uncovered(block)

    return blocks
# EVOLVE-BLOCK-END
