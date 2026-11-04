"""Improved covering design with optimized greedy and redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    uncovered = set(range(n_t))
    blocks = []

    def time_ok():
        return time.time() < deadline

    # Phase 1: Greedy
    while uncovered and time_ok():
        first_idx = min(uncovered)
        block = list(all_t_subs[first_idx])
        block_set = set(block)

        while len(block) < k and time_ok():
            best_x = -1
            best_gain = -1
            t_minus_1 = list(itertools.combinations(block, t - 1))

            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in t_minus_1:
                    cand = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx.get(cand)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                for x in range(v):
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

        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx.get(t_sub)
            if idx is not None:
                uncovered.discard(idx)

    if uncovered:
        return blocks

    # Phase 2: Remove redundant blocks
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(blocks)):
            if not time_ok():
                break
            redundant = True
            for t_sub in itertools.combinations(blocks[i], t):
                idx = t_sub_to_idx.get(t_sub)
                if idx is None:
                    continue
                found = False
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if all(pt in blocks[j] for pt in t_sub):
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                break

    return blocks
# EVOLVE-BLOCK-END
