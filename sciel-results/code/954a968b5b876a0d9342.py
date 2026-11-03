import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    # Deadline with safety margin
    deadline = time.time() + 115

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute which t-subsets contain it
    point_tsub_lists = [[] for _ in range(v)]
    for ts in all_tsubs:
        for x in ts:
            point_tsub_lists[x].append(tsub_to_idx[ts])

    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)

    def _remove_redundant(blocks, deadline):
        if not blocks:
            return blocks

        block_tsub_indices = []
        cover_count = [0] * n_tsubs

        for block in blocks:
            tsubs = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    tsubs.append(idx)
                    cover_count[idx] += 1
            block_tsub_indices.append(tsubs)

        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3:
                    break
                is_redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsub_indices.pop(i)
                    changed = True
                    break

        return blocks

    def _local_search(blocks, deadline):
        if not blocks:
            return blocks

        block_tsub_indices = []
        cover_count = [0] * n_tsubs
        for block in blocks:
            tsubs = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    tsubs.append(idx)
                    cover_count[idx] += 1
            block_tsub_indices.append(tsubs)

        improved = True
        max_iterations = 50
        iter_count = 0

        while improved and time.time() < deadline - 8 and iter_count < max_iterations:
            improved = False
            iter_count += 1

            for i in range(len(blocks)):
                if time.time() > deadline - 8:
                    break

                criticals = [idx for idx in block_tsub_indices[i] if cover_count[idx] == 1]

                if not criticals:
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsub_indices.pop(i)
                    improved = True
                    break

                found_swap = False
                for crit_idx in criticals:
                    if found_swap:
                        break
                    if time.time() > deadline - 8:
                        break

                    crit_tsub = all_tsubs[crit_idx]
                    crit_tsub_set = set(crit_tsub)

                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if time.time() > deadline - 8:
                            break
                        if found_swap:
                            break

                        block_j = blocks[j]
                        block_j_set = set(block_j)
                        common = block_j_set.intersection(crit_tsub_set)

                        if len(common) >= t - 1:
                            missing = [p for p in crit_tsub if p not in block_j_set]
                            if len(missing) == 1:
                                extra = [p for p in block_j if p not in common]
                                if extra:
                                    for rem_p in extra:
                                        if found_swap:
                                            break
                                        new_block_j = [p if p != rem_p else missing[0] for p in block_j]
                                        old_tsubs_set = set(block_tsub_indices[j])
                                        new_tsubs_set = set()
                                        for ts in itertools.combinations(new_block_j, t):
                                            idx2 = tsub_to_idx.get(tuple(ts))
                                            if idx2 is not None:
                                                new_tsubs_set.add(idx2)
                                        can_swap = True
                                        for idx2 in old_tsubs_set:
                                            if idx2 not in new_tsubs_set and cover_count[idx2] == 1:
                                                can_swap = False
                                                break
                                        if can_swap:
                                            for idx2 in old_tsubs_set:
                                                cover_count[idx2] -= 1
                                            for idx2 in new_tsubs_set:
                                                cover_count[idx2] += 1
                                            blocks[j] = new_block_j
                                            block_tsub_indices[j] = list(new_tsubs_set)
                                            improved = True
                                            found_swap = True
                                            break

                            if not found_swap and len(common) >= t - 2:
                                missing = [p for p in crit_tsub if p not in block_j_set]
                                if len(missing) == 2:
                                    extra = [p for p in block_j if p not in common]
                                    if len(extra) >= 2:
                                        for rem1, rem2 in itertools.combinations(extra, 2):
                                            if found_swap:
                                                break
                                            new_block_j = [p for p in block_j if p != rem1 and p != rem2] + missing
                                            if len(set(new_block_j)) != k:
                                                continue
                                            old_tsubs_set = set(block_tsub_indices[j])
                                            new_tsubs_set = set()
                                            for ts in itertools.combinations(new_block_j, t):
                                                idx2 = tsub_to_idx.get(tuple(ts))
                                                if idx2 is not None:
                                                    new_tsubs_set.add(idx2)
                                            can_swap = True
                                            for idx2 in old_tsubs_set:
                                                if idx2 not in new_tsubs_set and cover_count[idx2] == 1:
                                                    can_swap = False
                                                    break
                                            if can_swap:
                                                for idx2 in old_tsubs_set:
                                                    cover_count[idx2] -= 1
                                                for idx2 in new_tsubs_set:
                                                    cover_count[idx2] += 1
                                                blocks[j] = new_block_j
                                                block_tsub_indices[j] = list(new_tsubs_set)
                                                improved = True
                                                found_swap = True
                                                break

                if found_swap:
                    break

        return blocks

    def _greedy_construct(uncovered_set, deadline, seed_idx, rng):
        blocks = []
        uncovered = set(uncovered_set)

        while uncovered:
            if time.time() > deadline - 10:
                break

            # Pick the hardest uncovered t-subset to seed a new block
            # Use a heuristic: prefer t-subsets whose points are less frequently used
            # For speed, just pick from uncovered
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)

            while len(block) < k:
                best_gain = -1
                best_point = -1
                tie_points = []

                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1

                    # Small bonus for global potential
                    global_potential = sum(1 for idx in point_tsub_lists[x] if idx in uncovered)
                    total_score = gain * 1000 + global_potential

                    if total_score > best_gain:
                        best_gain = total_score
                        best_point = x
                        tie_points = [x]
                    elif total_score == best_gain and total_score > -1:
                        tie_points.append(x)

                if best_point >= 0:
                    if len(tie_points) > 1:
                        best_point = rng.choice(tie_points)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(rng.choice(remaining))
                        block_set.add(block[-1])
                    else:
                        break

            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    uncovered.discard(idx)

        return blocks, uncovered

    # Main loop: multiple randomized restarts
    for restart in range(150):
        if time.time() > deadline - 25:
            break

        uncovered = set(range(n_tsubs))
        order = list(range(n_tsubs))
        rng.shuffle(order)

        # Greedy construction
        blocks, uncovered = _greedy_construct(uncovered, deadline, order[0], rng)

        if time.time() > deadline - 15 or uncovered:
            continue

        # Post-processing
        blocks = _remove_redundant(blocks, deadline)

        if time.time() < deadline - 20:
            blocks = _local_search(blocks, deadline)
            blocks = _remove_redundant(blocks, deadline)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 325:
                break

    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            seed_idx = min(uncovered)
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(remaining[0])
                        block_set.add(block[-1])
                    else:
                        break
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    uncovered.discard(idx)
        best_blocks = blocks

    # Final cleanup
    best_blocks = _remove_redundant(best_blocks, deadline)

    return best_blocks
