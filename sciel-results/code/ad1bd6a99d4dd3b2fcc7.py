import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 112

    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)

    def _get_block_tsubs(block):
        """Return list of t-subset indices covered by this block."""
        result = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(tuple(ts))
            if idx is not None:
                result.append(idx)
        return result

    def _remove_redundant(blocks, deadline):
        """Remove blocks that don't uniquely cover any t-subset."""
        if not blocks:
            return blocks

        block_tsub_indices = [_get_block_tsubs(b) for b in blocks]
        cover_count = [0] * n_tsubs
        for tsubs in block_tsub_indices:
            for idx in tsubs:
                cover_count[idx] += 1

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

    def _dual_greedy_reduce(blocks, deadline):
        """
        Iteratively try to remove each block by finding a replacement
        block that covers all its uniquely-covered t-subsets.
        This is a 'dual' approach: instead of adding blocks greedily,
        we try to remove them and replace with better ones.
        """
        if not blocks:
            return blocks

        block_tsub_indices = [_get_block_tsubs(b) for b in blocks]
        cover_count = [0] * n_tsubs
        for tsubs in block_tsub_indices:
            for idx in tsubs:
                cover_count[idx] += 1

        # Build set of all existing blocks for quick lookup
        existing_blocks = set(tuple(sorted(b)) for b in blocks)

        improved = True
        max_rounds = 50
        round_num = 0

        while improved and time.time() < deadline - 8 and round_num < max_rounds:
            improved = False
            round_num += 1

            # Find the block with the fewest unique t-subsets (easiest to replace)
            best_remove_idx = -1
            best_unique_count = float('inf')

            for i in range(len(blocks)):
                unique_count = sum(1 for idx in block_tsub_indices[i] if cover_count[idx] == 1)
                if unique_count < best_unique_count:
                    best_unique_count = unique_count
                    best_remove_idx = i
                if unique_count == 0:
                    break  # Can remove for free

            if best_remove_idx == -1:
                break

            i = best_remove_idx
            unique_tsubs = [idx for idx in block_tsub_indices[i] if cover_count[idx] == 1]

            if len(unique_tsubs) == 0:
                # Remove for free
                for idx in block_tsub_indices[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_tsub_indices.pop(i)
                improved = True
                continue

            # Try to find a replacement block that covers all unique_tsubs
            # The replacement must be a k-subset that contains all points in each unique t-subset
            # and is not already in the block list

            # Find the intersection of all unique t-subsets - these points MUST be in the replacement
            must_include = set()
            for idx in unique_tsubs:
                must_include.update(all_tsubs[idx])

            # Find points that are in SOME but not ALL unique t-subsets
            # The replacement must include must_include, and for each unique t-subset,
            # it must include the remaining t-|must_include| points from that t-subset

            # Strategy: build a candidate block starting from must_include,
            # then greedily add points to cover remaining unique t-subsets
            candidate = list(must_include)
            candidate_set = set(candidate)

            # For each unique t-subset, determine which points still need to be added
            needed_points = {}  # tsub_idx -> list of points still needed
            for idx in unique_tsubs:
                ts = all_tsubs[idx]
                needed = [p for p in ts if p not in candidate_set]
                if needed:
                    needed_points[idx] = needed

            # Greedily add points to cover all needed points
            while needed_points and len(candidate) < k:
                if time.time() > deadline - 8:
                    break

                # For each candidate point, count how many needed points it covers
                best_point = -1
                best_coverage = -1

                for x in range(v):
                    if x in candidate_set:
                        continue
                    # Count how many needed t-subsets this point helps with
                    coverage = 0
                    for idx, needed in needed_points.items():
                        if x in needed:
                            coverage += 1
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_point = x

                if best_point == -1 or best_coverage == 0:
                    # Can't cover all unique t-subsets with a single new block
                    break

                candidate.append(best_point)
                candidate_set.add(best_point)
                # Update needed points
                for idx in list(needed_points.keys()):
                    needed_points[idx] = [p for p in needed_points[idx] if p != best_point]
                    if not needed_points[idx]:
                        del needed_points[idx]

            if needed_points:
                # Couldn't cover all unique t-subsets
                continue

            # Check if candidate is valid (all distinct, size k)
            if len(candidate) != k or len(set(candidate)) != k:
                continue

            candidate.sort()
            candidate_tuple = tuple(candidate)

            # Check if this block already exists
            if candidate_tuple in existing_blocks:
                continue

            # Verify the candidate covers all unique t-subsets
            candidate_tsubs = set(_get_block_tsubs(candidate))
            all_covered = True
            for idx in unique_tsubs:
                if idx not in candidate_tsubs:
                    all_covered = False
                    break

            if not all_covered:
                continue

            # Check that removing block i and adding candidate doesn't break coverage
            # Block i's non-unique t-subsets are still covered by other blocks
            # Candidate's t-subsets: some may be new, some may overlap
            # We need to ensure no t-subset becomes uncovered

            # Simulate: remove block i, add candidate
            new_cover_count = cover_count[:]
            for idx in block_tsub_indices[i]:
                new_cover_count[idx] -= 1
            for idx in candidate_tsubs:
                new_cover_count[idx] += 1

            # Check all t-subsets still covered
            valid = True
            for idx in range(n_tsubs):
                if new_cover_count[idx] <= 0:
                    valid = False
                    break

            if not valid:
                continue

            # Apply the swap
            for idx in block_tsub_indices[i]:
                cover_count[idx] -= 1
            for idx in candidate_tsubs:
                cover_count[idx] += 1

            blocks.pop(i)
            block_tsub_indices.pop(i)
            existing_blocks.discard(tuple(blocks[i]) if i < len(blocks) else tuple(sorted(blocks[i-1])))

            blocks.append(candidate)
            block_tsub_indices.append(list(candidate_tsubs))
            existing_blocks.add(candidate_tuple)

            improved = True

        return blocks

    def _local_search(blocks, deadline):
        """Try to swap points in blocks to enable removal of other blocks."""
        if not blocks:
            return blocks

        block_tsub_indices = [_get_block_tsubs(b) for b in blocks]
        cover_count = [0] * n_tsubs
        for tsubs in block_tsub_indices:
            for idx in tsubs:
                cover_count[idx] += 1

        improved = True
        max_iterations = 30
        iter_count = 0

        while improved and time.time() < deadline - 10 and iter_count < max_iterations:
            improved = False
            iter_count += 1

            for i in range(len(blocks)):
                if time.time() > deadline - 10:
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
                    if time.time() > deadline - 10:
                        break

                    crit_tsub = all_tsubs[crit_idx]
                    crit_set = set(crit_tsub)

                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if time.time() > deadline - 10:
                            break
                        if found_swap:
                            break

                        block_j = blocks[j]
                        block_j_set = set(block_j)
                        common = block_j_set & crit_set

                        if len(common) >= t - 1:
                            missing = [p for p in crit_tsub if p not in block_j_set]
                            extra = [p for p in block_j if p not in common]

                            if len(missing) == 1 and extra:
                                for rem_p in extra:
                                    if found_swap:
                                        break
                                    new_block_j = [p if p != rem_p else missing[0] for p in block_j]
                                    new_block_j_set = set(new_block_j)
                                    if len(new_block_j_set) != k:
                                        continue

                                    old_tsubs = set(block_tsub_indices[j])
                                    new_tsubs = set(_get_block_tsubs(new_block_j))

                                    can_swap = True
                                    for idx in old_tsubs:
                                        if idx not in new_tsubs and cover_count[idx] == 1:
                                            can_swap = False
                                            break

                                    if can_swap:
                                        for idx in old_tsubs:
                                            cover_count[idx] -= 1
                                        for idx in new_tsubs:
                                            cover_count[idx] += 1
                                        blocks[j] = new_block_j
                                        block_tsub_indices[j] = list(new_tsubs)
                                        improved = True
                                        found_swap = True
                                        break

                            if not found_swap and len(common) >= t - 2:
                                missing = [p for p in crit_tsub if p not in block_j_set]
                                if len(missing) == 2 and len(extra) >= 2:
                                    for rem1, rem2 in itertools.combinations(extra, 2):
                                        if found_swap:
                                            break
                                        new_block_j = [p for p in block_j if p not in (rem1, rem2)] + missing
                                        new_block_j_set = set(new_block_j)
                                        if len(new_block_j_set) != k:
                                            continue

                                        old_tsubs = set(block_tsub_indices[j])
                                        new_tsubs = set(_get_block_tsubs(new_block_j))

                                        can_swap = True
                                        for idx in old_tsubs:
                                            if idx not in new_tsubs and cover_count[idx] == 1:
                                                can_swap = False
                                                break

                                        if can_swap:
                                            for idx in old_tsubs:
                                                cover_count[idx] -= 1
                                            for idx in new_tsubs:
                                                cover_count[idx] += 1
                                            blocks[j] = new_block_j
                                            block_tsub_indices[j] = list(new_tsubs)
                                            improved = True
                                            found_swap = True
                                            break

                if found_swap:
                    break

        return blocks

    # Precompute point coverage for tie-breaking
    point_tsub_lists = [[] for _ in range(v)]
    for ts in all_tsubs:
        for x in ts:
            idx = tsub_to_idx[ts]
            point_tsub_lists[x].append(idx)

    # Main loop: Randomized Greedy with dual reduction
    for restart in range(300):
        if time.time() > deadline - 25:
            break

        uncovered = set(range(n_tsubs))
        blocks = []
        order = list(range(n_tsubs))
        rng.shuffle(order)

        while uncovered:
            if time.time() > deadline - 15:
                break

            seed_idx = next(idx for idx in order if idx in uncovered)
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

        if time.time() > deadline - 15 or uncovered:
            continue

        # Post-processing: redundancy removal, dual greedy, local search
        blocks = _remove_redundant(blocks, deadline)

        if time.time() < deadline - 20:
            blocks = _dual_greedy_reduce(blocks, deadline)
            blocks = _remove_redundant(blocks, deadline)

        if time.time() < deadline - 15:
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

    best_blocks = _remove_redundant(best_blocks, deadline)

    return best_blocks
