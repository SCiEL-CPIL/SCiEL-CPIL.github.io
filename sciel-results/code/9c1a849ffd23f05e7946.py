"""Covering design C(19,7,4) – dynamic greedy with local search."""
import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    # Map each t-subset to its index
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    # Build reverse index: for each point, list of t-subsets containing it
    point_to_t_subs = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subs):
        for pt in s:
            point_to_t_subs[pt].append(i)

    # For a given set of points, compute which t-subsets are covered
    def covered_by(points):
        """Return a set of t-subset indices covered by the given points."""
        result = set()
        for comb in itertools.combinations(points, t):
            result.add(t_sub_to_idx[comb])
        return result

    def greedy_cover(points, t_remaining, deadline):
        """Build a block containing points, then greedily add best points."""
        current = set(points)
        block_covered = covered_by(current)
        remaining_points = [p for p in range(v) if p not in current]

        # Compute current uncovered
        uncovered_set = set(range(n_t)) - block_covered

        while len(current) < k:
            if time.time() > deadline:
                break
            best_pt = None
            best_gain = -1
            # For each candidate point, count how many new t-subsets it covers
            for pt in remaining_points:
                # A t-subset is newly covered if it contains pt and exactly t-1 other points in current
                gain = 0
                for rest in itertools.combinations(current, t - 1):
                    sub = tuple(sorted(rest + (pt,)))
                    idx = t_sub_to_idx.get(sub)
                    if idx is not None and idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_pt = pt
            if best_pt is None:
                break
            current.add(best_pt)
            # Update block_covered and uncovered_set
            new_covered = covered_by(current)
            new_new = new_covered - block_covered
            uncovered_set -= new_new
            block_covered = new_covered
            remaining_points.remove(best_pt)

        return sorted(current), block_covered

    # Initial greedy pass
    uncovered = set(range(n_t))
    blocks = []

    while uncovered and time.time() < deadline:
        # Pick the first uncovered t-subset
        first_sub = min(uncovered)
        first_points = list(all_t_subs[first_sub])

        block, covered = greedy_cover(first_points, t, deadline)
        if not block:
            break
        blocks.append(block)
        uncovered -= covered

    if not uncovered:
        return blocks

    # Local search: try to remove redundant blocks
    def try_remove_block(blocks, uncovered_set):
        """Try to remove each block if it's redundant."""
        n = len(blocks)
        removed = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Check if all t-subsets covered by block i are also covered by other blocks
            block_t_subs = covered_by(blocks[i])
            redundant = True
            for ts_idx in block_t_subs:
                if ts_idx in uncovered_set:
                    # This t-subset is only covered by this block (not in uncovered means already covered)
                    # Wait: uncovered_set tracks what's NOT yet covered. If ts_idx is in uncovered_set, it means it's uncovered, which shouldn't happen if we have a valid covering.
                    # Let me re-think: uncovered_set should be empty for a valid covering.
                    redundant = False
                    break
            # Actually, we need to check if removing block i leaves any t-subset uncovered
            # For each t-subset that block i covers, check if some other block also covers it
            for ts_idx in block_t_subs:
                ts = all_t_subs[ts_idx]
                other_covers = False
                for j, other_block in enumerate(blocks):
                    if j == i:
                        continue
                    if all(p in other_block for p in ts):
                        other_covers = True
                        break
                if not other_covers:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                removed = True
                break
        return removed

    # Try to remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = try_remove_block(blocks, set())

    # Multi-start with randomization
    best_blocks = blocks
    best_len = len(blocks)

    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        random.shuffle(all_t_subs)
        uncovered = set(range(n_t))
        blocks_r = []

        while uncovered and time.time() < deadline:
            # Pick a random uncovered t-subset
            first_sub = random.choice(list(uncovered))
            first_points = list(all_t_subs[first_sub])

            # Greedy with some randomization
            current = set(first_points)
            block_covered = covered_by(current)
            remaining_points = [p for p in range(v) if p not in current]

            while len(current) < k:
                if time.time() > deadline:
                    break
                candidates = remaining_points[:]
                random.shuffle(candidates)

                best_pt = None
                best_gain = -1
                for pt in candidates:
                    gain = 0
                    for rest in itertools.combinations(current, t - 1):
                        sub = tuple(sorted(rest + (pt,)))
                        idx = t_sub_to_idx.get(sub)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = pt
                if best_pt is None:
                    break
                current.add(best_pt)
                new_covered = covered_by(current)
                new_new = new_covered - block_covered
                uncovered -= new_new
                block_covered = new_covered
                remaining_points.remove(best_pt)

            block = sorted(current)
            blocks_r.append(block)

        # Local search on this restart
        changed = True
        while changed and time.time() < deadline:
            changed = try_remove_block(blocks_r, set())

        if len(blocks_r) < best_len:
            best_len = len(blocks_r)
            best_blocks = blocks_r

    return best_blocks
