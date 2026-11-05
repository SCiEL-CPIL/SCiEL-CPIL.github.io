# EVOLVE-BLOCK-START
import time
import itertools
import random
import math
import sys

def construct(v, k, t):
    deadline = time.time() + 115.0
    rng = random.Random(12345)

    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_t = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute, for each point x, the list of t-subset indices containing x
    # tsubs_by_point[x] = list of indices i where x in all_tsubs[i]
    tsubs_by_point = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            tsubs_by_point[x].append(i)

    # Precompute C(k-1, t-1) as the number of t-subsets a single point contributes
    # For a block B, the t-subsets covered are all C(B, t).
    # Number of t-subsets covered by a block = C(k, t)
    # When we add point x to a partial block P (|P| = m), new t-subsets = C(m, t-1) * 1
    #   (the t-subsets that include x and t-1 points from P)

    def greedy_one(seed):
        rng2 = random.Random(seed)
        covered = bytearray(n_t)  # 0/1 coverage
        blocks = []
        # To speed up, track uncovered count
        uncovered_count = n_t

        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                break
            # Find an uncovered t-subset to start a block
            # Pick a random-ish uncovered t-subset: scan
            # For speed, just find first uncovered
            start = None
            for i in range(n_t):
                if not covered[i]:
                    start = i
                    break
            if start is None:
                break
            block = list(all_tsubs[start])
            # Mark covered
            for ts in itertools.combinations(block, t):
                idx = tsub_index[ts]
                if not covered[idx]:
                    covered[idx] = 1
                    uncovered_count -= 1
            # Grow the block to size k
            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break
                best_x = -1
                best_gain = -1
                # For each candidate x not in block, compute gain = number of uncovered t-subsets
                # that would be covered by adding x
                # = number of uncovered t-subsets containing x and t-1 points from block
                # Precompute: for the current block P, the set of (t-1)-subsets of P
                # For each x, gain = sum over (t-1)-subsets S of P of [tsub_index[S + (x,)] is uncovered]
                # Efficiently: iterate over t-1 subsets of P, and for each, check all x
                # But v is small (19), so just compute per x
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_index[ts]
                        if not covered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    # pick random
                    best_x = rng2.choice([x for x in range(v) if x not in block_set])
                block.append(best_x)
            block.sort()
            # Mark all t-subsets of this block as covered
            for ts in itertools.combinations(block, t):
                idx = tsub_index[ts]
                if not covered[idx]:
                    covered[idx] = 1
                    uncovered_count -= 1
            blocks.append(tuple(block))

        return blocks, uncovered_count

    def local_search(blocks, max_iters=200):
        """Local search: try to replace a block with a better one, or remove redundant blocks."""
        blocks = [list(b) for b in blocks]
        n = len(blocks)
        # Coverage count: how many blocks cover each t-subset
        cov_count = [0] * n_t
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov_count[tsub_index[ts]] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 3.0:
                break
            improved = False
            iters += 1
            # Try to remove a redundant block
            for i in range(n):
                b = blocks[i]
                # Check if any t-subset is only covered by this block
                redundant = True
                for ts in itertools.combinations(b, t):
                    idx = tsub_index[ts]
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for ts in itertools.combinations(b, t):
                        cov_count[tsub_index[ts]] -= 1
                    blocks.pop(i)
                    n -= 1
                    improved = True
                    break
            if improved:
                continue
            # Try to replace a block with a better one (covers more uncovered or enables removal)
            # For each block, try to find a replacement that covers all t-subsets the old block covers
            # and also covers some additional ones, or at least maintains coverage.
            # Simpler: try swapping one point in a block to reduce redundancy or cover more.
            for i in range(n):
                if time.time() > deadline - 3.0:
                    break
                b = blocks[i]
                # For each point in b, try replacing with a point not in b
                for pi, old_x in enumerate(b):
                    best_new = -1
                    best_score = 0
                    for new_x in range(v):
                        if new_x in b:
                            continue
                        # Compute the effect of swapping old_x -> new_x
                        # t-subsets lost: those containing old_x but not new_x (from b)
                        # t-subsets gained: those containing new_x but not old_x (from b)
                        # Net effect on coverage
                        # For simplicity, compute:
                        # lost = number of t-subsets in C(b, t) that contain old_x but not new_x
                        # gained = number of t-subsets in C(b with new_x, t) that contain new_x but not old_x
                        # But we need to check if lost t-subsets are still covered by other blocks
                        # This is expensive. Let's do a simpler heuristic:
                        # Try the swap and check if coverage is maintained.
                        b2 = b[:]
                        b2[pi] = new_x
                        b2.sort()
                        # Check: are all t-subsets of b still covered by b2 or other blocks?
                        # For each t-subset of b, check if it's in b2 or covered by another block
                        ok = True
                        for ts in itertools.combinations(b, t):
                            idx = tsub_index[ts]
                            if idx not in [tsub_index[ts2] for ts2 in itertools.combinations(b2, t)]:
                                # Not in b2, check if covered by another block
                                if cov_count[idx] <= 1:
                                    ok = False
                                    break
                        if ok:
                            # Count how many new t-subsets b2 covers that b didn't
                            b_set = set(b)
                            b2_set = set(b2)
                            new_gain = 0
                            for ts2 in itertools.combinations(b2, t):
                                if ts2 not in [ts for ts in itertools.combinations(b, t)]:
                                    idx = tsub_index[ts2]
                                    if cov_count[idx] == 0:
                                        new_gain += 1
                            if new_gain > best_score:
                                best_score = new_gain
                                best_new = new_x
                    if best_new != -1 and best_score > 0:
                        # Perform the swap
                        b[pi] = best_new
                        b.sort()
                        # Update cov_count
                        for ts in itertools.combinations(b, t):
                            cov_count[tsub_index[ts]] -= 1
                        for ts in itertools.combinations(b, t):
                            cov_count[tsub_index[ts]] += 1
                        # Wait, that's wrong. Let me redo.
                        # Actually, let me just recompute cov_count from scratch for simplicity
                        # since v=19, n is at most ~151, and C(7,4)=35 per block
                        # That's 151*35 = 5285 operations, fast enough
                        cov_count = [0] * n_t
                        for bb in blocks:
                            for ts in itertools.combinations(bb, t):
                                cov_count[tsub_index[ts]] += 1
                        improved = True
                        break
                if improved:
                    break
        return blocks

    # Generate multiple greedy solutions with different seeds
    best_blocks = None
    best_count = float('inf')

    # First, a basic greedy
    blocks, unc = greedy_one(0)
    if unc == 0:
        best_blocks = blocks
        best_count = len(blocks)

    # Try multiple seeds
    for seed in range(1, 50):
        if time.time() > deadline - 30.0:
            break
        blocks, unc = greedy_one(seed)
        if unc == 0 and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)

    # Local search on the best
    if best_blocks is not None:
        if time.time() < deadline - 10.0:
            best_blocks = local_search(best_blocks, max_iters=300)
            best_count = len(best_blocks)

    # If we still have time, try more local search or restarts
    if best_blocks is not None and time.time() < deadline - 5.0:
        # Try local search again with more iterations
        best_blocks = local_search(best_blocks, max_iters=500)
        best_count = len(best_blocks)

    # Ensure we have a valid solution
    if best_blocks is None:
        # Fallback: greedy without time limit (shouldn't happen)
        blocks, unc = greedy_one(0)
        best_blocks = blocks

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
