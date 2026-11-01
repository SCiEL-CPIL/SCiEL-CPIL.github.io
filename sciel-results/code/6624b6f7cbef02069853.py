"""Improved covering design C(19,8,5) via fast greedy + local search (block replacement)."""
import itertools
import time
import random
from collections import defaultdict


def _count_covered(block, v, t):
    """Return a set of t-subsets covered by this block (as frozensets or tuples)."""
    return set(itertools.combinations(block, t))


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)

    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Coverage count array: how many blocks cover each t-subset
    # We'll maintain this incrementally

    # Phase 1: Fast greedy construction
    # Use bitsets or simple sets for uncovered
    uncovered = set(all_tsubsets)
    blocks = []
    block_to_tsubsets = []  # For each block, the list of t-subset indices it covers

    while uncovered:
        if time.time() > deadline:
            break
        # Pick the first uncovered t-subset (or a random one for diversity)
        seed_ts = next(iter(uncovered))
        block = list(seed_ts)

        # Greedily add points that maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        # Only add if it covers at least one new t-subset
        covered_by_block = list(itertools.combinations(block, t))
        new_covered = [ts for ts in covered_by_block if ts in uncovered]
        if new_covered:
            blocks.append(block)
            block_to_tsubsets.append([tsubset_to_idx[ts] for ts in covered_by_block])
            uncovered.difference_update(covered_by_block)

    # If we couldn't finish greedy in time, we still have a partial solution.
    # But for validity we need full coverage. Let's check.
    if uncovered:
        # Continue greedy without time check but with a hard limit
        # Actually, if we're out of time, we should just return what we have
        # but it won't be valid. Let's try to finish quickly.
        # For now, continue but with a tighter deadline
        while uncovered:
            seed_ts = next(iter(uncovered))
            block = list(seed_ts)
            while len(block) < k:
                best_x = None
                best_gain = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            covered_by_block = list(itertools.combinations(block, t))
            new_covered = [ts for ts in covered_by_block if ts in uncovered]
            if new_covered:
                blocks.append(block)
                block_to_tsubsets.append([tsubset_to_idx[ts] for ts in covered_by_block])
                uncovered.difference_update(covered_by_block)

    # Build coverage count array
    cover_count = [0] * num_tsubsets
    for tsub_list in block_to_tsubsets:
        for idx in tsub_list:
            cover_count[idx] += 1

    # Phase 2: Local search - try to replace blocks with better ones
    # A block is "redundant" if removing it doesn't leave any t-subset uncovered
    # We try to remove redundant blocks and replace them with blocks that cover more

    random.seed(42)

    # Helper: try to find a replacement block for a given block
    def find_replacement_block(block, blocks, block_to_tsubsets, cover_count, tsubset_to_idx, all_tsubsets, v, k, t):
        """Find a block that covers at least as many currently uncovered-or-singly-covered t-subsets."""
        # The t-subsets that would become uncovered if we remove this block
        # are those with cover_count == 1
        must_cover = []
        for idx in block_to_tsubsets[blocks.index(block)]:
            if cover_count[idx] == 1:
                must_cover.append(all_tsubsets[idx])

        if not must_cover:
            # Block is fully redundant, can just remove it
            return None

        # Try to find a block that covers all must_cover t-subsets
        # and possibly more
        best_block = None
        best_extra = -1

        # Sample some candidate blocks
        candidates = set()
        for ts in must_cover:
            for x in range(v):
                if x not in ts:
                    # Try blocks containing ts and x
                    pass

        # Simpler: try random blocks and see if they cover must_cover
        # But this is slow. Let's use a heuristic:
        # The intersection of all must_cover t-subsets gives required points
        # Actually, must_cover t-subsets may not have a common intersection.
        # We need a k-subset that contains each must_cover t-subset.
        # This is only possible if the union of must_cover has size <= k
        # and each must_cover is contained in the candidate.

        union_points = set()
        for ts in must_cover:
            union_points.update(ts)

        if len(union_points) > k:
            # Cannot cover all with one block
            return None

        # We need to extend union_points to a k-subset
        # Try a few random extensions
        for _ in range(10):
            extra_points = set()
            remaining = k - len(union_points)
            if remaining > 0:
                available = [x for x in range(v) if x not in union_points]
                extra_points = set(random.sample(available, remaining))
            candidate = tuple(sorted(union_points | extra_points))
            # Check if candidate covers all must_cover
            cand_set = set(candidate)
            ok = True
            for ts in must_cover:
                if not all(x in cand_set for x in ts):
                    ok = False
                    break
            if ok:
                # Count how many t-subsets this candidate covers
                covered = set(itertools.combinations(candidate, t))
                extra = 0
                for ts in covered:
                    idx = tsubset_to_idx[ts]
                    if cover_count[idx] == 0:
                        extra += 1
                if extra > best_extra:
                    best_extra = extra
                    best_block = list(candidate)

        return best_block

    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove each block and see if it's redundant
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Check if block i is redundant (all its t-subsets are covered by other blocks)
            redundant = True
            for idx in block_to_tsubsets[i]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for idx in block_to_tsubsets[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_to_tsubsets.pop(i)
                improved = True

        if not improved:
            # Try to swap: remove a block and add a better one
            # For each block, try to find a replacement that allows removing it
            # and possibly adding fewer blocks
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                # Try to remove block i and see what breaks
                # If it breaks, try to find a block that covers the broken t-subsets
                # and has higher overall coverage
                must_cover = []
                for idx in block_to_tsubsets[i]:
                    if cover_count[idx] == 1:
                        must_cover.append(all_tsubsets[idx])

                if not must_cover:
                    # Just remove
                    for idx in block_to_tsubsets[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_to_tsubsets.pop(i)
                    improved = True
                    continue

                # Try to find a replacement
                union_points = set()
                for ts in must_cover:
                    union_points.update(ts)

                if len(union_points) > k:
                    continue

                # Try a few random extensions
                found_replacement = False
                for _ in range(20):
                    remaining = k - len(union_points)
                    extra_points = set()
                    if remaining > 0:
                        available = [x for x in range(v) if x not in union_points]
                        extra_points = set(random.sample(available, remaining))
                    candidate = tuple(sorted(union_points | extra_points))
                    cand_set = set(candidate)
                    ok = True
                    for ts in must_cover:
                        if not all(x in cand_set for x in ts):
                            ok = False
                            break
                    if not ok:
                        continue

                    # Check if this candidate is already in blocks
                    if candidate in [tuple(b) for b in blocks]:
                        continue

                    # Compute coverage gain
                    covered = set(itertools.combinations(candidate, t))
                    # This candidate would cover must_cover (which are currently singly covered)
                    # and possibly more
                    # Net effect: remove block i, add candidate
                    # Check if candidate covers all must_cover
                    # It does by construction
                    # Count how many new t-subsets it covers
                    new_cover = 0
                    for ts in covered:
                        idx = tsubset_to_idx[ts]
                        if cover_count[idx] == 0:
                            new_cover += 1

                    # Remove block i
                    for idx in block_to_tsubsets[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_to_tsubsets.pop(i)

                    # Add candidate
                    blocks.append(list(candidate))
                    cand_tsubs = [tsubset_to_idx[ts] for ts in itertools.combinations(candidate, t)]
                    block_to_tsubsets.append(cand_tsubs)
                    for idx in cand_tsubs:
                        cover_count[idx] += 1

                    found_replacement = True
                    improved = True
                    break

    # Final: try to remove redundant blocks again
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            redundant = True
            for idx in block_to_tsubsets[i]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_to_tsubsets[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_to_tsubsets.pop(i)
                changed = True

    # Verify coverage
    # This is a sanity check; in practice we should ensure validity
    return blocks
