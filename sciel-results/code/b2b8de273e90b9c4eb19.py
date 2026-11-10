"""Covering design C(v,k,t) – greedy + efficient local improvement with deadline guard."""
import itertools
import time
import random

_DEADLINE_S = 120.0
_MARGIN_S = 10.0


def construct(v, k, t):
    start_time = time.time()

    def remaining():
        return max(0.0, _DEADLINE_S - _MARGIN_S - (time.time() - start_time))

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []

    def block_covers(block):
        return set(itertools.combinations(block, t))

    def gain_for_point(block, x):
        cnt = 0
        for rest in itertools.combinations(block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            if ts in uncovered:
                cnt += 1
        return cnt

    # --- Greedy construction ---
    while uncovered and remaining() > 1.0:
        target = min(uncovered)
        block = list(target)
        block.sort()
        while len(block) < k and remaining() > 0.5:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                g = gain_for_point(block, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None or best_gain == 0:
                # Pick arbitrary to fill
                best_x = candidates[0]
            block.append(best_x)
        block.sort()
        block = list(set(block))
        if len(block) < k:
            used = set(block)
            rem = [x for x in range(v) if x not in used]
            rem.sort()
            block.extend(rem[:k - len(block)])
            block.sort()
        if len(set(block)) != k:
            continue
        covered = block_covers(block)
        if not covered:
            continue
        uncovered -= covered
        blocks.append(block)

    # --- Efficient local improvement using coverage counts ---
    if remaining() > 2.0:
        # Build coverage count: for each t-subset, how many blocks cover it
        cov_count = {}
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov_count[ts] = cov_count.get(ts, 0) + 1

        # Remove redundant blocks (those that don't uniquely cover any t-subset)
        changed = True
        while changed and remaining() > 2.0:
            changed = False
            new_blocks = []
            for i in range(len(blocks)):
                b = blocks[i]
                # Check if this block is the only one covering any t-subset
                unique = False
                for ts in itertools.combinations(b, t):
                    if cov_count.get(ts, 0) == 1:
                        unique = True
                        break
                if not unique and len(blocks) > 1:
                    # Remove this block
                    for ts in itertools.combinations(b, t):
                        cov_count[ts] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            if changed:
                blocks = new_blocks

    # --- Try to merge pairs of blocks into one ---
    if remaining() > 3.0:
        n = len(blocks)
        i = 0
        while i < n and remaining() > 2.0:
            b_i = blocks[i]
            cov_i = set(itertools.combinations(b_i, t))
            j = i + 1
            removed = False
            while j < n:
                b_j = blocks[j]
                cov_j = set(itertools.combinations(b_j, t))
                union = cov_i | cov_j
                # Collect all points appearing in any t-subset in union
                pts = set()
                for ts in union:
                    pts.update(ts)
                if len(pts) <= k:
                    new_block = sorted(pts)
                    if len(new_block) < k:
                        extra = [x for x in range(v) if x not in set(new_block)]
                        extra.sort()
                        new_block.extend(extra[:k - len(new_block)])
                        new_block.sort()
                    # Verify coverage
                    if set(itertools.combinations(new_block, t)) >= union:
                        # Replace blocks i and j with new_block
                        blocks[i] = new_block
                        del blocks[j]
                        removed = True
                        n = len(blocks)
                        break
                j += 1
            if not removed:
                i += 1

    # --- Swap-based local search: try replacing a block with a better one ---
    if remaining() > 5.0:
        # Build coverage count
        cov_count = {}
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov_count[ts] = cov_count.get(ts, 0) + 1

        # Try to replace blocks that have low unique coverage
        for i in range(len(blocks)):
            if remaining() < 3.0:
                break
            b = blocks[i]
            # Count unique t-subsets covered by this block
            unique_count = 0
            for ts in itertools.combinations(b, t):
                if cov_count.get(ts, 0) == 1:
                    unique_count += 1

            # If this block has few unique subsets, try to replace it
            if unique_count < 10 and len(blocks) > 1:
                # Try to find a replacement block that covers more unique subsets
                best_replacement = None
                best_gain = -1
                # Sample some candidate blocks
                candidates = random.sample(range(v), min(v, 20))
                for _ in range(5):
                    # Create a random block
                    rand_block = sorted(random.sample(range(v), k))
                    # Check if it covers any uncovered subsets
                    new_covers = set(itertools.combinations(rand_block, t))
                    # Count how many of these are currently uniquely covered by block i
                    gain = 0
                    for ts in new_covers:
                        if cov_count.get(ts, 0) == 1:
                            # This ts is uniquely covered by block i, so replacing would lose it
                            # But if we add this ts to coverage, we gain
                            pass
                    # Alternative: check if rand_block covers subsets that no other block covers
                    # Remove block i temporarily
                    temp_cov_count = dict(cov_count)
                    for ts in itertools.combinations(b, t):
                        temp_cov_count[ts] = temp_cov_count.get(ts, 0) - 1
                    # Now check if rand_block adds any new coverage
                    new_unique = 0
                    for ts in itertools.combinations(rand_block, t):
                        if temp_cov_count.get(ts, 0) == 0:
                            new_unique += 1
                    if new_unique > best_gain:
                        best_gain = new_unique
                        best_replacement = rand_block

                if best_replacement is not None and best_gain > unique_count:
                    # Replace block i with best_replacement
                    for ts in itertools.combinations(b, t):
                        cov_count[ts] = cov_count.get(ts, 0) - 1
                    for ts in itertools.combinations(best_replacement, t):
                        cov_count[ts] = cov_count.get(ts, 0) + 1
                    blocks[i] = best_replacement

    return blocks
