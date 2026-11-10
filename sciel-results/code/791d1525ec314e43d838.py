"""Covering design C(v,k,t) – greedy + local improvement with deadline guard."""
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

    # --- Local improvement: remove redundant blocks ---
    if remaining() > 2.0:
        # Precompute coverage for each block
        block_covs = [block_covers(b) for b in blocks]

        changed = True
        while changed and remaining() > 2.0:
            changed = False
            new_blocks = []
            new_covs = []
            for i in range(len(blocks)):
                # Check if block i covers any t-subset not covered by other blocks
                unique = False
                cov_i = block_covs[i]
                for ts in cov_i:
                    only_this = True
                    for j in range(len(blocks)):
                        if i == j:
                            continue
                        if ts in block_covs[j]:
                            only_this = False
                            break
                    if only_this:
                        unique = True
                        break
                if not unique and len(blocks) > 1:
                    changed = True
                else:
                    new_blocks.append(blocks[i])
                    new_covs.append(block_covs[i])
            if changed:
                blocks = new_blocks
                block_covs = new_covs

    # --- Try to merge pairs of blocks into one ---
    if remaining() > 3.0:
        n = len(blocks)
        i = 0
        while i < n and remaining() > 2.0:
            b_i = blocks[i]
            cov_i = block_covs[i]
            j = i + 1
            removed = False
            while j < n:
                b_j = blocks[j]
                cov_j = block_covs[j]
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
                    if block_covers(new_block) >= union:
                        # Replace blocks i and j with new_block
                        blocks[i] = new_block
                        block_covs[i] = block_covers(new_block)
                        del blocks[j]
                        del block_covs[j]
                        removed = True
                        n = len(blocks)
                        break
                j += 1
            if not removed:
                i += 1

    return blocks
