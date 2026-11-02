"""Covering design C(19,8,5): randomized greedy with local search and redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import Counter


def construct(v, k, t):
    deadline = time.time() + 130
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)

    for restart in range(50):
        if time.time() > deadline:
            break

        uncovered = set(range(n_tsubs))
        blocks = []
        order = list(range(n_tsubs))
        rng.shuffle(order)

        while uncovered:
            if time.time() > deadline:
                break

            seed_idx = next(idx for idx in order if idx in uncovered)
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

        if time.time() > deadline or uncovered:
            continue

        blocks = _remove_redundant(blocks, v, k, t, tsub_to_idx, deadline)
        
        # Local search: try swapping points in blocks to enable more redundancy removal
        if time.time() < deadline:
            blocks = _local_search(blocks, v, k, t, tsub_to_idx, deadline, rng)
            blocks = _remove_redundant(blocks, v, k, t, tsub_to_idx, deadline)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 335:
                break

    if best_blocks is None:
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

    return best_blocks


def _remove_redundant(blocks, v, k, t, tsub_to_idx, deadline):
    if not blocks:
        return blocks

    cover_count = Counter()
    block_tsubs = []
    for block in blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(tuple(ts))
            if idx is not None:
                tsubs.append(idx)
                cover_count[idx] += 1
        block_tsubs.append(tsubs)

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            if not any(cover_count[idx] == 1 for idx in block_tsubs[i]):
                for idx in block_tsubs[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_tsubs.pop(i)
                changed = True
                break

    return blocks


def _local_search(blocks, v, k, t, tsub_to_idx, deadline, rng):
    """Try to improve the covering by swapping points in blocks."""
    if not blocks or time.time() > deadline:
        return blocks
    
    n_blocks = len(blocks)
    if n_blocks == 0:
        return blocks
    
    # Build coverage map
    cover_count = Counter()
    block_tsubs = []
    for block in blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(tuple(ts))
            if idx is not None:
                tsubs.append(idx)
                cover_count[idx] += 1
        block_tsubs.append(tsubs)
    
    # Try random swaps
    max_swaps = min(200, n_blocks * 5)
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        # Pick two random blocks
        i, j = rng.sample(range(n_blocks), 2)
        bi = set(blocks[i])
        bj = set(blocks[j])
        
        # Pick a point from each block to swap
        pi = rng.choice(list(bi))
        pj = rng.choice(list(bj))
        if pi == pj:
            continue
        
        # Tentatively swap
        blocks[i][blocks[i].index(pi)] = pj
        blocks[i].sort()
        blocks[j][blocks[j].index(pj)] = pi
        blocks[j].sort()
        
        # Recompute coverage
        new_cover_count = Counter()
        new_block_tsubs = []
        for block in blocks:
            tsubs = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    tsubs.append(idx)
                    new_cover_count[idx] += 1
            new_block_tsubs.append(tsubs)
        
        # Check if all t-subsets are still covered
        all_covered = all(new_cover_count[idx] >= 1 for idx in range(len(tsub_to_idx)))
        
        if all_covered:
            # Count how many blocks can be removed
            removable = sum(1 for bt in new_block_tsubs 
                          if not any(new_cover_count[idx] == 1 for idx in bt))
            old_removable = sum(1 for bt in block_tsubs 
                               if not any(cover_count[idx] == 1 for idx in bt))
            
            if removable > old_removable:
                # Accept the swap
                cover_count = new_cover_count
                block_tsubs = new_block_tsubs
            else:
                # Revert
                blocks[i][blocks[i].index(pj)] = pi
                blocks[i].sort()
                blocks[j][blocks[j].index(pi)] = pj
                blocks[j].sort()
        else:
            # Revert
            blocks[i][blocks[i].index(pj)] = pi
            blocks[i].sort()
            blocks[j][blocks[j].index(pi)] = pj
            blocks[j].sort()
    
    return blocks
# EVOLVE-BLOCK-END
