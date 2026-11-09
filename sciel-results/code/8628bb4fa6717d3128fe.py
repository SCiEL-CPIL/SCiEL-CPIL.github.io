import time
import random

# EVOLVE-BLOCK-START
N = 12
M = 40
N_MASKS = 1 << N
DEADLINE = 70.0  # safety margin before the 90s wall-clock kill


def _score(mask_list):
    """Count unordered pairs where one mask is a strict subset of the other."""
    cnt = 0
    n = len(mask_list)
    for i in range(n):
        a = mask_list[i]
        for j in range(i + 1, n):
            b = mask_list[j]
            if (a & b) == a and a != b:
                cnt += 1
            elif (a & b) == b and a != b:
                cnt += 1
    return cnt


def _masks_to_lists(mask_list):
    out = []
    for mask in mask_list:
        out.append([b for b in range(N) if (mask >> b) & 1])
    return out


def _make_initial():
    """Build a family of nested chains + singleton sets."""
    masks = []
    # 3 chains of 8 nested subsets each (24 sets)
    for base in [0x0FF, 0xF00, 0x0FF]:
        # 0x0FF = {0..7}, 0xF00 = {8..11}
        # For chain 3 use a different base to add variety
        pass
    # Chain 1: subsets of {0..7}
    base1 = 0x0FF
    cur = base1
    for _ in range(8):
        masks.append(cur)
        # remove one element
        # find a set bit to remove
        b = 0
        while not ((cur >> b) & 1):
            b += 1
        cur &= ~(1 << b)
    # Chain 2: subsets of {8..11}
    base2 = 0xF00
    cur = base2
    for _ in range(8):
        masks.append(cur)
        b = 8
        while not ((cur >> b) & 1):
            b += 1
        cur &= ~(1 << b)
    # Chain 3: subsets of {0,1,2,3,8,9,10,11}
    base3 = 0x00F | 0xF00
    cur = base3
    for _ in range(8):
        masks.append(cur)
        b = 0
        while not ((cur >> b) & 1):
            b += 1
        cur &= ~(1 << b)
    # 16 singleton sets
    for b in range(N):
        masks.append(1 << b)
    # Need exactly 40; we have 24 + 16 = 40
    assert len(masks) == M
    return masks


def _local_search(masks):
    """Greedy 2-opt: for each position, try all 4096 masks, keep best improvement."""
    cur_score = _score(masks)
    improved = True
    t0 = time.time()
    while improved:
        if time.time() - t0 > DEADLINE:
            break
        improved = False
        for pos in range(M):
            best_mask = masks[pos]
            best_gain = 0
            for cand in range(N_MASKS):
                if cand == masks[pos]:
                    continue
                new_masks = list(masks)
                new_masks[pos] = cand
                s = _score(new_masks)
                gain = s - cur_score
                if gain > best_gain:
                    best_gain = gain
                    best_mask = cand
            if best_gain > 0:
                masks[pos] = best_mask
                cur_score += best_gain
                improved = True
    return masks, cur_score


def construct():
    t_start = time.time()
    masks = _make_initial()
    masks, score = _local_search(masks)
    # Convert to lists of ints
    result = _masks_to_lists(masks)
    # Ensure exactly 40 lists, each a list of distinct ints
    assert len(result) == M
    return result
# EVOLVE-BLOCK-END
