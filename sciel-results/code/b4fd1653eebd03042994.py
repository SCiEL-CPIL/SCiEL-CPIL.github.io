# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(v, k, t):
    deadline = time.time() + 115.0

    tsubs = list(itertools.combinations(range(v), t))
    n_tsub = len(tsubs)
    tsub_to_bit = {sub: i for i, sub in enumerate(tsubs)}

    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    ksub_to_idx = {c: i for i, c in enumerate(all_k_subsets)}

    block_masks = []
    for comb in all_k_subsets:
        mask = 0
        for sub in itertools.combinations(comb, t):
            mask |= (1 << tsub_to_bit[sub])
        block_masks.append(mask)

    full_mask = (1 << n_tsub) - 1
    uncovered = full_mask

    cover_count = [0] * n_tsub
    blocks_selected = []
    selected_set = set()

    while uncovered and time.time() < deadline - 15.0:
        best_gain = 0
        best_idx = -1
        uncv = uncovered
        b_masks = block_masks
        for i in range(n_ksub):
            g = (b_masks[i] & uncv).bit_count()
            if g > best_gain:
                best_gain = g
                best_idx = i

        if best_gain == 0:
            break

        blocks_selected.append(best_idx)
        selected_set.add(best_idx)

        mask = block_masks[best_idx]
        m = mask
        while m:
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            cover_count[bit_pos] += 1
            m ^= lsb

        uncovered &= ~mask

    if uncovered and time.time() < deadline - 10.0:
        m = uncovered
        while m:
            if time.time() > deadline - 3.0:
                break
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            m ^= lsb
            tsub = tsubs[bit_pos]
            have = set(tsub)
            need = k - len(have)
            remaining_points = [p for p in range(v) if p not in have]
            if need > 0:
                extra = random.sample(remaining_points, need)
            else:
                extra = []
            block = sorted(list(have) + extra)
            block_tuple = tuple(block)
            if block_tuple in selected_set:
                continue
            b_idx = ksub_to_idx.get(block_tuple)
            if b_idx is not None and b_idx not in selected_set:
                blocks_selected.append(b_idx)
                selected_set.add(b_idx)
                mask = block_masks[b_idx]
                m2 = mask
                while m2:
                    lsb2 = m2 & (-m2)
                    bit_pos2 = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    cover_count[bit_pos2] += 1
                uncovered &= ~mask

    # Local search: swap-based reduction
    def try_swap_remove(blocks_selected, selected_set, cover_count, block_masks, deadline):
        removed_any = False
        for i in range(len(blocks_selected) - 1, -1, -1):
            if time.time() > deadline - 2.0:
                return removed_any
            b_idx = blocks_selected[i]
            mask = block_masks[b_idx]
            m = mask
            unique_tsub = -1
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                m ^= lsb
                if cover_count[bit_pos] == 1:
                    unique_tsub = bit_pos
                    break
            if unique_tsub == -1:
                continue
            for j in range(len(blocks_selected)):
                if j == i:
                    continue
                other_idx = blocks_selected[j]
                other_mask = block_masks[other_idx]
                if other_mask >> unique_tsub & 1:
                    blocks_selected.pop(i)
                    selected_set.remove(b_idx)
                    blocks_selected.pop(j)
                    selected_set.remove(other_idx)
                    blocks_selected.append(other_idx)
                    selected_set.add(other_idx)
                    # Update cover counts: remove b_idx's contribution, re-add other_idx's
                    m1 = mask
                    while m1:
                        lsb1 = m1 & (-m1)
                        bp1 = lsb1.bit_length() - 1
                        m1 ^= lsb1
                        cover_count[bp1] -= 1
                    m2 = other_mask
                    while m2:
                        lsb2 = m2 & (-m2)
                        bp2 = lsb2.bit_length() - 1
                        m2 ^= lsb2
                        cover_count[bp2] += 1
                    return True
        return removed_any

    def remove_redundant(blocks_selected, selected_set, cover_count, block_masks, deadline):
        removed_any = False
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for i in range(len(blocks_selected) - 1, -1, -1):
                if time.time() > deadline - 1.0:
                    return removed_any
                b_idx = blocks_selected[i]
                mask = block_masks[b_idx]
                m = mask
                redundant = True
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    m ^= lsb
                    if cover_count[bit_pos] == 1:
                        redundant = False
                        break
                if redundant:
                    m = mask
                    while m:
                        lsb = m & (-m)
                        bit_pos = lsb.bit_length() - 1
                        m ^= lsb
                        cover_count[bit_pos] -= 1
                    blocks_selected.pop(i)
                    selected_set.remove(b_idx)
                    changed = True
                    removed_any = True
                    break
        return removed_any

    total_removals = 0
    while time.time() < deadline - 3.0:
        r1 = remove_redundant(blocks_selected, selected_set, cover_count, block_masks, deadline)
        total_removals += r1
        r2 = try_swap_remove(blocks_selected, selected_set, cover_count, block_masks, deadline)
        if not r1 and not r2:
            break

    # Final redundancy removal pass
    remove_redundant(blocks_selected, selected_set, cover_count, block_masks, deadline)

    result = []
    for idx in blocks_selected:
        result.append(list(all_k_subsets[idx]))
    return result

# EVOLVE-BLOCK-END
