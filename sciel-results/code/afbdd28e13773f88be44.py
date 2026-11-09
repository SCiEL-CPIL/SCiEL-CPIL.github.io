def greedy_cover_fast(masks, uncovered, num_blocks):
    chosen = []
    used = [False] * num_blocks
    while uncovered:
        best_gain = -1
        best_idx = -1
        # Local variables for speed
        masks_local = masks
        used_local = used
        for i in range(num_blocks):
            if used_local[i]:
                continue
            gain = (masks_local[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        chosen.append(best_idx)
        used[best_idx] = True
        uncovered &= ~masks_local[best_idx]
    return chosen
