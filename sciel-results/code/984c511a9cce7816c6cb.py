import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    rng = random.Random(42)

    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}

    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    # Build bitmasks for each block and reverse mapping
    block_masks = [0] * n_k
    t_to_k = [[] for _ in range(n_t)]
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        block_masks[i] = mask

    full_mask = (1 << n_t) - 1

    def greedy_cover(seed=None):
        """Greedy set cover with incremental gain updates."""
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        uncovered = full_mask
        selected = []
        used_set = set()
        gain = [m.bit_count() for m in block_masks]

        while uncovered:
            if time.time() > deadline:
                break

            # Find max gain
            best_gain = 0
            best_idx = -1
            for i in range(n_k):
                if i in used_set:
                    continue
                g = gain[i]
                if g > best_gain:
                    best_gain = g
                    best_idx = i

            if best_gain == 0:
                break

            # Random tie-breaking: collect all with best_gain
            if best_gain > 0:
                candidates = []
                for i in range(n_k):
                    if i in used_set:
                        continue
                    if gain[i] == best_gain:
                        candidates.append(i)
                best_idx = local_rng.choice(candidates)

            selected.append(best_idx)
            used_set.add(best_idx)

            newly = uncovered & block_masks[best_idx]
            uncovered &= ~block_masks[best_idx]

            nc = newly
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                nc ^= lsb
                for blk in t_to_k[j]:
                    if blk not in used_set:
                        gain[blk] -= 1

        counts = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            m = block_masks[b]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb

        return selected, counts

    def greedy_cover_weighted(seed=None):
        """Greedy with weighted scoring: prefer blocks covering rare t-subsets."""
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        uncovered = full_mask
        selected = []
        used_set = set()
        
        # Count how many blocks cover each t-subset
        t_freq = np.zeros(n_t, dtype=np.int32)
        for i in range(n_k):
            m = block_masks[i]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                t_freq[idx] += 1
                m ^= lsb
        
        # Score = sum of 1/freq for uncovered t-subsets in block
        score = [0.0] * n_k
        for i in range(n_k):
            m = block_masks[i]
            s = 0.0
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                s += 1.0 / t_freq[idx]
                m ^= lsb
            score[i] = s

        while uncovered:
            if time.time() > deadline:
                break

            best_score = 0.0
            candidates = []
            for i in range(n_k):
                if i in used_set:
                    continue
                if score[i] > best_score:
                    best_score = score[i]
                    candidates = [i]
                elif score[i] == best_score:
                    candidates.append(i)

            if best_score == 0.0:
                break

            best_idx = local_rng.choice(candidates)
            selected.append(best_idx)
            used_set.add(best_idx)

            newly = uncovered & block_masks[best_idx]
            uncovered &= ~block_masks[best_idx]

            nc = newly
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                nc ^= lsb
                for blk in t_to_k[j]:
                    if blk not in used_set:
                        score[blk] -= 1.0 / t_freq[j]

        counts = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            m = block_masks[b]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb

        return selected, counts

    def remove_redundant(blocks, counts):
        """Remove blocks that don't uniquely cover any t-subset."""
        if time.time() > deadline:
            return blocks, counts

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []

            for b in blocks:
                m = block_masks[b]
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                if not essential:
                    to_remove.append(b)

            if to_remove:
                remove_set = set(to_remove)
                blocks = [b for b in blocks if b not in remove_set]
                for b in to_remove:
                    m = block_masks[b]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                changed = True

        return blocks, counts

    def swap_improve(blocks, counts):
        """Try 1-for-1 swaps to reduce block count."""
        if time.time() > deadline:
            return blocks, counts

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            block_set = set(blocks)
            n_blocks = len(blocks)

            for i in range(n_blocks):
                if time.time() > deadline:
                    break

                b = blocks[i]
                mask_remove = block_masks[b]

                # Temporarily remove
                m = mask_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb

                # Find uncovered
                uncovered = 0
                for j in range(n_t):
                    if counts[j] == 0:
                        uncovered |= (1 << j)

                if uncovered == 0:
                    blocks = [x for j, x in enumerate(blocks) if j != i]
                    improved = True
                    break

                # Try to find a single block that covers all uncovered
                n_uncov = uncovered.bit_count()
                best_new = -1
                best_gain = 0
                for new_b in range(n_k):
                    if new_b == b or new_b in block_set:
                        continue
                    g = (uncovered & block_masks[new_b]).bit_count()
                    if g > best_gain:
                        best_gain = g
                        best_new = new_b
                        if g == n_uncov:
                            break

                # Restore
                m = mask_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb

                if best_new != -1 and best_gain == n_uncov:
                    # Swap
                    m = mask_remove
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb

                    blocks[i] = best_new
                    m = block_masks[best_new]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb
                    improved = True
                    break

        return blocks, counts

    def two_opt(blocks, counts):
        """Try 2-for-1 swaps: remove 2 blocks, add 1."""
        if time.time() > deadline:
            return blocks, counts

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            n = len(blocks)
            block_set = set(blocks)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break

                    b1, b2 = blocks[i], blocks[j]
                    mask1, mask2 = block_masks[b1], block_masks[b2]

                    # Temporarily remove both
                    m = mask1
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    m = mask2
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb

                    # Find uncovered
                    uncovered = 0
                    for t_idx in range(n_t):
                        if counts[t_idx] == 0:
                            uncovered |= (1 << t_idx)

                    if uncovered == 0:
                        # Both were redundant
                        new_blocks = [x for p, x in enumerate(blocks) if p != i and p != j]
                        counts2 = np.zeros(n_t, dtype=np.int32)
                        for b in new_blocks:
                            m = block_masks[b]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                counts2[idx] += 1
                                m ^= lsb
                        blocks = new_blocks
                        counts = counts2
                        improved = True
                        break

                    # Try to find one block covering all uncovered
                    n_uncov = uncovered.bit_count()
                    best_new = -1
                    best_gain = 0
                    for new_b in range(n_k):
                        if new_b == b1 or new_b == b2 or new_b in block_set:
                            continue
                        g = (uncovered & block_masks[new_b]).bit_count()
                        if g > best_gain:
                            best_gain = g
                            best_new = new_b
                            if g == n_uncov:
                                break

                    # Restore
                    m = mask1
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb
                    m = mask2
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb

                    if best_new != -1 and best_gain == n_uncov:
                        # Do the swap
                        m = mask1
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb
                        m = mask2
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb

                        new_blocks = [x for p, x in enumerate(blocks) if p != i and p != j]
                        new_blocks.append(best_new)
                        m = block_masks[best_new]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            m ^= lsb
                        blocks = new_blocks
                        improved = True
                        break

                if improved:
                    break

        return blocks, counts

    def cover_holes(blocks, holes):
        """Greedily cover remaining holes."""
        new_blocks = []
        uncovered = holes
        used = set(blocks)
        gain = [(uncovered & block_masks[i]).bit_count() for i in range(n_k)]

        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k):
                if i in used:
                    continue
                if gain[i] > best_gain:
                    best_gain = gain[i]
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            used.add(best_idx)
            newly = uncovered & block_masks[best_idx]
            uncovered &= ~block_masks[best_idx]
            nc = newly
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                nc ^= lsb
                for blk in t_to_k[j]:
                    if blk not in used:
                        gain[blk] -= 1
        return new_blocks

    def rebuild_counts(blocks):
        counts = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            m = block_masks[b]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    def coverage_mask(blocks):
        mask = 0
        for b in blocks:
            mask |= block_masks[b]
        return mask

    def polish(blocks, counts):
        """Apply remove_redundant, swap_improve, two_opt repeatedly."""
        for _ in range(3):
            if time.time() > deadline:
                break
            blocks, counts = remove_redundant(blocks, counts)
            blocks, counts = swap_improve(blocks, counts)
            blocks, counts = two_opt(blocks, counts)
        return blocks, counts

    # === Main search ===
    best_blocks = None
    best_counts = None
    best_count = float('inf')

    # 1. Deterministic greedy
    blocks, counts = greedy_cover(seed=None)
    blocks, counts = polish(blocks, counts)
    best_blocks = blocks
    best_counts = counts
    best_count = len(blocks)

    # 2. Weighted greedy
    if time.time() < deadline - 40:
        blocks, counts = greedy_cover_weighted(seed=777)
        blocks, counts = polish(blocks, counts)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_counts = counts
            best_count = len(blocks)

    # 3. Multiple randomized restarts
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        seed = 100 + r * 7919
        blocks, counts = greedy_cover(seed=seed)
        blocks, counts = polish(blocks, counts)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_counts = counts
            best_count = len(blocks)

    # 4. Weighted greedy restarts
    for r in range(10):
        if time.time() > deadline - 30:
            break
        seed = 500 + r * 31337
        blocks, counts = greedy_cover_weighted(seed=seed)
        blocks, counts = polish(blocks, counts)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_counts = counts
            best_count = len(blocks)

    # 5. Perturbation-based local search
    attempts = 0
    while time.time() < deadline - 15:
        attempts += 1
        if attempts % 3 == 0:
            num_remove = min(len(best_blocks), rng.randint(4, 8))
        else:
            num_remove = min(len(best_blocks), rng.randint(1, 4))
        if num_remove == 0:
            break

        to_remove = rng.sample(best_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed = [b for b in best_blocks if b not in remove_set]

        covered = coverage_mask(perturbed)
        holes = full_mask & ~covered

        added = cover_holes(perturbed, holes)
        new_blocks = perturbed + added

        if len(new_blocks) < best_count:
            new_counts = rebuild_counts(new_blocks)
            new_blocks, new_counts = polish(new_blocks, new_counts)
            if len(new_blocks) < best_count:
                best_blocks = new_blocks
                best_counts = new_counts
                best_count = len(new_blocks)

    # 6. Final polish
    for _ in range(3):
        if time.time() > deadline - 3:
            break
        best_blocks, best_counts = polish(best_blocks, best_counts)

    # 7. One more perturbation attempt
    if time.time() < deadline - 3 and len(best_blocks) > 10:
        n_rem = rng.randint(3, min(8, len(best_blocks) // 4))
        to_rem = rng.sample(best_blocks, n_rem)
        rem_set = set(to_rem)
        base = [b for b in best_blocks if b not in rem_set]
        cov = coverage_mask(base)
        holes = full_mask & ~cov
        added = cover_holes(base, holes)
        cand = base + added
        if len(cand) < best_count:
            cand_counts = rebuild_counts(cand)
            cand, cand_counts = polish(cand, cand_counts)
            if len(cand) < best_count:
                best_blocks = cand
                best_counts = cand_counts
                best_count = len(cand)

    final_blocks = [list(all_k_blocks[i]) for i in best_blocks]
    return final_blocks
