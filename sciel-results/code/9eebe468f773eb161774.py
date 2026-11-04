import itertools
import random
import time

def construct(v, k, t):
    start = time.time()
    deadline = start + 118
    rng = random.Random(1234567890)

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}

    # Precompute block masks and per-subset block lists
    block_masks = [0] * n_blocks
    subset_blocks = [[] for _ in range(n_subsets)]
    for bi, blk in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(blk, t):
            si = sub_idx[sub]
            mask |= 1 << si
            subset_blocks[si].append(bi)
        block_masks[bi] = mask
    for si in range(n_subsets):
        subset_blocks[si].sort(key=lambda bi: -block_masks[bi].bit_count())

    def greedy_construction(rng, target_size):
        """Greedy: always pick block maximizing new coverage, stop when size >= target."""
        uncovered = (1 << n_subsets) - 1
        chosen = []
        chosen_set = set()
        while uncovered and len(chosen) < target_size:
            # Find best block for new coverage
            best_bi = -1
            best_gain = 0
            best_mask = 0
            for bi in range(n_blocks):
                if bi in chosen_set:
                    continue
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    best_mask = block_masks[bi]
                    if best_gain == uncovered.bit_count():
                        break
            if best_bi < 0:
                break
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            uncovered &= ~best_mask
        # Continue if not covered (shouldn't happen but for safety)
        while uncovered:
            best_bi = -1
            best_gain = 0
            best_mask = 0
            for bi in range(n_blocks):
                if bi in chosen_set:
                    continue
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    best_mask = block_masks[bi]
            if best_bi < 0:
                break
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            uncovered &= ~best_mask
        return chosen, uncovered == 0

    def prune(blocks_list):
        """Remove redundant blocks. Returns list of block indices and their counts."""
        chosen = list(blocks_list)
        chosen_set = set(chosen)
        counts = [0] * n_subsets
        for bi in chosen:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
        # Iteratively remove redundant
        changed = True
        while changed:
            changed = False
            for i, bi in enumerate(chosen):
                mask = block_masks[bi]
                redundant = True
                while mask:
                    low = mask & -mask
                    si = low.bit_length() - 1
                    if counts[si] == 1:
                        redundant = False
                        break
                    mask ^= low
                if redundant:
                    # remove
                    mask = block_masks[bi]
                    while mask:
                        low = mask & -mask
                        si = low.bit_length() - 1
                        counts[si] -= 1
                        mask ^= low
                    chosen.pop(i)
                    chosen_set.discard(bi)
                    changed = True
                    break
        return chosen, counts

    def swap_search(blocks_list, counts, time_budget):
        """Do 1-out/1-in and 1-out/2-in swaps, plus pruning."""
        chosen = list(blocks_list)
        chosen_set = set(chosen)
        current_time = time.time()
        
        def can_remove(bi):
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                if counts[si] == 1:
                    return False
                mask ^= low
            return True

        def remove_block(bi):
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] -= 1
                mask ^= low
            chosen_set.discard(bi)

        def add_block(bi):
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
            chosen_set.add(bi)

        improved = True
        max_rounds = 50
        for _round in range(max_rounds):
            if time.time() - current_time > time_budget:
                break
            improved = False
            # 1-out/1-in swap
            for i, bi_out in enumerate(chosen):
                if time.time() - current_time > time_budget:
                    break
                if not can_remove(bi_out):
                    continue
                # Find best replacement
                best_bi_in = -1
                best_gain = 0
                for bi_in in range(n_blocks):
                    if bi_in in chosen_set:
                        continue
                    mask = block_masks[bi_in]
                    # gain = number of subsets that would have count 0 after removal that are covered
                    # Simplified: just count coverage overlap with what bi_out covers
                    gain = (mask & block_masks[bi_out]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_bi_in = bi_in
                if best_bi_in < 0:
                    continue
                # Do swap
                remove_block(bi_out)
                chosen[i] = best_bi_in
                add_block(best_bi_in)
                improved = True
                # Try to prune
                j = 0
                while j < len(chosen):
                    if can_remove(chosen[j]):
                        remove_block(chosen[j])
                        chosen.pop(j)
                        improved = True
                        break
                    j += 1
                if time.time() - current_time > time_budget:
                    break
            # 1-out/2-in swap
            for i, bi_out in enumerate(chosen):
                if time.time() - current_time > time_budget:
                    break
                if not can_remove(bi_out):
                    continue
                # Find best pair
                best_pair = None
                best_gain = 0
                mask_out = block_masks[bi_out]
                # Try all pairs is too expensive; use greedy pair selection
                # First, find best single
                candidates = []
                for bi in range(n_blocks):
                    if bi in chosen_set:
                        continue
                    gain = (block_masks[bi] & mask_out).bit_count()
                    if gain > 0:
                        candidates.append((gain, bi))
                candidates.sort(key=lambda x: -x[0])
                # Take top few and try pairs
                top = candidates[:20]
                for a_idx in range(len(top)):
                    for b_idx in range(a_idx + 1, len(top)):
                        g1, bi1 = top[a_idx]
                        g2, bi2 = top[b_idx]
                        combined = block_masks[bi1] | block_masks[bi2]
                        gain = (combined & mask_out).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (bi1, bi2)
                if best_pair and best_gain >= mask_out.bit_count():
                    remove_block(bi_out)
                    chosen.pop(i)
                    add_block(best_pair[0])
                    add_block(best_pair[1])
                    improved = True
                elif best_pair and best_gain > 0:
                    # Partial coverage, might still help
                    pass
            if not improved:
                break

        # Final prune
        changed = True
        while changed:
            changed = False
            for i, bi in enumerate(chosen):
                if can_remove(bi):
                    remove_block(bi)
                    chosen.pop(i)
                    changed = True
                    break
        return chosen, counts

    # Main search
    best_size = float('inf')
    best_blocks = None

    # Initial greedy with several random seeds (varying target size)
    for target in [155, 160, 165, 170, 175, 180, 190, 200, 220, 250, 300]:
        if time.time() > deadline - 5:
            break
        chosen, valid = greedy_construction(rng, target)
        if not valid:
            continue
        pruned, counts = prune(chosen)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_blocks = pruned[:]

    # Local search from best
    if best_blocks:
        if time.time() < deadline - 10:
            chosen, counts = prune(best_blocks[:])
            chosen, counts = swap_search(chosen, counts, time_budget=15)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_blocks = chosen[:]

    # ILS: perturb and local search
    if best_blocks:
        while time.time() < deadline - 5:
            # Perturb: remove random 2-5 blocks, re-add greedy
            perturb_size = rng.randint(2, 5)
            perturbed = best_blocks[:]
            for _ in range(perturb_size):
                if perturbed:
                    idx = rng.randrange(len(perturbed))
                    perturbed.pop(idx)
            # Re-add greedily until valid
            counts = [0] * n_subsets
            chosen_set = set(perturbed)
            for bi in perturbed:
                mask = block_masks[bi]
                while mask:
                    low = mask & -mask
                    si = low.bit_length() - 1
                    counts[si] += 1
                    mask ^= low
            uncovered = 0
            for si in range(n_subsets):
                if counts[si] == 0:
                    uncovered |= 1 << si
            while uncovered:
                best_bi = -1
                best_gain = 0
                for bi in range(n_blocks):
                    if bi in chosen_set:
                        continue
                    gain = (block_masks[bi] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
                if best_bi < 0:
                    break
                perturbed.append(best_bi)
                chosen_set.add(best_bi)
                mask = block_masks[best_bi]
                while mask:
                    low = mask & -mask
                    si = low.bit_length() - 1
                    counts[si] += 1
                    mask ^= low
                uncovered &= ~block_masks[best_bi]
            # Prune
            pruned, _ = prune(perturbed)
            # Local search
            if time.time() < deadline - 3:
                pruned, counts2 = prune(pruned)
                pruned, counts2 = swap_search(pruned, counts2, time_budget=3)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_blocks = pruned[:]

    # Final prune
    if best_blocks:
        pruned, _ = prune(best_blocks)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_blocks = pruned

    if not best_blocks:
        # Fallback: pure greedy
        chosen, valid = greedy_construction(rng, 200)
        best_blocks = chosen

    return [list(all_blocks[bi]) for bi in best_blocks]
