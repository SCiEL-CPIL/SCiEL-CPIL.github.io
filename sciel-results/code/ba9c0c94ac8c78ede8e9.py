import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 130.0  # Safety margin: kill at 150s, leave 20s buffer

    def elapsed():
        return time.time() - start_time

    # 1. Generate all t-subsets and map to index
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)

    # 2. Generate all k-blocks and compute their bitmask coverage
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Precompute block masks: which t-subsets each block covers
    block_masks = np.zeros(num_k, dtype=np.uint64)
    # For v=19, k=8, t=5: C(8,5)=56 t-subsets per block, num_t=C(19,5)=11628
    # 11628 bits doesn't fit in uint64, so we need a different approach
    # Use Python integers (arbitrary precision) for masks
    block_masks = [0] * num_k
    for idx, block in enumerate(k_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            mask |= (1 << t_index_map[combo])
        block_masks[idx] = mask

    full_mask = (1 << num_t) - 1

    # Greedy set cover with optional randomization
    def greedy_set_cover(randomize=False, seed=42, alpha=0.0):
        if randomize:
            rng = random.Random(seed)
        else:
            rng = None

        uncovered = full_mask
        selected = []
        selected_set = set()

        while uncovered and elapsed() < deadline - 5:
            best_idx = -1
            best_gain = -1
            # For randomization, collect all candidates with gain close to max
            if randomize and alpha > 0:
                # Find max gain first
                gains = []
                for i in range(num_k):
                    if i in selected_set:
                        continue
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > 0:
                        gains.append((gain, i))
                if not gains:
                    break
                max_gain = max(g for g, _ in gains)
                # Select from top candidates with gain >= max_gain * (1 - alpha)
                threshold = max_gain * (1 - alpha)
                candidates = [(g, i) for g, i in gains if g >= threshold]
                # Weighted random selection
                weights = [g for g, _ in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                for (g, i), w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        best_idx = i
                        best_gain = g
                        break
            else:
                # Deterministic: find max gain
                for i in range(num_k):
                    if i in selected_set:
                        continue
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i

            if best_idx == -1 or best_gain <= 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

        return selected

    # Remove redundant blocks
    def remove_redundant(selected_indices):
        if not selected_indices:
            return selected_indices
        changed = True
        while changed and elapsed() < deadline - 5:
            changed = False
            new_selected = []
            for i, idx in enumerate(selected_indices):
                # Check if block idx is redundant
                others_mask = 0
                for j, jdx in enumerate(selected_indices):
                    if i != j:
                        others_mask |= block_masks[jdx]
                if (block_masks[idx] & ~others_mask) == 0:
                    changed = True
                else:
                    new_selected.append(idx)
            selected_indices = new_selected
        return selected_indices

    # Local search polish: try to replace a block with one that covers same or more
    def local_search_polish(selected_indices):
        if not selected_indices:
            return selected_indices
        selected_set = set(selected_indices)
        improved = True
        while improved and elapsed() < deadline - 5:
            improved = False
            for i, idx in enumerate(selected_indices):
                # Find t-subsets that are ONLY covered by this block
                others_mask = 0
                for j, jdx in enumerate(selected_indices):
                    if i != j:
                        others_mask |= block_masks[jdx]
                unique_to_idx = block_masks[idx] & ~others_mask
                if unique_to_idx == 0:
                    # This block is redundant, remove it
                    selected_indices = [x for j, x in enumerate(selected_indices) if j != i]
                    selected_set.discard(idx)
                    improved = True
                    break
                # Try to find a replacement block that covers unique_to_idx and adds more
                best_replace = -1
                best_new_gain = -1
                for new_idx in range(num_k):
                    if new_idx in selected_set:
                        continue
                    # Gain: how many new t-subsets does new_idx cover that aren't covered by others
                    # Others = all blocks except idx
                    # We want new_idx to cover unique_to_idx (mandatory)
                    if (block_masks[new_idx] & unique_to_idx) != unique_to_idx:
                        continue
                    # Additional gain: t-subsets covered by new_idx but not by (others + idx)
                    # Actually, we're replacing idx, so new coverage = block_masks[new_idx] & ~(others_mask)
                    # But we need to ensure no t-subset becomes uncovered
                    # t-subsets that would be lost: unique_to_idx & ~block_masks[new_idx] = 0 (already checked)
                    # Additional gain: block_masks[new_idx] & ~others_mask & ~block_masks[idx]
                    add_gain = (block_masks[new_idx] & ~others_mask & ~block_masks[idx]).bit_count()
                    if add_gain > best_new_gain:
                        best_new_gain = add_gain
                        best_replace = new_idx
                if best_replace != -1:
                    selected_indices[i] = best_replace
                    selected_set.discard(idx)
                    selected_set.add(best_replace)
                    improved = True
                    break
        return selected_indices

    # Run multiple strategies and keep the best
    best_result = None
    best_count = float('inf')

    # Strategy 1: Deterministic greedy
    sel = greedy_set_cover(randomize=False)
    sel = remove_redundant(sel)
    sel = local_search_polish(sel)
    if len(sel) < best_count:
        best_count = len(sel)
        best_result = sel

    # Strategy 2: Randomized greedy with various parameters
    for seed in range(20):
        if elapsed() > deadline - 10:
            break
        for alpha in [0.0, 0.1, 0.2, 0.3]:
            if elapsed() > deadline - 10:
                break
            sel = greedy_set_cover(randomize=True, seed=seed * 100 + int(alpha * 100), alpha=alpha)
            sel = remove_redundant(sel)
            sel = local_search_polish(sel)
            if len(sel) < best_count:
                best_count = len(sel)
                best_result = sel

    if best_result is None:
        # Fallback: should not happen, but just in case
        best_result = greedy_set_cover(randomize=False)
        best_result = remove_redundant(best_result)

    return [k_blocks[i] for i in best_result]
