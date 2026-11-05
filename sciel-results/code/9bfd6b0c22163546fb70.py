import itertools
import time
import random
import sys

V = 19
K = 7
T = 4

# Precomputation
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

# For each block, list of 4-subset indices it covers
block_subsets = []
for block in all_blocks:
    subs = [subset_to_idx[s] for s in itertools.combinations(block, T)]
    block_subsets.append(subs)

# For each 4-subset, list of blocks containing it
subset_blocks = [[] for _ in range(N_SUBSETS)]
for bi, subs in enumerate(block_subsets):
    for s in subs:
        subset_blocks[s].append(bi)

# Sort subset_blocks by rarity (fewest blocks first)
subset_rarity = [len(subset_blocks[s]) for s in range(N_SUBSETS)]
rarity_order = sorted(range(N_SUBSETS), key=lambda s: subset_rarity[s])


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k = list(itertools.combinations(range(v), k))
        random.shuffle(all_k)
        for b in all_k:
            if not uncovered:
                break
            cov = set(itertools.combinations(b, t))
            if uncovered & cov:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start = time.time()
    deadline = start + 118  # safety margin before 150s kill

    def remaining():
        return deadline - time.time()

    best_blocks = None
    best_len = float('inf')

    def update_best(blocks):
        nonlocal best_blocks, best_len
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

    # ---- Fast greedy with precomputed gains ----
    def fast_greedy(counts, in_cover, time_limit):
        """Greedy: repeatedly pick block with max new coverage."""
        # Track uncovered as a set for fast membership
        uncov = set()
        for s in range(N_SUBSETS):
            if counts[s] == 0:
                uncov.add(s)

        while uncov and time.time() < time_limit:
            best_bi = -1
            best_gain = 0
            for bi in range(N_BLOCKS):
                if in_cover[bi]:
                    continue
                gain = 0
                for s in block_subsets[bi]:
                    if s in uncov:
                        gain += 1
                        if gain <= best_gain:
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if best_gain == len(block_subsets[bi]):
                        break
            if best_bi == -1:
                break
            in_cover[best_bi] = True
            for s in block_subsets[best_bi]:
                counts[s] += 1
                if counts[s] == 1:
                    uncov.discard(s)
        return uncov

    # ---- Prune redundant blocks ----
    def prune(counts, in_cover):
        changed = True
        while changed:
            changed = False
            for bi in range(N_BLOCKS):
                if not in_cover[bi]:
                    continue
                redundant = True
                for s in block_subsets[bi]:
                    if counts[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_subsets[bi]:
                        counts[s] -= 1
                    in_cover[bi] = False
                    changed = True
        return [bi for bi in range(N_BLOCKS) if in_cover[bi]]

    # ---- Dual local search: swap to maximize min coverage ----
    def dual_local_search(counts, in_cover, time_limit, max_iters=2000):
        """Remove redundant, then swap blocks to minimize count."""
        iters = 0
        while time.time() < time_limit and iters < max_iters:
            iters += 1

            # Phase 1: remove any redundant block
            removed_any = False
            for bi in range(N_BLOCKS):
                if not in_cover[bi]:
                    continue
                redundant = True
                for s in block_subsets[bi]:
                    if counts[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_subsets[bi]:
                        counts[s] -= 1
                    in_cover[bi] = False
                    removed_any = True
            if removed_any:
                continue

            # Phase 2: try to swap a block with another to reduce total
            # Find a block that covers the most "scarce" subsets
            # and try to replace it with a block that covers those + more
            n_in = sum(in_cover)
            if n_in == 0:
                break

            # Pick a random block in cover to try replacing
            in_list = [bi for bi in range(N_BLOCKS) if in_cover[bi]]
            random.shuffle(in_list)
            swapped = False
            for bi_old in in_list[:min(15, len(in_list))]:
                if time.time() > time_limit:
                    break
                # Subsets uniquely covered by bi_old
                unique = []
                for s in block_subsets[bi_old]:
                    if counts[s] == 1:
                        unique.append(s)
                if not unique:
                    # It's redundant (shouldn't happen after prune, but safety)
                    for s in block_subsets[bi_old]:
                        counts[s] -= 1
                    in_cover[bi_old] = False
                    swapped = True
                    break

                # Find candidate that covers all unique subsets
                # Use the rarest unique subset to narrow candidates
                unique.sort(key=lambda s: subset_rarity[s])
                candidates = set(subset_blocks[unique[0]])
                for s in unique[1:]:
                    candidates &= set(subset_blocks[s])
                    if not candidates:
                        break

                # Filter: not in cover, and covers all unique
                valid = []
                for c in candidates:
                    if in_cover[c]:
                        continue
                    # Check it covers all unique
                    ok = True
                    cset = set(block_subsets[c])
                    for s in unique:
                        if s not in cset:
                            ok = False
                            break
                    if ok:
                        valid.append(c)

                if valid:
                    # Pick the one that maximizes total new coverage
                    best_c = valid[0]
                    best_new = 0
                    for c in valid:
                        new_cov = 0
                        cset = set(block_subsets[c])
                        for s in block_subsets[c]:
                            if s not in set(block_subsets[bi_old]):
                                if counts[s] == 0:
                                    new_cov += 1
                        if new_cov > best_new:
                            best_new = new_cov
                            best_c = c
                    # Perform swap
                    for s in block_subsets[bi_old]:
                        counts[s] -= 1
                    in_cover[bi_old] = False
                    for s in block_subsets[best_c]:
                        counts[s] += 1
                    in_cover[best_c] = True
                    swapped = True
                    break

            if not swapped:
                break

        return [bi for bi in range(N_BLOCKS) if in_cover[bi]]

    # ---- Perturbation + re-greedy ----
    def perturb_and_refill(counts, in_cover, time_limit):
        # Remove some random blocks
        in_list = [bi for bi in range(N_BLOCKS) if in_cover[bi]]
        n_remove = random.randint(3, 7)
        to_remove = random.sample(in_list, min(n_remove, len(in_list)))
        for bi in to_remove:
            for s in block_subsets[bi]:
                counts[s] -= 1
            in_cover[bi] = False

        # Re-greedy to fill gaps
        uncov = set()
        for s in range(N_SUBSETS):
            if counts[s] == 0:
                uncov.add(s)
        if uncov:
            fast_greedy(counts, in_cover, time_limit)
        return prune(counts, in_cover)

    # ---- Strategy 1: Probabilistic + alteration + dual local search ----
    p_values = [0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.028, 0.040]

    for p in p_values:
        if remaining() < 25:
            break

        counts = [0] * N_SUBSETS
        in_cover = [False] * N_BLOCKS

        # Random selection
        for bi in range(N_BLOCKS):
            if random.random() < p:
                in_cover[bi] = True
                for s in block_subsets[bi]:
                    counts[s] += 1

        # Fill gaps greedily
        uncov = set()
        for s in range(N_SUBSETS):
            if counts[s] == 0:
                uncov.add(s)
        if uncov:
            fast_greedy(counts, in_cover, deadline - 12)

        # Prune
        blocks = prune(counts, in_cover)
        update_best(blocks)

        # Dual local search
        if remaining() > 15:
            blocks = dual_local_search(counts, in_cover, deadline - 10, max_iters=3000)
            update_best(blocks)

        # Perturbation rounds
        for _ in range(5):
            if remaining() < 12:
                break
            blocks = perturb_and_refill(counts, in_cover, deadline - 8)
            update_best(blocks)
            blocks = dual_local_search(counts, in_cover, deadline - 8, max_iters=2000)
            update_best(blocks)

        if best_len <= 145:
            break

    # ---- Strategy 2: Pure greedy + dual local search ----
    if remaining() > 20:
        counts = [0] * N_SUBSETS
        in_cover = [False] * N_BLOCKS
        fast_greedy(counts, in_cover, deadline - 15)
        blocks = prune(counts, in_cover)
        update_best(blocks)

        blocks = dual_local_search(counts, in_cover, deadline - 8, max_iters=5000)
        update_best(blocks)

        # More perturbation
        for _ in range(8):
            if remaining() < 10:
                break
            blocks = perturb_and_refill(counts, in_cover, deadline - 6)
            update_best(blocks)
            blocks = dual_local_search(counts, in_cover, deadline - 6, max_iters=3000)
            update_best(blocks)

    # ---- Final polish: aggressive pruning ----
    if best_blocks and remaining() > 3:
        counts = [0] * N_SUBSETS
        in_cover = [False] * N_BLOCKS
        for bi in best_blocks:
            in_cover[bi] = True
            for s in block_subsets[bi]:
                counts[s] += 1
        blocks = prune(counts, in_cover)
        update_best(blocks)

    return [list(all_blocks[bi]) for bi in best_blocks]
