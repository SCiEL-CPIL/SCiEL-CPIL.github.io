import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices (35 each) for fast iteration
block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Weighted greedy weights: rare subsets get higher priority
weights = [1.0 / f for f in subset_frequencies]


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 112  # Safety margin below 120s kill

    def weighted_greedy():
        """Weighted greedy: pick block maximizing sum of weights over uncovered subsets it covers."""
        uncovered_mask = (1 << N_SUBSETS) - 1
        chosen = []
        while uncovered_mask:
            best_idx = -1
            best_gain = 0.0
            for i in range(N_BLOCKS):
                gain = 0.0
                for bit in block_bit_indices[i]:
                    if (uncovered_mask >> bit) & 1:
                        gain += weights[bit]
                        if gain <= best_gain:
                            # can't beat best even if rest added? no, keep accumulating
                            pass
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx < 0 or best_gain == 0.0:
                break
            chosen.append(best_idx)
            uncovered_mask &= ~all_masks[best_idx]
        return chosen

    # 1. Weighted greedy initialization
    current_blocks = weighted_greedy()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # Build counts (list of ints, fast)
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def add_block(idx):
        current_blocks.append(idx)
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def remove_block(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            counts[bit] -= 1
        current_blocks.pop(pos)

    def is_redundant(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True

    def find_redundant():
        for pos in range(len(current_blocks)):
            if is_redundant(pos):
                return pos
        return -1

    # 2. Local search: repeatedly remove redundant blocks, then do swaps
    no_improve = 0
    max_no_improve = 60

    while time.time() < deadline:
        # Phase A: prune all redundant blocks
        progress = True
        while progress:
            progress = False
            pos = find_redundant()
            if pos >= 0:
                remove_block(pos)
                progress = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve = 0
                continue

        if len(current_blocks) == 0:
            break

        # Phase B: try a 1-for-1 swap (remove one, add one that recovers uncovered)
        # pick a random block to remove
        i = random.randint(0, len(current_blocks) - 1)
        idx_i = current_blocks[i]
        uncovered_bits = []
        for bit in block_bit_indices[idx_i]:
            if counts[bit] == 1:
                uncovered_bits.append(bit)

        if not uncovered_bits:
            # it is redundant (should have been pruned, but counts may have shifted)
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve = 0
            continue

        # Need a replacement block covering all uncovered_bits
        # intersect candidate sets of uncovered bits
        if len(uncovered_bits) > 4:
            no_improve += 1
            if no_improve > max_no_improve:
                # perturbation: add a random block then re-prune
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                no_improve = 0
            continue

        # find candidate blocks covering all uncovered bits
        cands = None
        for b in uncovered_bits:
            lst = blocks_containing_subset[b]
            if cands is None:
                cands = set(lst)
            else:
                cands.intersection_update(lst)
            if not cands:
                break

        if cands:
            cands.discard(idx_i)
            if cands:
                j = random.choice(list(cands))
                if j not in current_blocks:
                    remove_block(i)
                    add_block(j)
                    no_improve = 0
                    continue

        no_improve += 1
        if no_improve > max_no_improve:
            # perturbation: add a couple random blocks then prune
            for _ in range(random.randint(1, 3)):
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
            no_improve = 0

    # 3. If time remains, do a fresh weighted-greedy restart and keep the better solution
    if time.time() < deadline - 15:
        alt = weighted_greedy()
        if len(alt) < best_len:
            best_len = len(alt)
            best_blocks = alt
            # quick prune pass on the alt solution
            counts2 = [0] * N_SUBSETS
            for idx in best_blocks:
                for bit in block_bit_indices[idx]:
                    counts2[bit] += 1
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for pos in range(len(best_blocks)):
                    idx = best_blocks[pos]
                    redundant = True
                    for bit in block_bit_indices[idx]:
                        if counts2[bit] == 1:
                            redundant = False
                            break
                    if redundant:
                        for bit in block_bit_indices[idx]:
                            counts2[bit] -= 1
                        best_blocks.pop(pos)
                        changed = True
                        if len(best_blocks) < best_len:
                            best_len = len(best_blocks)
                        break

    return [list(all_blocks[i]) for i in best_blocks]
