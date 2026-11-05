import itertools
import time
import random
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

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute numpy array of masks for vectorized gain computation
# Each mask is a Python int; convert to numpy array of uint64 pairs won't work directly
# Instead, precompute coverage counts per subset for each block
# all_blocks_covered[i] = list of subset indices covered by block i
all_blocks_covered = []
for i in range(N_BLOCKS):
    m = all_masks[i]
    covered = []
    mm = m
    while mm:
        lsb = mm & -mm
        bit_idx = lsb.bit_length() - 1
        covered.append(bit_idx)
        mm ^= lsb
    all_blocks_covered.append(covered)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy
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
    deadline = start_time + 115  # Safety margin

    # 1. Greedy Initialization - optimized
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        for bit_idx in all_blocks_covered[idx]:
            counts[bit_idx] += 1

    def remove_block(i):
        for bit_idx in all_blocks_covered[current_blocks[i]]:
            counts[bit_idx] -= 1
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        for bit_idx in all_blocks_covered[current_blocks[i]]:
            if counts[bit_idx] == 1:
                return False
        return True

    # Greedy: pick max gain each step
    uncovered_count = N_SUBSETS
    uncovered_mask = (1 << N_SUBSETS) - 1
    
    while uncovered_count > 0:
        if time.time() > deadline - 60:
            break
        best_idx = -1
        best_gain = 0
        for i in range(N_BLOCKS):
            gain = 0
            for bit_idx in all_blocks_covered[i]:
                if (uncovered_mask >> bit_idx) & 1:
                    gain += 1
                    if gain <= best_gain:
                        break
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        add_block(best_idx)
        for bit_idx in all_blocks_covered[best_idx]:
            uncovered_mask &= ~(1 << bit_idx)
        uncovered_count -= best_gain
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search
    no_improve_count = 0
    max_no_improve = 20
    max_no_improve_cap = 100
    
    while time.time() < deadline:
        # Phase 1: Prune redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break
        
        if removed_any:
            no_improve_count = 0
            max_no_improve = max(15, max_no_improve // 2)
            continue
            
        # Phase 2: Try 1-for-1 swap
        if len(current_blocks) == 0:
            break
            
        found_swap = False
        for attempt in range(8):
            if time.time() > deadline:
                break
            i = random.randint(0, len(current_blocks) - 1)
            
            # Find subsets that would become uncovered if we remove block i
            m_i = current_masks[i]
            uncovered_bits = []
            for bit_idx in all_blocks_covered[i]:
                if counts[bit_idx] == 1:
                    uncovered_bits.append(bit_idx)
                
            if not uncovered_bits:
                # It was redundant, remove it
                remove_block(i)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                max_no_improve = max(15, max_no_improve // 2)
                found_swap = True
                break
            elif len(uncovered_bits) <= 4:
                # Find a block that covers all uncovered_bits
                # Use the rarest uncovered bit to filter candidates
                uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                
                # Start with candidates for the rarest bit
                candidates = set(blocks_containing_subset[uncovered_bits[0]])
                
                for b in uncovered_bits[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                    
                if candidates:
                    valid_candidates = [c for c in candidates if c != i]
                    if valid_candidates:
                        # Prefer candidates that also cover other subsets
                        best_cand = random.choice(valid_candidates)
                        remove_block(i)
                        add_block(best_cand)
                        found_swap = True
                        no_improve_count = 0
                        break
        
        if found_swap:
            continue
            
        # Phase 3: Try removing one block and adding two different blocks
        if time.time() > deadline - 20:
            break
            
        i = random.randint(0, len(current_blocks) - 1)
        uncovered_bits = []
        for bit_idx in all_blocks_covered[i]:
            if counts[bit_idx] == 1:
                uncovered_bits.append(bit_idx)
            
        if 2 <= len(uncovered_bits) <= 6:
            # Split uncovered bits into two groups
            split = len(uncovered_bits) // 2
            bits1 = uncovered_bits[:split]
            bits2 = uncovered_bits[split:]
            
            # Find blocks covering bits1
            cands1 = set(blocks_containing_subset[bits1[0]])
            for b in bits1[1:]:
                cands1 &= set(blocks_containing_subset[b])
            
            # Find blocks covering bits2
            cands2 = set(blocks_containing_subset[bits2[0]])
            for b in bits2[1:]:
                cands2 &= set(blocks_containing_subset[b])
            
            cands1.discard(i)
            cands2.discard(i)
            cands2.discard(cands1.pop() if len(cands1) == 1 else -1)
            
            if cands1 and cands2:
                c1 = random.choice(list(cands1))
                c2 = random.choice(list(cands2))
                if c1 != c2:
                    remove_block(i)
                    add_block(c1)
                    add_block(c2)
                    no_improve_count = 0
                    continue
        
        # Phase 4: Perturbation
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            # Perturbation: remove several random blocks and re-add via greedy
            n_remove = random.randint(4, 10)
            to_remove = random.sample(range(len(current_blocks)), min(n_remove, len(current_blocks)))
            to_remove.sort(reverse=True)
            for i in to_remove:
                remove_block(i)
            
            # Re-add blocks greedily to cover uncovered subsets
            if time.time() < deadline - 5:
                # Find uncovered bits
                uncov_bits = [b for b in range(N_SUBSETS) if counts[b] == 0]
                if uncov_bits:
                    uncov_set = set(uncov_bits)
                    # Greedily add blocks
                    while uncov_set and time.time() < deadline - 5:
                        best_idx = -1
                        best_gain = 0
                        for idx in range(N_BLOCKS):
                            if idx in current_blocks:
                                continue
                            gain = 0
                            for bit_idx in all_blocks_covered[idx]:
                                if bit_idx in uncov_set:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_idx = idx
                        if best_idx == -1 or best_gain == 0:
                            break
                        add_block(best_idx)
                        for bit_idx in all_blocks_covered[best_idx]:
                            uncov_set.discard(bit_idx)
            
            no_improve_count = 0
            max_no_improve = min(max_no_improve_cap, max_no_improve * 2)

    # Final cleanup: prune any remaining redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            if is_redundant(i):
                remove_block(i)
                changed = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()

    return [list(all_blocks[i]) for i in best_blocks]
