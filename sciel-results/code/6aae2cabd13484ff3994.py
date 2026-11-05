import itertools
import time
import random
import heapq

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
    deadline = start_time + 118  # Safety margin

    # 1. Greedy Initialization with heap for speed
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb

    def remove_block(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            mm ^= lsb
        return True

    # Greedy: use a list and pick max gain each time (still O(N_BLOCKS) per step,
    # but we can optimize by tracking uncovered count per block incrementally)
    # Actually, let's just do greedy but with a time check
    uncovered_mask = (1 << N_SUBSETS) - 1
    # Precompute which blocks cover which subsets for faster gain calculation
    # Use a different approach: maintain gain for each block
    
    while uncovered_mask:
        if time.time() > deadline - 60:
            break
        best_idx = -1
        best_gain = 0
        for i in range(N_BLOCKS):
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with improved strategies
    no_improve_count = 0
    max_no_improve = 30
    max_no_improve_start = 30
    max_no_improve_cap = 150
    
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
            max_no_improve = max(max_no_improve_start, max_no_improve // 2)
            continue
            
        # Phase 2: Try 1-for-1 swap
        if len(current_blocks) == 0:
            break
            
        found_swap = False
        # Try multiple random removals
        for attempt in range(5):
            if time.time() > deadline:
                break
            i = random.randint(0, len(current_blocks) - 1)
            
            # Find subsets that would become uncovered
            m_i = current_masks[i]
            uncovered_bits = []
            mm = m_i
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    uncovered_bits.append(bit_idx)
                mm ^= lsb
                
            if not uncovered_bits:
                # It was redundant, remove it
                remove_block(i)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                max_no_improve = max(max_no_improve_start, max_no_improve // 2)
                found_swap = True
                break
            elif len(uncovered_bits) <= 3:
                # Find a block that covers all uncovered_bits
                uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                
                candidates = blocks_containing_subset[uncovered_bits[0]]
                
                for b in uncovered_bits[1:]:
                    if not candidates:
                        break
                    set_cands = set(candidates)
                    set_b = set(blocks_containing_subset[b])
                    candidates = list(set_cands.intersection(set_b))
                    
                if candidates:
                    valid_candidates = [c for c in candidates if c != i]
                    if valid_candidates:
                        j = random.choice(valid_candidates)
                        # Check if adding j and removing i is beneficial or neutral
                        # We'll do it and then prune
                        remove_block(i)
                        add_block(j)
                        found_swap = True
                        no_improve_count = 0
                        break
        
        if found_swap:
            continue
            
        # Phase 3: Try 1-for-2 swap (remove one, add two)
        # This can help when a single block can't cover all unique bits
        if time.time() > deadline - 30:
            break
            
        i = random.randint(0, len(current_blocks) - 1)
        m_i = current_masks[i]
        uncovered_bits = []
        mm = m_i
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                uncovered_bits.append(bit_idx)
            mm ^= lsb
            
        if uncovered_bits and len(uncovered_bits) <= 5:
            # Try to find two blocks that together cover all uncovered_bits
            # Pick a random subset of uncovered_bits for block1, rest for block2
            if len(uncovered_bits) >= 2:
                split = random.randint(1, len(uncovered_bits) - 1)
                bits1 = uncovered_bits[:split]
                bits2 = uncovered_bits[split:]
                
                # Find blocks covering bits1
                cands1 = blocks_containing_subset[bits1[0]]
                for b in bits1[1:]:
                    if not cands1:
                        break
                    cands1 = [c for c in cands1 if c in set(blocks_containing_subset[b])]
                
                cands2 = blocks_containing_subset[bits2[0]]
                for b in bits2[1:]:
                    if not cands2:
                        break
                    cands2 = [c for c in cands2 if c in set(blocks_containing_subset[b])]
                
                if cands1 and cands2:
                    # Pick random candidates, ensure they're not i
                    c1 = random.choice([c for c in cands1 if c != i] or cands1)
                    c2 = random.choice([c for c in cands2 if c != i and c != c1] or cands2)
                    if c1 != i and c2 != i and c1 != c2:
                        remove_block(i)
                        add_block(c1)
                        add_block(c2)
                        no_improve_count = 0
                        continue
        
        # Phase 4: Perturbation
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            # Perturbation: remove several random blocks and re-add via greedy
            n_remove = random.randint(3, 8)
            to_remove = random.sample(range(len(current_blocks)), min(n_remove, len(current_blocks)))
            to_remove.sort(reverse=True)
            for i in to_remove:
                remove_block(i)
            
            # Re-add blocks greedily to cover uncovered subsets
            if time.time() < deadline - 10:
                # Compute what's uncovered
                temp_counts = counts[:]
                # Actually counts already reflects current state
                # Find uncovered bits
                uncov = [b for b in range(N_SUBSETS) if counts[b] == 0]
                if uncov:
                    # Greedily add blocks to cover uncovered
                    uncov_set = set(uncov)
                    while uncov_set and time.time() < deadline - 10:
                        best_idx = -1
                        best_gain = 0
                        for idx in range(N_BLOCKS):
                            if idx in current_blocks:
                                continue
                            gain = 0
                            m = all_masks[idx]
                            mm = m
                            while mm:
                                lsb = mm & -mm
                                bit_idx = lsb.bit_length() - 1
                                if bit_idx in uncov_set:
                                    gain += 1
                                mm ^= lsb
                            if gain > best_gain:
                                best_gain = gain
                                best_idx = idx
                        if best_idx == -1 or best_gain == 0:
                            break
                        add_block(best_idx)
                        # Update uncov_set
                        m = all_masks[best_idx]
                        mm = m
                        while mm:
                            lsb = mm & -mm
                            bit_idx = lsb.bit_length() - 1
                            uncov_set.discard(bit_idx)
                            mm ^= lsb
            
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
