import itertools
import random
import time
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets
all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

# Precompute masks as Python ints for fast bit_count
# Also precompute list of subsets for each block (for fast gain calculation in greedy)
# And list of blocks containing each subset (for swap search)
all_masks = [0] * N_BLOCKS
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for bi, block in enumerate(all_blocks):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(bi)
    all_masks[bi] = mask

# Precompute gain for initial greedy: number of subsets covered by each block
# This is just the popcount of the mask, which is constant for all blocks since all blocks have same size
# But during greedy, we need gain relative to uncovered set.

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
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
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # State: use numpy for counts, but keep block list as list of ints
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    current_blocks = []
    current_masks = []
    
    # For fast redundancy check, we need to know which subsets have count==1
    # We'll maintain this implicitly by scanning the mask when needed

    def add_block(idx):
        current_blocks.append(idx)
        m = all_masks[idx]
        current_masks.append(m)
        # Update counts: find set bits in m
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
        # If i is the last element, just pop
        if i == len(current_blocks) - 1:
            current_blocks.pop()
            current_masks.pop()
        else:
            current_blocks[i] = current_blocks[-1]
            current_masks[i] = current_masks[-1]
            current_blocks.pop()
            current_masks.pop()

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

    # 1. Greedy Initialization
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Full scan with vectorized gain calculation would be ideal, but let's stick to Python loop for now
        # Optimization: precompute for each block, which subsets it covers. 
        # Actually, we can precompute a list of lists: for each block, the list of subset indices it covers.
        # Then gain = number of those indices that are in uncovered_mask.
        # But checking "in uncovered_mask" requires bit operations.
        # Alternative: use numpy bit_count on the intersection.
        # Let's precompute the masks as numpy arrays? No, they are big ints.
        
        # Let's try a different approach for greedy:
        # Precompute for each subset, the list of blocks covering it.
        # Then, for each uncovered subset, look at all blocks covering it and increment a score.
        # This is the "reverse" greedy. It might be faster if the number of uncovered subsets is small initially? No, it starts large.
        
        # Let's stick to the forward greedy but optimize the inner loop.
        # We can precompute the list of subset indices for each block.
        # block_subset_indices[bi] = list of subset indices covered by block bi
        # Then gain = sum(1 for idx in block_subset_indices[bi] if (uncovered_mask >> idx) & 1)
        # This is still slow in Python.
        
        # Let's use the precomputed masks and bit_count on the intersection.
        # gain = (all_masks[i] & uncovered_mask).bit_count()
        # This is a single big-int AND and bit_count. This is likely fast in Python 3.10+.
        
        for i in range(N_BLOCKS):
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Early break if gain is maximal (number of subsets in a block = C(7,4)=35)
                if best_gain == 35:
                    break
        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with Perturbation
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Phase 1: Aggressive Pruning
        removed_any = False
        # Iterate in random order to avoid bias
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(current_blocks):
                continue
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                # After removal, indices are invalid, restart pruning
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # Phase 2: Swap Search
        if len(current_blocks) == 0:
            break
            
        # Pick a random block to remove
        i = random.randint(0, len(current_blocks) - 1)
        
        # Find subsets that would become uncovered if we remove block i
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
            continue
            
        # If too many bits to cover, this swap is likely hard.
        # Maximum possible is 35.
        if len(uncovered_bits) > 5:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Remove a random block and add a random different one?
                # Or just add a random block to create new redundancy opportunities?
                # Let's try: Add a random block. This increases size, but might enable pruning.
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve_count = 0
                    max_no_improve = 20 # Reset to allow quick re-perturbation if it fails
                continue
        else:
            # Find a block that covers all uncovered_bits
            # Sort by frequency to speed up intersection
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            
            # Start with the smallest list
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            # Intersect with others
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                # Optimization: If the list is small, use set intersection
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                # Pick a random candidate
                # Exclude i if it's in candidates
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    j = random.choice(valid_candidates)
                    # Perform swap
                    remove_block(i)
                    add_block(j)
                    no_improve_count = 0
                    continue
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturbation
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_blocks:
                        add_block(c)
                        no_improve_count = 0
                        max_no_improve = 20
                    continue

    # Final cleanup: ensure we return the best found
    return [list(all_blocks[i]) for i in best_blocks]
