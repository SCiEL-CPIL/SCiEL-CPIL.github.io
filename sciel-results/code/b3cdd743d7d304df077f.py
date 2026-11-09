import time
import random
import itertools

# Precomputation for v=19, k=7, t=4
V = 19
K = 7
T = 4

def get_4subsets():
    return list(itertools.combinations(range(V), T))

SUBSETS_4 = get_4subsets()
NUM_SUBSETS_4 = len(SUBSETS_4)
SUBSET_TO_IDX = {s: i for i, s in enumerate(SUBSETS_4)}

BLOCKS_7 = list(itertools.combinations(range(V), K))
NUM_BLOCKS_7 = len(BLOCKS_7)

# Precompute masks and subset indices for each block
BLOCK_MASKS = [0] * NUM_BLOCKS_7
BLOCK_SUBSET_IDXS = [[] for _ in range(NUM_BLOCKS_7)]

for j, b in enumerate(BLOCKS_7):
    mask = 0
    idxs = []
    for sub4 in itertools.combinations(b, T):
        idx = SUBSET_TO_IDX[sub4]
        mask |= (1 << idx)
        idxs.append(idx)
    BLOCK_MASKS[j] = mask
    BLOCK_SUBSET_IDXS[j] = idxs

# For each 4-subset, list of 7-block indices containing it
SUBSET_TO_BLOCKS = [[] for _ in range(NUM_SUBSETS_4)]
for j, b in enumerate(BLOCKS_7):
    for idx in BLOCK_SUBSET_IDXS[j]:
        SUBSET_TO_BLOCKS[idx].append(j)

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0
    
    best_blocks = None
    best_count = float('inf')
    
    # If parameters don't match precomputed, we'd need to recompute.
    # Assuming 19, 7, 4.
    if v != 19 or k != 7 or t != 4:
        # Fallback to simple greedy for other params (though task is specific)
        # This part is likely never executed.
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    while time.time() - start_time < timeout:
        uncovered_bits = (1 << NUM_SUBSETS_4) - 1
        blocks_used = []
        counts = [0] * NUM_SUBSETS_4
        
        while uncovered_bits:
            lowest_bit = uncovered_bits & -uncovered_bits
            s_idx = lowest_bit.bit_length() - 1
            
            candidates = SUBSET_TO_BLOCKS[s_idx]
            # Shuffle a copy to avoid modifying global list
            # and to randomize tie-breaking
            candidates_copy = candidates[:]
            random.shuffle(candidates_copy)
            
            best_block_idx = -1
            best_gain = -1
            best_mask = 0
            
            for b_idx in candidates_copy:
                mask = BLOCK_MASKS[b_idx]
                gain = (mask & uncovered_bits).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    best_mask = mask
            
            blocks_used.append(best_block_idx)
            uncovered_bits &= ~best_mask
            
            # Update counts
            for idx in BLOCK_SUBSET_IDXS[best_block_idx]:
                counts[idx] += 1
        
        # Reduction Phase
        changed = True
        while changed:
            changed = False
            to_remove = []
            for i, b_idx in enumerate(blocks_used):
                is_redundant = True
                for idx in BLOCK_SUBSET_IDXS[b_idx]:
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                changed = True
                for i in sorted(to_remove, reverse=True):
                    b_idx = blocks_used[i]
                    blocks_used.pop(i)
                    for idx in BLOCK_SUBSET_IDXS[b_idx]:
                        counts[idx] -= 1
        
        if len(blocks_used) < best_count:
            best_count = len(blocks_used)
            best_blocks = [BLOCKS_7[i] for i in blocks_used]
            
    if best_blocks is None:
        return []
        
    return best_blocks
