import itertools
import time
import random

# Precomputation
v, k, t = 19, 7, 4
# Map 4-subsets to indices
subsets_4 = list(itertools.combinations(range(v), t))
subset_to_idx = {s: i for i, s in enumerate(subsets_4)}
N_subsets = len(subsets_4)

# Generate all 7-subsets and their masks
blocks_7 = list(itertools.combinations(range(v), k))
block_masks = []
block_subsets = [] # Store the actual 4-subsets indices for fast count update
for b in blocks_7:
    mask = 0
    idxs = []
    for s in itertools.combinations(b, t):
        idx = subset_to_idx[s]
        mask |= (1 << idx)
        idxs.append(idx)
    block_masks.append(mask)
    block_subsets.append(idxs)

# Greedy with Randomization
def solve():
    deadline = time.time() + 110
    best_blocks = None
    best_len = float('inf')
    
    # Initial full mask
    full_mask = (1 << N_subsets) - 1
    
    while time.time() < deadline:
        uncovered = full_mask
        used_blocks = []
        cover_counts = [0] * N_subsets
        
        while uncovered:
            best_gain = -1
            candidates = []
            # Iterate all blocks
            for i, mask in enumerate(block_masks):
                gain = (uncovered & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                # Should not happen if uncovered is not 0
                break
            
            # Pick randomly from candidates to diversify
            idx = random.choice(candidates)
            used_blocks.append(idx)
            
            # Update uncovered
            uncovered &= ~block_masks[idx]
            
            # Update cover counts
            for s_idx in block_subsets[idx]:
                cover_counts[s_idx] += 1
        
        # Redundancy Removal
        # We need to check if a block is redundant.
        # A block is redundant if all its subsets have count > 1.
        # We iterate until no more removals.
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow easy removal? Or just mark.
            # Let's just filter the list.
            new_blocks = []
            for idx in used_blocks:
                is_redundant = True
                for s_idx in block_subsets[idx]:
                    if cover_counts[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove it
                    for s_idx in block_subsets[idx]:
                        cover_counts[s_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(idx)
            used_blocks = new_blocks
        
        if len(used_blocks) < best_len:
            best_len = len(used_blocks)
            best_blocks = [blocks_7[i] for i in used_blocks]
            # Optional: If we hit the known best (151), we can stop? 
            # No, we want to beat it. But if we hit 131 (lower bound), we can stop.
            if best_len <= 131:
                break

    return best_blocks

def construct(v, k, t):
    # The precomputation above assumes v=19, k=7, t=4.
    # The task says "evaluator passes this task's parameters v, k and t".
    # However, the task description specifically fixes C(19,7,4).
    # "The evaluator passes this task's parameters v, k and t."
    # If v,k,t are different, my hardcoded precomputation fails.
    # I must make it generic or check if it's the specific case.
    # Given "Covering design C(19,7,4)" is the title, it's highly likely v=19, k=7, t=4.
    # But to be safe, I should handle generic v,k,t if possible, or at least not crash.
    # However, generic implementation with bitsets for larger v might be slow/memory heavy.
    # For v=19, it's fine.
    # If v is larger, e.g., 20, N_subsets = 4845. 2^4845 is huge but Python handles big ints.
    # Number of blocks C(20,7) = 77520.
    # It scales reasonably for small v.
    
    if v == 19 and k == 7 and t == 4:
        return solve()
    else:
        # Fallback to simple greedy for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Find block covering most uncovered
            best_block = None
            max_covered = 0
            # This is slow for generic, but v is likely small in other tests?
            # Or just use the seed logic?
            # Let's use a slightly better generic greedy
            for b in itertools.combinations(range(v), k):
                covered = 0
                for s in itertools.combinations(b, t):
                    if s in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_block = b
            if not best_block:
                break
            blocks.append(list(best_block))
            for s in itertools.combinations(best_block, t):
                uncovered.discard(s)
        return blocks
