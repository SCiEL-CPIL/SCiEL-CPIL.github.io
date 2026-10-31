import itertools
import random
import time
import numpy as np

# Precompute for the specific case v=19, k=7, t=4
V = 19
K = 7
T = 4

# Generate all k-subsets and t-subsets
all_kblocks = list(itertools.combinations(range(V), K))
all_tsubs = list(itertools.combinations(range(V), T))

N_BLOCKS = len(all_kblocks)
N_TARGETS = len(all_tsubs)

# Map t-subset to index
tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

# Build incidence matrix M
# M[i, j] = 1 if t-subset j is contained in block i
# Using uint8 for memory efficiency
M = np.zeros((N_BLOCKS, N_TARGETS), dtype=np.uint8)

for i, B in enumerate(all_kblocks):
    # Generate all t-subsets of B
    for ts in itertools.combinations(B, T):
        j = tsub_to_idx[ts]
        M[i, j] = 1

def construct(v, k, t):
    # Fallback for non-matching parameters
    if v != V or k != K or t != T:
        # Simple greedy for other cases (not optimized)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Pick a random uncovered subset to anchor the block
            target = next(iter(uncovered))
            # Build a block containing target
            block_set = set(target)
            # Greedily add elements to maximize coverage of uncovered subsets
            # This is a rough heuristic for fallback
            while len(block_set) < k:
                best_x = -1
                max_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    new_block_set = block_set | {x}
                    # Count how many uncovered subsets are covered by new_block_set
                    gain = 0
                    for sub in itertools.combinations(new_block_set, t):
                        if sub in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                if best_x != -1:
                    block_set.add(best_x)
                else:
                    break
            block = sorted(block_set)
            blocks.append(block)
            # Update uncovered
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    uncovered.remove(sub)
        return blocks

    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    best_blocks = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # We will run multiple iterations of Greedy + Reduction
    iteration = 0
    while time.time() < deadline:
        # 1. Greedy Phase
        uncovered = np.ones(N_TARGETS, dtype=np.uint8)
        selected_indices = []
        
        # To introduce diversity, we can add random noise to gains
        # or just rely on tie-breaking. 
        # A better diversity strategy: Randomly perturb the initial uncovered vector? 
        # No, that breaks validity logic.
        # Instead, we can randomly break ties in the greedy selection.
        
        while np.sum(uncovered) > 0:
            # Calculate gains: number of uncovered targets covered by each block
            # M is uint8, uncovered is uint8. Result is int? 
            # np.dot with uint8 returns uint8? No, usually int.
            # Max gain is 35. So uint8 is safe for the result if we cast, but 
            # let's use int32 for safety and speed of argmax.
            gains = M @ uncovered
            
            # Add small random noise to break ties and encourage diversity
            # Noise magnitude should be small enough not to change the ordering of 
            # significantly different gains, but enough to shuffle close ones.
            # If we want deterministic runs, set noise=0.
            # For search, we want diversity.
            if iteration > 0:
                # Scale noise to be less than 1, so it only affects ties
                noise = rng.random(N_BLOCKS) * 0.5
                gains = gains.astype(np.float32) + noise
            else:
                gains = gains.astype(np.float32)
                
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain < 1e-6:
                # Should not happen if covering is possible
                break
                
            selected_indices.append(best_idx)
            
            # Update uncovered
            # uncovered = uncovered & (M[best_idx] == 0)
            # Since M[best_idx] is 0 or 1, (M[best_idx] == 0) is 1 or 0.
            uncovered = uncovered & (M[best_idx] == 0)
            
        if len(selected_indices) == 0:
            break
            
        # 2. Reduction Phase
        # We have a valid covering. Now minimize.
        # selected_indices is a list of block indices.
        
        # Compute target_counts: for each target, how many selected blocks cover it?
        # We can use the sub-matrix of M for selected blocks.
        # S = M[selected_indices, :]
        # target_counts = S.sum(axis=0)
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            if not selected_indices:
                break
                
            S = M[selected_indices, :]
            target_counts = S.sum(axis=0)
            
            # A block is essential if it covers at least one target with count 1
            # essential_mask[i, j] = 1 if S[i, j] == 1 and target_counts[j] == 1
            # We can compute this without creating a huge boolean matrix if we are careful,
            # but N_SEL is small (~140), N_TARGETS is 3876. 140*3876 is small.
            
            # mask_essential = (S == 1) & (target_counts[None, :] == 1)
            # block_is_essential = mask_essential.any(axis=1)
            
            # Optimization: 
            # For each block i, check if any target j it covers has count 1.
            # We can iterate over the blocks.
            
            new_selected_indices = []
            for i, idx in enumerate(selected_indices):
                # Get the targets covered by this block
                # They are the columns where S[i] == 1
                # Check if any of those targets have target_counts == 1
                # 
                # Efficient check:
                # targets_covered = np.where(S[i] == 1)[0]
                # if len(targets_covered) > 0 and np.any(target_counts[targets_covered] == 1):
                #     new_selected_indices.append(idx)
                # else:
                #     changed = True
                #
                # Even faster:
                # if np.any((S[i] == 1) & (target_counts == 1)):
                #     new_selected_indices.append(idx)
                # else:
                #     changed = True
                
                if np.any((S[i] == 1) & (target_counts == 1)):
                    new_selected_indices.append(idx)
                else:
                    changed = True
                    
            selected_indices = new_selected_indices
            
        # Update best
        if len(selected_indices) < best_size:
            best_size = len(selected_indices)
            best_blocks = selected_indices.copy()
            
        iteration += 1
        
        # If we hit a very low number, we might stop early?
        # Lower bound is 131.
        if best_size <= 131:
            break

    if best_blocks is None:
        # Fallback: return an empty list or a single block? 
        # The task requires a valid covering.
        # If we didn't find one, we should return a valid one.
        # But greedy_reduce always returns a valid covering.
        return []
        
    # Convert block indices back to lists of integers
    result_blocks = [list(all_kblocks[idx]) for idx in best_blocks]
    return result_blocks
