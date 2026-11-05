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

# Generate all 7-subsets and their covered subset indices
all_blocks = []
block_to_subsets = []
subset_to_blocks = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    block_idx = len(all_blocks)
    all_blocks.append(block)
    covered_indices = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        covered_indices.append(idx)
        subset_to_blocks[idx].append(block_idx)
    block_to_subsets.append(covered_indices)

N_BLOCKS = len(all_blocks)

# Weights for greedy: rare subsets are more important
subset_frequencies = [len(lst) for lst in subset_to_blocks]
weights = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)

# Coverage matrix: N_BLOCKS x N_SUBSETS
# Using sparse representation for speed in greedy, but dense for dot products if needed?
# N_BLOCKS = C(19,7) = 50388. N_SUBSETS = C(19,4) = 3876.
# Dense matrix: 50388 * 3876 ~ 195M entries. 8 bytes each ~ 1.5GB. Too big for memory if float64.
# Use int8: ~195MB. Acceptable.
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, indices in enumerate(block_to_subsets):
    for j in indices:
        coverage_matrix[i, j] = 1

# Precompute list of subset indices for each block for faster iteration in local search
# block_to_subsets is already a list of lists.

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
    deadline = start_time + 115  # Safety margin

    def run_greedy():
        """Fast vectorized weighted greedy."""
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        
        while True:
            if not uncovered.any():
                break
            
            # Calculate gain: sum of weights of uncovered subsets covered by block
            # gains = coverage_matrix @ (weights * uncovered)
            effective_weights = weights * uncovered
            gains = coverage_matrix.dot(effective_weights)
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
                
            chosen.append(best_idx)
            uncovered[coverage_matrix[best_idx] == 1] = 0
            
            if time.time() > deadline - 10:
                break

        return chosen

    def prune_redundant(blocks_list, counts):
        """
        Remove blocks that are redundant.
        Returns new_blocks, new_counts, and number of removals.
        """
        if not blocks_list:
            return blocks_list, counts, 0
        
        new_blocks = []
        removals = 0
        # Iterate in reverse to allow popping from end? Or just build new list.
        # We need to update counts as we go? 
        # Redundancy check: block i is redundant if for all s in block_to_subsets[i], counts[s] > 1.
        # If we remove block i, counts[s] decreases for s in block_to_subsets[i].
        # This might make other blocks redundant.
        
        # To be efficient, we can just check against current counts.
        # If we remove a block, it only *increases* the chance of others being redundant (counts go down).
        # So a single pass is not enough. We need to loop until no more removals.
        
        # However, doing a full loop every time is slow.
        # Let's do a single pass first. If removals > 0, we return and the main loop will call prune again.
        
        for b_idx in blocks_list:
            redundant = True
            for s_idx in block_to_subsets[b_idx]:
                if counts[s_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for s_idx in block_to_subsets[b_idx]:
                    counts[s_idx] -= 1
                removals += 1
            else:
                new_blocks.append(b_idx)
                
        return new_blocks, counts, removals

    def local_search(initial_blocks):
        """
        Local search using coverage counts.
        """
        current_blocks = list(initial_blocks)
        current_blocks = list(set(current_blocks))
        
        # Initialize counts
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in current_blocks:
            counts[block_to_subsets[b_idx]] += 1
            
        best_blocks = current_blocks.copy()
        best_len = len(current_blocks)
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5:
            # 1. Prune redundant blocks
            new_blocks, counts, removals = prune_redundant(current_blocks, counts)
            
            if removals > 0:
                current_blocks = new_blocks
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve = 0
                continue
            
            # If no removals, we are in a minimal covering (or stuck)
            
            # 2. Try to find a 2-for-1 move (remove 2, add 1)
            # This reduces size by 1.
            found_2for1 = False
            if len(current_blocks) >= 2:
                # Sample pairs to avoid O(N^2)
                # We want to find i, k such that removing them leaves deficits D,
                # and there exists j covering D.
                
                # Heuristic: Pick a random block i.
                # Find its "unique" subsets U_i.
                # If we remove i, we need to cover U_i.
                # If we also remove k, we need to cover U_i U U_k (minus overlaps covered by others).
                
                # Simpler approach for 2-for-1:
                # Pick random i.
                # Let U_i = {s | counts[s] == 1 and s in block_to_subsets[i]}
                # If U_i is empty, i is redundant (should have been pruned).
                # So U_i is non-empty.
                # We need a block j that covers U_i.
                # If we just swap i->j, size is same.
                # To reduce size, we need to remove i AND k, and add j.
                # This requires j to cover (U_i U U_k) where U_k are subsets uniquely covered by k.
                # And j must not be i or k.
                
                # Let's try a simpler perturbation first if stuck: 1-for-1 swap to change structure.
                # Or 1-for-2 to increase size and escape.
                
                pass # We will implement 1-for-1 swap and 1-for-2 perturbation below.

            # 3. Try 1-for-1 Swaps
            # Goal: Change the set without changing size, hopefully to create redundancy.
            # For a random block i in current_blocks:
            #   Find U_i (subsets with count 1 covered by i).
            #   Find candidates J covering U_i.
            #   Pick j in J (j != i).
            #   If j is not in current_blocks:
            #       Swap i with j.
            #       Update counts.
            #       Check if this creates any new redundancies? (Next prune pass will handle).
            
            improved = False
            if len(current_blocks) > 0:
                i = random.randint(0, len(current_blocks) - 1)
                b_out = current_blocks[i]
                
                # Find unique subsets
                unique_subsets = []
                for s_idx in block_to_subsets[b_out]:
                    if counts[s_idx] == 1:
                        unique_subsets.append(s_idx)
                
                if unique_subsets:
                    # Find candidates covering all unique_subsets
                    # Intersect subset_to_blocks for each s in unique_subsets
                    cands = None
                    for s_idx in unique_subsets:
                        lst = subset_to_blocks[s_idx]
                        if cands is None:
                            cands = set(lst)
                        else:
                            cands.intersection_update(lst)
                        if not cands:
                            break
                    
                    if cands:
                        cands.discard(b_out)
                        # Filter out blocks already in current set
                        current_set = set(current_blocks)
                        valid_cands = [c for c in cands if c not in current_set]
                        
                        if valid_cands:
                            # Pick one randomly
                            b_in = random.choice(valid_cands)
                            
                            # Perform swap
                            # Remove b_out
                            for s_idx in block_to_subsets[b_out]:
                                counts[s_idx] -= 1
                            # Add b_in
                            for s_idx in block_to_subsets[b_in]:
                                counts[s_idx] += 1
                            
                            # Update block list
                            current_blocks[i] = b_in
                            improved = True
                            no_improve = 0
                            break # Restart prune phase

            if improved:
                continue

            # 4. If no swap found, try 1-for-2 Perturbation (increase size by 1)
            # This helps escape local minima.
            if no_improve > 10:
                i = random.randint(0, len(current_blocks) - 1)
                b_out = current_blocks[i]
                
                unique_subsets = []
                for s_idx in block_to_subsets[b_out]:
                    if counts[s_idx] == 1:
                        unique_subsets.append(s_idx)
                
                if unique_subsets:
                    # We need 2 blocks to cover these.
                    # Just pick 2 random blocks that cover some of them?
                    # Better: Pick a random block j1 covering the first unique subset.
                    # Then pick j2 covering the rest?
                    
                    # Simple random perturbation: Add 1 random block, remove 1 random block?
                    # No, that's a swap.
                    # Let's add 1 random block that covers at least one unique subset.
                    s0 = unique_subsets[0]
                    cands = subset_to_blocks[s0]
                    if cands:
                        b_add = random.choice(cands)
                        if b_add not in current_blocks:
                            # Add b_add
                            for s_idx in block_to_subsets[b_add]:
                                counts[s_idx] += 1
                            current_blocks.append(b_add)
                            no_improve = 0
                            continue

            no_improve += 1
            if no_improve > max_no_improve:
                # Hard perturbation: Restart with a slightly different greedy?
                # Or just add 2 random blocks and prune.
                for _ in range(2):
                    b_rand = random.randint(0, N_BLOCKS - 1)
                    if b_rand not in current_blocks:
                        current_blocks.append(b_rand)
                        for s_idx in block_to_subsets[b_rand]:
                            counts[s_idx] += 1
                no_improve = 0

        return best_blocks

    # 1. Greedy Initialization
    init_blocks = run_greedy()
    
    # 2. Local Search
    best_blocks = local_search(init_blocks)
    
    # Final check/prune
    # Rebuild counts for best_blocks to be sure
    counts_final = np.zeros(N_SUBSETS, dtype=np.int32)
    for b_idx in best_blocks:
        counts_final[block_to_subsets[b_idx]] += 1
        
    # One last prune pass
    final_blocks, _, _ = prune_redundant(best_blocks, counts_final)
    
    return [list(all_blocks[i]) for i in final_blocks]
