"""
Combines the fast greedy structure of the current program with a local search 
polishing phase that attempts block swaps to enable further redundancy removal.
"""
import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We target finishing well before.
    # Precomputation takes ~2-5s. Greedy takes ~1-2s. Polish takes the rest.
    deadline = start_time + 135.0 

    # 1. Precompute all t-subsets and map to indices
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Build incidence structures
    # block_to_subsets[b] = list of subset indices covered by block b
    block_to_subsets = [[] for _ in range(num_blocks)]
    # subset_to_blocks[u] = list of block indices covering subset u
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access
    # We keep block_to_subsets as lists of lists for easy iteration, 
    # but we will use numpy for the gain vector.
    
    # 4. Initialize state
    # uncovered[u] = True if subset u is not yet covered
    uncovered = np.ones(num_subsets, dtype=bool)
    # gains[b] = number of currently uncovered subsets covered by block b
    gains = np.zeros(num_blocks, dtype=np.int32)
    
    # Initial gains
    for u in range(num_subsets):
        for b in subset_to_blocks[u]:
            gains[b] += 1
            
    blocks_found = []
    used_blocks = set()
    
    # 5. Greedy Set Cover
    while True:
        if time.time() > deadline - 20.0:  # Reserve 20s for polishing
            break
            
        if not uncovered.any():
            break
            
        # Find block with max gain
        b_star = int(np.argmax(gains))
        
        if gains[b_star] <= 0:
            break
            
        # Add this block to solution
        blocks_found.append(list(all_blocks[b_star]))
        used_blocks.add(b_star)
        
        # Update uncovered and gains
        covered_subsets = block_to_subsets[b_star]
        
        for u in covered_subsets:
            if uncovered[u]:
                uncovered[u] = False
                # Decrement gain for all blocks that also cover this subset u
                for b in subset_to_blocks[u]:
                    gains[b] -= 1
                    
        if not uncovered.any():
            break

    # 6. Polish Phase: Local Search
    # Strategy:
    # 1. Redundancy Removal: Remove any block that is not essential.
    # 2. Swap Search: If a block is essential, try to swap it with another block 
    #    that covers the same "critical" subsets, hoping to make other blocks redundant.
    #    Specifically, if removing block B_i leaves uncovered set U, 
    #    look for a block B_j (not in solution) that covers all of U.
    #    If found, swap B_i with B_j. This keeps count same but changes structure.
    #    Then re-run redundancy removal.
    
    # Helper function to check if a specific list of blocks covers all subsets
    def is_valid_cover(block_list):
        if not block_list:
            return len(all_subsets) == 0
        # Fast check: count covered subsets
        # We use a set of indices for speed
        covered_idx = set()
        for b in block_list:
            covered_idx.update(block_to_subsets[b])
        return len(covered_idx) == num_subsets

    # Helper to get uncovered subsets for a given list of blocks
    def get_uncovered(block_list):
        covered_idx = set()
        for b in block_list:
            covered_idx.update(block_to_subsets[b])
        return [u for u in range(num_subsets) if u not in covered_idx]

    # We work with indices of blocks in all_blocks for efficiency
    # blocks_found_idx is the list of indices into all_blocks
    blocks_found_idx = [all_blocks.index(b) for b in blocks_found]
    
    # Redundancy Removal Pass 1
    # Iterate multiple times as removal of one block can make another redundant
    max_redundancy_passes = 20
    for _ in range(max_redundancy_passes):
        if time.time() > deadline - 5.0:
            break
        removed_any = False
        # Iterate backwards to allow popping
        for i in range(len(blocks_found_idx) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            b_idx = blocks_found_idx[i]
            # Check if b_idx is redundant
            # A block is redundant if every subset it covers is covered by at least one other block
            is_redundant = True
            for u in block_to_subsets[b_idx]:
                # Check if any OTHER block in blocks_found_idx covers u
                # Optimization: Precompute cover counts? 
                # For small N, simple check is okay.
                # But we can be smarter: 
                # If we maintain a count of how many times each subset is covered, 
                # we can check if count[u] > 1.
                # Let's maintain cover_counts.
                pass
                
            # Let's implement a more efficient redundancy check using cover counts
            # Rebuild cover counts for current blocks_found_idx
            # This is O(N_blocks_in_sol * C(k,t)). 
            # With ~76 blocks and 35 subsets per block, it's ~2660 ops.
            # Doing this for each block in the list is O(N^2 * C(k,t)) ~ 76*2660 ~ 200k ops.
            # This is fast enough.
            
            # To avoid rebuilding from scratch every time, we could update incrementally, 
            # but for simplicity and robustness, we rebuild if needed or just do the check.
            
            # Efficient Check:
            # For each subset u in block b_idx, is there another block b' in blocks_found_idx (b' != b_idx) 
            # such that u is in block_to_subsets[b']?
            
            # We can precompute a set of covered subsets for the current solution?
            # No, we need to know if it's covered by *another* block.
            
            # Let's just do the direct check.
            for u in block_to_subsets[b_idx]:
                covered_by_other = False
                for j in blocks_found_idx:
                    if j == b_idx:
                        continue
                    # Check if u is in block_to_subsets[j]
                    # block_to_subsets[j] is a list. Membership is O(L).
                    # We can convert block_to_subsets to a list of sets for O(1) lookup.
                    if u in block_to_subsets[j]: 
                         # Wait, block_to_subsets[j] is a list of ints.
                         # We should convert to sets for faster lookup in this polish phase.
                         pass
                
            # Let's convert block_to_subsets to sets for the polish phase to speed up checks.
            # This is a structural change in representation for the polish phase.
            
            # Actually, let's just use the "cover count" method which is standard.
            # 1. Compute cover_count[u] for all u based on current blocks_found_idx.
            # 2. For block b_idx, it is redundant if for all u in block_to_subsets[b_idx], cover_count[u] > 1.
            
            # Let's implement this efficiently.
            
            if is_redundant: # Placeholder, logic below
                pass

    # Let's restructure the polish phase properly.
    
    # Convert block_to_subsets to a list of sets for O(1) membership checks
    block_subsets_sets = [set(bs) for bs in block_to_subsets]
    
    def compute_cover_counts(block_indices):
        counts = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in block_indices:
            for u in block_to_subsets[b_idx]:
                counts[u] += 1
        return counts

    # Pass 1: Pure Redundancy Removal
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        counts = compute_cover_counts(blocks_found_idx)
        for i in range(len(blocks_found_idx) - 1, -1, -1):
            b_idx = blocks_found_idx[i]
            is_redundant = True
            for u in block_to_subsets[b_idx]:
                if counts[u] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                blocks_found_idx.pop(i)
                changed = True
                break # Restart the loop with new counts

    # Pass 2: Swap-based Improvement
    # Try to swap blocks to enable redundancy.
    # For each block B_i in solution:
    #   1. Remove B_i.
    #   2. Find uncovered subsets U.
    #   3. If U is empty, we already handled this in Pass 1.
    #   4. If U is not empty, look for a block B_j (not in solution) that covers ALL of U.
    #   5. If found, swap B_i with B_j.
    #   6. Run Redundancy Removal again.
    
    # To speed up step 4, we can precompute for every subset u, the list of blocks covering it.
    # But we need a block that covers a *set* of subsets U.
    # This is an intersection problem.
    # Candidate blocks for U are the intersection of subset_to_blocks[u] for all u in U.
    # Since U is small (usually), this is feasible.
    
    max_swap_passes = 50
    for _ in range(max_swap_passes):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        # Shuffle blocks to avoid bias? Or iterate in order.
        # Let's iterate in order.
        for i in range(len(blocks_found_idx)):
            if time.time() > deadline - 5.0:
                break
            
            b_old = blocks_found_idx[i]
            # Temporarily remove b_old
            # Find uncovered subsets U
            # We can compute U by checking which subsets in b_old have count 1 (if we had counts)
            # But counts change if we remove b_old.
            # Let's just compute the covered set of the remaining blocks.
            remaining = blocks_found_idx[:i] + blocks_found_idx[i+1:]
            
            # Fast check: what is uncovered?
            # We can use the fact that only subsets in b_old *might* become uncovered.
            # If a subset u in b_old was covered by another block in remaining, it's still covered.
            # So U is subset of block_to_subsets[b_old].
            
            U = []
            for u in block_to_subsets[b_old]:
                # Check if u is covered by any block in remaining
                covered = False
                for b_rem in remaining:
                    if u in block_subsets_sets[b_rem]:
                        covered = True
                        break
                if not covered:
                    U.append(u)
            
            if not U:
                # This block is redundant! (Should have been caught by Pass 1, but maybe not if Pass 1 stopped early)
                blocks_found_idx.pop(i)
                improved = True
                break # Restart with new state
            
            # If U is not empty, try to find a replacement block
            # A replacement block must cover all u in U.
            # Candidate blocks: Intersection of subset_to_blocks[u] for u in U
            # Exclude blocks already in remaining (and b_old itself, though it's removed)
            
            # Get candidates from the first u in U
            candidates = set(subset_to_blocks[U[0]])
            # Intersect with others
            for u in U[1:]:
                candidates.intersection_update(subset_to_blocks[u])
            
            # Filter out blocks already in the solution
            current_solution_set = set(blocks_found_idx)
            candidates.difference_update(current_solution_set)
            
            # If any candidate exists, pick one (e.g., the one with highest gain or just the first)
            # To make it deterministic, pick the one with the smallest index or just the first.
            # Actually, picking the one that covers the most *total* uncovered (which is U) is good, 
            # but they all cover U. 
            # Maybe pick the one that covers the most *other* subsets to be "useful"?
            # For now, just pick the first valid candidate.
            if candidates:
                # Pick a candidate
                b_new = min(candidates) # Deterministic
                blocks_found_idx[i] = b_new
                improved = True
                # Don't break immediately, but re-run redundancy removal might be needed.
                # Let's break to re-run the outer loop which will do redundancy removal.
                break
        
        if not improved:
            break

    # Pass 3: Final Redundancy Removal
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        counts = compute_cover_counts(blocks_found_idx)
        for i in range(len(blocks_found_idx) - 1, -1, -1):
            b_idx = blocks_found_idx[i]
            is_redundant = True
            for u in block_to_subsets[b_idx]:
                if counts[u] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                blocks_found_idx.pop(i)
                changed = True
                break

    # Convert indices back to blocks
    final_blocks = [list(all_blocks[idx]) for idx in blocks_found_idx]
    
    # Final validity check
    # Ensure all subsets are covered
    if time.time() < deadline - 1.0:
        covered_set = set()
        for b in final_blocks:
            covered_set.update(block_to_subsets[all_blocks.index(b)]) # This is slow. 
            # Better:
            # We already have block_to_subsets indexed by block index.
            # Let's just use the indices.
        
        # Re-check using indices
        covered_idx = set()
        for b_idx in blocks_found_idx:
            covered_idx.update(block_to_subsets[b_idx])
            
        if len(covered_idx) != num_subsets:
            # Repair: Greedily add blocks for missing subsets
            missing = [u for u in range(num_subsets) if u not in covered_idx]
            # Add blocks to cover missing
            while missing and time.time() < deadline - 0.5:
                # Pick a missing subset
                u = missing[0]
                # Find best block covering u
                best_b = -1
                best_gain = -1
                for b in subset_to_blocks[u]:
                    # Count how many missing it covers
                    gain = 0
                    for u2 in block_to_subsets[b]:
                        if u2 in set(missing):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
                if best_b != -1:
                    blocks_found_idx.append(best_b)
                    # Update missing
                    for u2 in block_to_subsets[best_b]:
                        if u2 in set(missing):
                            missing.remove(u2)
                else:
                    break
                    
            # Convert again
            final_blocks = [list(all_blocks[idx]) for idx in blocks_found_idx]

    # Deduplicate
    seen = set()
    unique_blocks = []
    for b in final_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
            
    return unique_blocks

# EVOLVE-BLOCK-END
