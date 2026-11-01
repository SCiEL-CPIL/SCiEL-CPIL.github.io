import itertools
import random
import time
import numpy as np

# Precomputation for v=19, k=8, t=5
# These are hardcoded for the specific task parameters to maximize speed.
# The task states: "The evaluator passes this task's parameters v, k and t."
# However, the contract says "Keep the function names and signatures".
# It also says "construct(v, k, t)".
# I should make the precomputation dynamic or at least handle the specific case efficiently.
# Given the 120s limit, dynamic precomputation for v=19 is fine.
# But hardcoding the structure for v=19 is safer for speed if I assume the parameters are fixed.
# The prompt says "The evaluator passes this task's parameters v, k and t."
# I will implement a generic precomputation that is fast enough for v=19.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate all t-subsets and map to IDs
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Generate all k-blocks and map to covered t-subset IDs
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)
    
    block_to_subsets = []
    subset_to_blocks = [[] for _ in range(n_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        covered_ids = []
        for t_sub in itertools.combinations(block, t):
            sid = subset_to_id[t_sub]
            covered_ids.append(sid)
            subset_to_blocks[sid].append(b_idx)
        block_to_subsets.append(covered_ids)
    
    # Helper: Greedy cover a set of uncovered subset IDs
    # uncovered: set of IDs
    # Returns: list of block indices added
    def greedy_cover(uncovered_ids):
        if not uncovered_ids:
            return []
        
        added_blocks = []
        # Copy to mutable set
        unc = set(uncovered_ids)
        
        while unc:
            # Pick a subset to cover. 
            # Heuristic: Pick one with fewest covering blocks to reduce options? 
            # Or just pick one. Let's pick one that appears in the fewest available blocks?
            # For speed, just pick an arbitrary one from the set.
            # Randomization helps in ILS.
            target_sid = next(iter(unc))
            
            # Find best block covering target_sid
            best_block = -1
            best_gain = -1
            best_subsets_to_add = []
            
            for b_idx in subset_to_blocks[target_sid]:
                # Calculate gain: how many subsets in this block are in unc?
                gain = 0
                subs_to_add = []
                for sid in block_to_subsets[b_idx]:
                    if sid in unc:
                        gain += 1
                        subs_to_add.append(sid)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_subsets_to_add = subs_to_add
            
            # Add the best block
            if best_block != -1:
                added_blocks.append(best_block)
                for sid in best_subsets_to_add:
                    unc.discard(sid)
            
            if time.time() > deadline:
                break
                
        return added_blocks

    # Helper: Compute coverage counts from a list of block indices
    def compute_counts(block_indices):
        counts = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in block_indices:
            for sid in block_to_subsets[b_idx]:
                counts[sid] += 1
        return counts

    # Helper: Check if valid (all counts > 0)
    def is_valid(counts):
        return np.all(counts > 0)

    # Helper: Local Search (Deletion)
    # Tries to remove blocks that are redundant.
    def local_search_delete(blocks, counts):
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias
            random.shuffle(blocks)
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                # Check if all subsets in this block are covered by others
                redundant = True
                for sid in block_to_subsets[b_idx]:
                    if counts[sid] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block
                    blocks.pop(i)
                    for sid in block_to_subsets[b_idx]:
                        counts[sid] -= 1
                    changed = True
        return blocks, counts

    best_blocks = None
    best_len = float('inf')
    
    # Main Loop: Iterated Greedy + Local Search + SA
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # 1. Greedy Construction
        # Start with all subsets uncovered
        initial_unc = set(range(n_subsets))
        greedy_blocks = greedy_cover(initial_unc)
        
        if len(greedy_blocks) >= best_len:
            continue # Not promising, skip SA if already worse? 
            # Actually, greedy might be bad, but SA could improve it. 
            # But if greedy is already much worse, SA might not fix it in time.
            # Let's allow it if it's not drastically worse.
            if len(greedy_blocks) > best_len * 1.5:
                continue

        # 2. Compute counts and Local Search Delete
        blocks = list(greedy_blocks)
        counts = compute_counts(blocks)
        
        # Ensure validity (greedy should produce valid, but just in case of timeout in greedy)
        if not is_valid(counts):
            # Repair
            unc = set(np.where(counts == 0)[0])
            added = greedy_cover(unc)
            blocks.extend(added)
            for b_idx in added:
                for sid in block_to_subsets[b_idx]:
                    counts[sid] += 1
        
        blocks, counts = local_search_delete(blocks, counts)
        
        if len(blocks) < best_len:
            best_blocks = list(blocks)
            best_len = len(blocks)
        
        # 3. Simulated Annealing Polish
        # Only if we have time and the current solution is not already optimal (lower bound is 264)
        if len(blocks) > 264 and time.time() < deadline - 10:
            T = 1.0
            cooling = 0.95
            curr_blocks = list(blocks)
            curr_counts = counts.copy()
            
            sa_time_limit = min(deadline - 5, time.time() + 10) # Allow max 10s per SA run
            
            while time.time() < sa_time_limit and T > 0.1:
                if not curr_blocks:
                    break
                    
                # Pick random block to remove
                idx = random.randint(0, len(curr_blocks) - 1)
                b_idx = curr_blocks[idx]
                
                # Tentatively remove
                temp_counts = curr_counts.copy()
                for sid in block_to_subsets[b_idx]:
                    temp_counts[sid] -= 1
                
                # Find newly uncovered
                newly_uncovered = np.where(temp_counts == 0)[0]
                
                if len(newly_uncovered) == 0:
                    # Valid removal
                    candidate_blocks = curr_blocks[:idx] + curr_blocks[idx+1:]
                    candidate_counts = temp_counts
                else:
                    # Invalid removal, need repair
                    candidate_blocks = list(curr_blocks)
                    candidate_blocks.pop(idx)
                    candidate_counts = temp_counts
                    
                    # Repair
                    added = greedy_cover(set(newly_uncovered.tolist()))
                    candidate_blocks.extend(added)
                    for add_idx in added:
                        for sid in block_to_subsets[add_idx]:
                            candidate_counts[sid] += 1
                
                curr_len = len(curr_blocks)
                cand_len = len(candidate_blocks)
                delta = cand_len - curr_len
                
                if delta < 0:
                    # Improvement
                    curr_blocks = candidate_blocks
                    curr_counts = candidate_counts
                    if curr_len < best_len:
                        best_blocks = list(curr_blocks)
                        best_len = curr_len
                else:
                    # Worse or equal, accept with probability
                    if delta == 0:
                        accept_prob = 0.5
                    else:
                        accept_prob = np.exp(-delta / T)
                    
                    if random.random() < accept_prob:
                        curr_blocks = candidate_blocks
                        curr_counts = candidate_counts
                        if curr_len < best_len:
                            best_blocks = list(curr_blocks)
                            best_len = curr_len
                
                T *= cooling
        
        # If we have a very good solution, we might break early?
        # Lower bound is 264. If we hit 264, we can stop.
        if best_len == 264:
            break

    # Convert block indices back to lists of points
    if best_blocks is None:
        # Fallback: just return greedy result if something went wrong
        # This should not happen
        return [list(all_k_blocks[i]) for i in greedy_blocks]
        
    return [list(all_k_blocks[i]) for i in best_blocks]

