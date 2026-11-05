import itertools
import time
import random
import numpy as np
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # 1. Pre-computation
    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    n_k = len(all_k_blocks)
    
    # Build M (n_k x n_t) and reverse index (n_t x list of blocks)
    row_indices = []
    col_indices = []
    data = []
    subset_to_blocks = [[] for _ in range(n_t)]
    
    for i, block in enumerate(all_k_blocks):
        for sub in itertools.combinations(block, t):
            j = t_sub_to_idx[sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            subset_to_blocks[j].append(i)
            
    M = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int32)
    # Convert subset_to_blocks to numpy arrays for faster intersection?
    # Lists are fine for size ~364.
    
    best_blocks = None
    best_count = float('inf')
    
    def run_greedy_and_optimize(randomize=False):
        # Greedy
        uncovered = np.ones(n_t, dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline:
                return None # Timeout in greedy? Unlikely, but check.
            
            gains = M.dot(uncovered)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            if randomize:
                # Pick from top candidates
                # Find indices with gain > 0.8 * max_gain (or similar)
                # To keep it simple, pick random from those with gain == max_gain?
                # Or just pick random block with gain > 0?
                # Better: Pick random from top 100 gains?
                # Let's stick to deterministic greedy for now, maybe add noise later.
                # Actually, randomized greedy is key for escaping local minima.
                # Let's pick a block with gain >= 0.9 * max_gain randomly.
                threshold = 0.9 * max_gain
                candidates = np.where(gains >= threshold)[0]
                if len(candidates) == 0:
                    best_idx = np.argmax(gains)
                else:
                    best_idx = random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                
            selected.append(best_idx)
            
            # Update uncovered
            cols = M[best_idx].indices
            uncovered[cols] = 0
            
        # Local Optimization
        current_blocks = selected[:]
        counts = np.zeros(n_t, dtype=np.int32)
        for b in current_blocks:
            cols = M[b].indices
            counts[cols] += 1
            
        def delete_redundant():
            nonlocal current_blocks, counts
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Iterate in reverse to handle pops safely? Or just mark for deletion.
                # Marking is safer.
                to_delete = []
                for i in range(len(current_blocks)):
                    b = current_blocks[i]
                    cols = M[b].indices
                    if np.min(counts[cols]) > 1:
                        to_delete.append(i)
                
                if to_delete:
                    # Remove from end to start to keep indices valid
                    for i in sorted(to_delete, reverse=True):
                        b = current_blocks[i]
                        cols = M[b].indices
                        counts[cols] -= 1
                        current_blocks.pop(i)
                    changed = True
                    
        def try_swap_improve():
            nonlocal current_blocks, counts
            # Try to swap each block
            # To avoid O(N^2), we only look for swaps that cover critical subsets.
            # We iterate blocks.
            # If time is tight, limit swaps.
            swaps_made = 0
            # Randomize order to avoid bias
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    return
                if i >= len(current_blocks):
                    continue
                b = current_blocks[i]
                cols = M[b].indices
                # Find critical subsets
                critical_mask = (counts[cols] == 1)
                if not np.any(critical_mask):
                    continue # Redundant, handled by delete_redundant
                
                critical_subs = cols[critical_mask]
                
                # Find candidates covering all critical_subs
                # Intersection of subset_to_blocks for each critical sub
                # Start with the smallest list
                lists = [subset_to_blocks[s] for s in critical_subs]
                if not lists:
                    continue
                lists.sort(key=len)
                candidates_set = set(lists[0])
                for lst in lists[1:]:
                    if not candidates_set:
                        break
                    candidates_set.intersection_update(lst)
                    
                if not candidates_set:
                    continue
                    
                # Pick a candidate
                # Avoid picking the current block
                candidates_list = [c for c in candidates_set if c != b]
                if not candidates_list:
                    continue
                
                cand = random.choice(candidates_list)
                
                # Perform swap
                # Remove b
                old_cols = M[b].indices
                counts[old_cols] -= 1
                # Check if valid?
                # We know cand covers all critical subs of b.
                # Are there any other subsets that become uncovered?
                # No, because critical subs are those with count 1.
                # Removing b reduces count to 0 for critical subs.
                # Adding cand increases count back to 1 for critical subs (since cand covers them).
                # So validity is guaranteed.
                
                # Add cand
                new_cols = M[cand].indices
                counts[new_cols] += 1
                
                current_blocks[i] = cand
                swaps_made += 1
                # After swap, some other blocks might become redundant.
                # We don't delete immediately to avoid O(N^2) in inner loop.
                # We'll call delete_redundant after the pass.
                
            if swaps_made > 0:
                delete_redundant()
                
        # Optimization Loop
        # Alternate between deletion and swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Delete
            old_len = len(current_blocks)
            delete_redundant()
            if len(current_blocks) < old_len:
                improved = True
                
            # Swap
            old_len = len(current_blocks)
            try_swap_improve()
            # try_swap_improve calls delete_redundant if swaps made
            # So if swaps made and deletions happened, improved is True
            # But try_swap_improve doesn't return improved status directly.
            # Let's check length change or just assume if swaps made, we might improve.
            # To be safe, check if length changed after swap pass.
            # Actually, try_swap_improve updates current_blocks.
            # We need to know if it helped.
            # Let's modify try_swap_improve to return True if any swap was made.
            
        return current_blocks
    
    # Run multiple iterations
    # 1. Deterministic Greedy
    res = run_greedy_and_optimize(randomize=False)
    if res:
        best_blocks = res
        best_count = len(res)
        
    # 2. Randomized Greedies
    while time.time() < deadline - 10:
        res = run_greedy_and_optimize(randomize=True)
        if res and len(res) < best_count:
            best_blocks = res
            best_count = len(res)
            
    if best_blocks is None:
        # Fallback: Return empty or something? No, must be valid.
        # If greedy failed, return empty? No.
        # Greedy should always succeed.
        return []
        
    return [list(all_k_blocks[i]) for i in best_blocks]
