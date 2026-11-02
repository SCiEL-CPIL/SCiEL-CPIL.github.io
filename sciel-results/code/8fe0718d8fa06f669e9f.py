import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the list of t-subset IDs it covers
    block_t_ids = []
    for block in all_k_subsets:
        ids = [t_subset_to_id[s] for s in itertools.combinations(block, t)]
        block_t_ids.append(ids)
        
    # For each t-subset ID, store the list of k-subset indices that cover it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, ids in enumerate(block_t_ids):
        for tid in ids:
            t_to_blocks[tid].append(b_idx)

    # 1. Greedy Construction
    # We maintain a list of uncovered t-subsets.
    # To make this efficient, we use a boolean array for covered status and a list of uncovered indices.
    
    def greedy_construct():
        is_covered = [False] * num_t_subsets
        uncovered_indices = list(range(num_t_subsets))
        selected_blocks = []
        
        while uncovered_indices:
            # Find the block that covers the most uncovered t-subsets
            best_block_idx = -1
            max_covered = -1
            
            # Optimization: Instead of scanning all 75k blocks every time,
            # we can pick a random uncovered t-subset and look at its candidates.
            # This is a "randomized greedy" which is often faster and good enough.
            target_tid = random.choice(uncovered_indices)
            candidates = t_to_blocks[target_tid]
            
            # Among candidates, pick the one covering most total uncovered
            for b_idx in candidates:
                # Count how many uncovered t-subsets this block covers
                # We can do this by checking is_covered for each tid in block_t_ids[b_idx]
                # But we only care about those in uncovered_indices?
                # Actually, if we just count total uncovered it covers, that's fine.
                
                # To speed up, we can precompute nothing, just loop.
                # 56 iterations per candidate.
                # Number of candidates for a t-subset is C(v-t, k-t) = C(14, 3) = 364.
                # So 364 * 56 ~ 20k ops per step.
                # Number of steps ~ 300. Total ~ 6M ops. Acceptable.
                
                count = 0
                for tid in block_t_ids[b_idx]:
                    if not is_covered[tid]:
                        count += 1
                
                if count > max_covered:
                    max_covered = count
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                selected_blocks.append(best_block_idx)
                # Update covered status
                for tid in block_t_ids[best_block_idx]:
                    if not is_covered[tid]:
                        is_covered[tid] = True
                        # Remove from uncovered_indices?
                        # Removing from a list is O(N).
                        # Better to just scan is_covered in the next step?
                        # Or maintain a set of uncovered?
                        # Let's just keep the list and filter it occasionally or just use the boolean array.
                        # Since we pick a random uncovered, we need to know which are uncovered.
                        # We can just pick a random index from 0..N-1 and check if covered.
                        # If covered, try again.
                        # This is inefficient if many are covered.
                        
                        # Alternative: Maintain a list of uncovered indices.
                        # Removal is O(N).
                        # Let's stick to the boolean array and a helper to get an uncovered one.
            
            # To avoid O(N) removal, we can just rebuild the uncovered list every K steps?
            # Or, simpler: just pick a random index from 0 to N-1, if covered, try another.
            # Since coverage increases, the probability of picking an uncovered one drops.
            # However, for the first 100 blocks, most are uncovered.
            
            # Let's optimize the "find uncovered" part.
            # We'll just use the boolean array.
            
        return selected_blocks

    # Better Greedy:
    def greedy_construct_v2():
        is_covered = [False] * num_t_subsets
        selected_blocks = []
        
        # We will maintain a list of uncovered t-ids to sample from.
        # To handle removal efficiently, we can just filter the list when it gets too small or every few steps.
        # Or, we can just iterate through all t-ids to find an uncovered one? No, that's O(N) per step.
        
        # Let's use a random probe approach for finding an uncovered target.
        # And for finding the best block, we look at candidates of that target.
        
        while True:
            # Find an uncovered t-subset
            # If we maintain a list, we can pick from it.
            # Let's maintain a list `uncovered_list`.
            # Initially all.
            # When we cover a tid, we don't remove it immediately, but mark is_covered[tid]=True.
            # When picking a target, we pick random from uncovered_list.
            # If is_covered[target] is True, we remove it from uncovered_list (O(N) removal).
            # To avoid O(N) removal, we can use a set? Or just allow stale entries and filter?
            
            # Let's try a different strategy:
            # 1. Start with all t-subsets uncovered.
            # 2. Pick a random uncovered t-subset S.
            # 3. Find block B covering S that covers max number of *other* uncovered t-subsets.
            # 4. Add B, mark its t-subsets covered.
            
            # To pick a random uncovered t-subset efficiently:
            # We can keep a list of all t-subset indices.
            # We pick a random index. If it's covered, we try again.
            # As coverage increases, this becomes slow.
            
            # Better: Keep a list of uncovered indices.
            # When we cover a tid, we don't remove it from the list immediately.
            # We just mark it covered.
            # When we need a target, we pick a random index from the list.
            # If it's covered, we remove it from the list (using swap-pop for O(1) removal).
            
            uncovered_indices = list(range(num_t_subsets))
            
            while uncovered_indices:
                # Pick a random index from the list
                idx = random.randint(0, len(uncovered_indices) - 1)
                target_tid = uncovered_indices[idx]
                
                if is_covered[target_tid]:
                    # Remove from list (swap with last, pop)
                    uncovered_indices[idx] = uncovered_indices[-1]
                    uncovered_indices.pop()
                    continue
                
                # target_tid is uncovered.
                # Find best block covering target_tid.
                candidates = t_to_blocks[target_tid]
                best_block = -1
                max_new_cover = -1
                
                for b_idx in candidates:
                    # Count uncovered t-subsets in this block
                    new_cover = 0
                    for tid in block_t_ids[b_idx]:
                        if not is_covered[tid]:
                            new_cover += 1
                            
                    if new_cover > max_new_cover:
                        max_new_cover = new_cover
                        best_block = b_idx
                        
                if best_block != -1:
                    selected_blocks.append(best_block)
                    # Update covered status
                    for tid in block_t_ids[best_block]:
                        if not is_covered[tid]:
                            is_covered[tid] = True
                            # Note: We do NOT remove tid from uncovered_indices here.
                            # It will be removed when it is picked as a target and found to be covered.
                            # This keeps the list size large but allows O(1) "removal" when encountered.
                            
        return selected_blocks

    initial_solution = greedy_construct_v2()
    
    if not initial_solution:
        return [list(all_k_subsets[0])]
    
    # 2. Local Search Optimization
    # State: list of block indices (set for O(1) lookup)
    active_blocks = set(initial_solution)
    
    # Maintain cover counts
    cover_counts = [0] * num_t_subsets
    for b_idx in active_blocks:
        for tid in block_t_ids[b_idx]:
            cover_counts[tid] += 1
            
    # Helper to check if a block is redundant
    def is_redundant(b_idx):
        for tid in block_t_ids[b_idx]:
            if cover_counts[tid] == 1:
                return False
        return True

    # Helper to find a swap for a redundant block
    # If block A is redundant, we can just delete it.
    # If block A is NOT redundant, can we swap it with B?
    # Condition: For all tid in A, if cover_counts[tid] == 1, then tid must be in B.
    # Let U_A = { tid in A | cover_counts[tid] == 1 }
    # We need B such that U_A subset B.
    
    def find_swap(b_idx):
        # Calculate U_A
        unique_tids = []
        for tid in block_t_ids[b_idx]:
            if cover_counts[tid] == 1:
                unique_tids.append(tid)
                
        if not unique_tids:
            return None # It's redundant, should be handled by deletion
        
        # We need a block B not in active_blocks such that all tids in unique_tids are in B.
        # To speed this up, we can intersect the candidate lists of the unique tids.
        # Candidates for B must be in t_to_blocks[tid] for all tid in unique_tids.
        
        # Start with candidates of the first unique tid
        # To minimize intersection size, pick the one with fewest candidates?
        # All tids have same number of candidates C(v-t, k-t).
        
        candidates = set(t_to_blocks[unique_tids[0]])
        for tid in unique_tids[1:]:
            candidates.intersection_update(t_to_blocks[tid])
            
        # Now candidates contains all blocks covering all unique tids of A.
        # We need one that is NOT in active_blocks.
        for b_candidate in candidates:
            if b_candidate not in active_blocks:
                return b_candidate
                
        return None

    # Main Local Search Loop
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Delete redundant blocks
        # Iterate through a copy of active_blocks
        for b_idx in list(active_blocks):
            if is_redundant(b_idx):
                # Remove it
                active_blocks.remove(b_idx)
                for tid in block_t_ids[b_idx]:
                    cover_counts[tid] -= 1
                improved = True
                # Restart deletion phase? Or continue?
                # Continuing is fine, but cover counts change.
                # If we remove A, B might become redundant.
                # So we should loop until no more deletions.
                # The `improved` flag handles the outer loop.
                # But within this loop, we are iterating over a static list.
                # Newly redundant blocks won't be checked until next outer loop iteration.
                # That's okay.
        
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Phase 2: Swaps
        # Try to swap non-redundant blocks to make them redundant or reduce size?
        # Swaps don't reduce size directly. They enable future deletions.
        # Strategy: For each block A, if it's not redundant, try to find B such that swapping A->B makes B redundant?
        # Or simply, try to swap A->B if it allows a deletion later?
        
        # A simpler heuristic:
        # Try to swap A with B if B is "better".
        # "Better" could mean B covers more t-subsets with count 1?
        
        # Let's try: For each block A, find U_A.
        # If U_A is not empty, find B containing U_A.
        # If we swap A->B, the cover counts change.
        # Does this help?
        # If B contains U_A, then the tids in U_A are still covered by B.
        # Other tids in A (count > 1) lose one cover.
        # Tids in B (count > 0) gain one cover.
        
        # A common effective move:
        # Pick a block A.
        # Pick a block B not in solution.
        # If replacing A with B keeps the solution valid, accept it.
        # This is what find_swap does (checking U_A subset B).
        # But why do it?
        # If we swap A->B, and then B becomes redundant, we can delete B.
        # Net effect: A removed, B added then removed. Size same.
        # But the structure changed. Maybe other blocks become redundant.
        
        # Let's just try random swaps that maintain validity.
        # And then run the deletion phase again.
        
        # To avoid infinite loops of useless swaps, we should only accept a swap if it leads to a deletion in the subsequent phase?
        # Or just do a limited number of swaps.
        
        # Let's try:
        # Pick a random block A from active_blocks.
        # If A is redundant, it would have been deleted. So A is not redundant.
        # Find B such that swap A->B is valid.
        # If found, perform swap.
        # Mark improved = True (to trigger another deletion pass).
        
        # To make it efficient, we only try this for a few blocks per iteration.
        
        # Shuffle active blocks to pick random ones
        active_list = list(active_blocks)
        random.shuffle(active_list)
        
        swaps_done = 0
        for b_idx in active_list:
            if time.time() - start_time > time_limit - 2.0:
                break
                
            # Check if still in active_blocks (might have been deleted in previous phase? No, we are in swap phase)
            if b_idx not in active_blocks:
                continue
                
            # Find a valid swap
            b_new = find_swap(b_idx)
            if b_new is not None:
                # Perform swap
                # Remove b_idx
                active_blocks.remove(b_idx)
                for tid in block_t_ids[b_idx]:
                    cover_counts[tid] -= 1
                    
                # Add b_new
                active_blocks.add(b_new)
                for tid in block_t_ids[b_new]:
                    cover_counts[tid] += 1
                    
                improved = True
                swaps_done += 1
                # Limit swaps per iteration to avoid getting stuck
                if swaps_done > 10:
                    break

    # Final Repair (just in case)
    final_solution = set(active_blocks)
    final_cover_counts = [0] * num_t_subsets
    for b_idx in final_solution:
        for tid in block_t_ids[b_idx]:
            final_cover_counts[tid] += 1
            
    uncovered_t_ids = [i for i in range(num_t_subsets) if final_cover_counts[i] == 0]
    
    while uncovered_t_ids:
        if time.time() - start_time > time_limit - 1.0:
            break
        target_tid = uncovered_t_ids[0]
        candidates = t_to_blocks[target_tid]
        best_block = -1
        max_new_cover = -1
        
        for b_idx in candidates:
            if b_idx in final_solution:
                continue
            new_cover = 0
            for tid in block_t_ids[b_idx]:
                if final_cover_counts[tid] == 0:
                    new_cover += 1
            if new_cover > max_new_cover:
                max_new_cover = new_cover
                best_block = b_idx
                
        if best_block != -1:
            final_solution.add(best_block)
            for tid in block_t_ids[best_block]:
                final_cover_counts[tid] += 1
                if final_cover_counts[tid] == 1:
                    # Remove from uncovered list
                    # O(N) removal, but list is small
                    try:
                        uncovered_t_ids.remove(tid)
                    except ValueError:
                        pass
        else:
            break

    result = [list(all_k_subsets[i]) for i in final_solution]
    return result
