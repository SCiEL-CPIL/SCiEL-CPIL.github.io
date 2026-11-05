import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # 1. Precomputation
    # Generate all t-subsets (constraints)
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Generate all k-subsets (potential blocks/variables)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map t-subsets to indices for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each block, find which t-subsets it covers
    # block_tsubs[i] is a list of indices of t-subsets covered by block i
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
    
    # For each t-subset, find which blocks cover it
    # tsub_blocks[j] is a list of indices of blocks covering t-subset j
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # Convert to numpy arrays for fast vectorized operations where possible
    # Note: block_tsubs is a list of lists of varying lengths. 
    # We will use Python lists for the sparse structure in the WalkSAT logic 
    # as the "clause" structure is irregular.
    
    # 2. WalkSAT / Max-SAT Solver
    
    # State: 
    # selected: boolean array of size num_blocks. True if block is in solution.
    # cover_count: array of size num_tsubs. Count of selected blocks covering each t-subset.
    # num_uncovered: count of t-subsets with cover_count == 0.
    
    def greedy_init():
        """
        Fast greedy initialization to get a valid starting point.
        """
        rng = random.Random(42)
        selected = np.zeros(num_blocks, dtype=bool)
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        uncovered_count = num_tsubs
        selected_list = []
        
        while uncovered_count > 0:
            if check_time(10.0):
                break
            # Find an uncovered t-subset
            # To be fast, we can pick a random index and check, or maintain a list of uncovered.
            # Maintaining a list is O(1) to pick but O(n) to update.
            # Picking random is O(1) to pick but might hit covered ones.
            # Given num_tsubs is ~11628 for C(19,5), and coverage is dense, 
            # random picking is likely fast enough in early stages.
            # However, as we approach completion, random picking becomes inefficient.
            # Let's use a list of uncovered indices.
            
            # Optimization: Instead of maintaining a dynamic list which is hard to update 
            # with numpy cover_count, we will just scan if needed or use a simpler heuristic.
            # For initialization, let's just pick random until we find one.
            found = False
            for _ in range(100): # Try 100 random picks
                idx = rng.randint(0, num_tsubs - 1)
                if cover_count[idx] == 0:
                    found = True
                    break
            
            if not found:
                # Fallback: linear scan
                for idx in range(num_tsubs):
                    if cover_count[idx] == 0:
                        found = True
                        break
                if not found:
                    break # Should not happen if logic is correct
            
            # Get blocks covering this t-subset
            candidates = tsub_blocks[idx]
            if not candidates:
                break # Impossible to cover
            
            # Choose block that covers most *other* uncovered t-subsets
            # Gain = number of uncovered t-subsets in the block
            best_block = -1
            best_gain = -1
            for b_idx in candidates:
                gain = 0
                for t_idx in block_tsubs[b_idx]:
                    if cover_count[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            # Select block
            selected[best_block] = True
            selected_list.append(best_block)
            for t_idx in block_tsubs[best_block]:
                if cover_count[t_idx] == 0:
                    uncovered_count -= 1
                cover_count[t_idx] += 1
                
        return selected_list, cover_count, uncovered_count

    def reduce_blocks(selected_list, cover_count):
        """
        Remove redundant blocks.
        """
        if not selected_list:
            return selected_list, cover_count
            
        selected_set = set(selected_list)
        new_selected_list = []
        
        # We need to check if a block is essential.
        # A block is essential if it is the *only* block covering some t-subset.
        # We can compute this by checking cover_count for t-subsets in the block.
        
        # To be safe and fast, we iterate through blocks and try to remove them.
        # If removal causes any t-subset to become uncovered (count drops to 0), we keep it.
        
        # Optimization: Only consider blocks that might be redundant.
        # Actually, just iterate all.
        
        # We must be careful with cover_count updates.
        # Let's make a copy of cover_count to simulate removals?
        # No, we can update in place if we revert on failure.
        
        changed = True
        max_passes = 50
        for _ in range(max_passes):
            if check_time(5.0):
                break
            changed = False
            new_list = []
            for b_idx in selected_list:
                # Simulate removal
                b_tsubs = block_tsubs[b_idx]
                # Check if any t-subset in this block has count == 1
                # If yes, removing it makes count 0 -> Bad.
                # If no, removing it keeps counts >= 1 -> Good (redundant).
                
                is_essential = False
                for t_idx in b_tsubs:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                        
                if not is_essential:
                    # Remove it
                    changed = True
                    for t_idx in b_tsubs:
                        cover_count[t_idx] -= 1
                    # Do not add to new_list
                else:
                    # Keep it
                    new_list.append(b_idx)
                    
            selected_list = new_list
            if not changed:
                break
                
        return selected_list, cover_count

    # WalkSAT Main Loop
    def walk_sat(selected_list, cover_count, uncovered_count):
        """
        Stochastic Local Search.
        Objective: Minimize len(selected_list).
        Constraint: uncovered_count == 0.
        
        Moves:
        1. Try to remove a random block. If valid (uncovered remains 0), accept.
        2. If invalid, revert. Then, try to "repair" by adding a block that covers 
           the newly uncovered t-subsets, potentially swapping with another block 
           or just adding if it leads to a better state in the long run?
           
        Actually, standard WalkSAT for Min-SAT:
        State: Set of True variables.
        Score: Number of True variables + Penalty * Number of Unsatisfied Clauses.
        
        Let's use Energy = len(selected) + 1000 * uncovered_count.
        
        Moves:
        - Flip a variable (add or remove a block).
        - If energy decreases, accept.
        - If energy increases, accept with probability exp(-delta/T).
        """
        
        rng = random.Random(123)
        T = 50.0
        alpha = 0.99
        T_min = 0.1
        max_iters = 50000
        
        # Current state
        cur_selected = set(selected_list)
        cur_cover = cover_count.copy()
        cur_uncovered = uncovered_count
        cur_energy = len(cur_selected) + 1000 * cur_uncovered
        
        best_selected = set(cur_selected)
        best_cover = cur_cover.copy()
        best_uncovered = cur_uncovered
        best_energy = cur_energy
        
        iter_count = 0
        
        while T > T_min and iter_count < max_iters:
            if check_time(10.0):
                break
            iter_count += 1
            
            # Choose a move: Add or Remove a block
            # Bias towards removal to minimize size, but need to maintain coverage.
            # If we remove and coverage breaks, we must repair.
            
            # Strategy: 
            # 1. Pick a block to remove (random from selected).
            # 2. Check if removal breaks coverage.
            # 3. If it breaks, pick a block to add (from those covering the broken t-subsets) 
            #    that minimizes the "damage" or restores coverage best.
            #    This is a "swap" move effectively.
            # 4. If removal doesn't break coverage, accept (pure reduction).
            
            if not cur_selected:
                break
                
            remove_idx = rng.choice(list(cur_selected))
            
            # Calculate effect of removal
            b_tsubs = block_tsubs[remove_idx]
            newly_uncovered = 0
            for t_idx in b_tsubs:
                if cur_cover[t_idx] == 1:
                    newly_uncovered += 1
            
            if newly_uncovered == 0:
                # Safe removal
                cur_selected.discard(remove_idx)
                for t_idx in b_tsubs:
                    cur_cover[t_idx] -= 1
                cur_uncovered = 0 # Remains 0
                cur_energy = len(cur_selected) + 1000 * cur_uncovered
                
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_selected = set(cur_selected)
                    best_cover = cur_cover.copy()
                    best_uncovered = cur_uncovered
            else:
                # Removal breaks coverage. 
                # We must add a block to repair.
                # Find t-subsets that became uncovered (those with count 1 before removal)
                # Actually, we know which t-subsets are at risk.
                # Let's pick one of the newly uncovered t-subsets and find a block to cover it.
                
                # Identify the specific t-subsets that will become uncovered
                broken_tsubs = []
                for t_idx in b_tsubs:
                    if cur_cover[t_idx] == 1:
                        broken_tsubs.append(t_idx)
                        
                if not broken_tsubs:
                    continue # Should not happen if newly_uncovered > 0
                
                # Pick a random broken t-subset to target for repair
                target_tsub = rng.choice(broken_tsubs)
                
                # Candidates for addition: blocks that cover target_tsub
                # Exclude the one we are removing (since we are swapping)
                candidates = tsub_blocks[target_tsub]
                # candidates is a list of block indices.
                
                if not candidates:
                    continue # Impossible to repair, revert/skip
                
                # Choose a candidate block to add
                # Heuristic: Pick one that covers the most *other* broken t-subsets
                # or generally has high coverage of currently low-count t-subsets.
                
                best_add_idx = -1
                best_score = -1
                
                # Limit candidates to speed up? 
                # tsub_blocks[target_tsub] can be large (up to C(19-5, 8-5) = C(14,3) = 364? No, C(14,3) is for points.
                # Number of blocks covering a specific 5-subset: 
                # We need to choose k-t=3 more points from v-t=14. C(14,3) = 364.
                # So candidates list is small (~364). We can check all.
                
                for add_idx in candidates:
                    if add_idx == remove_idx:
                        continue
                    if add_idx in cur_selected:
                        continue # We want a swap or add. If already selected, adding does nothing.
                                 # Actually, if it's already selected, it's already covering the target?
                                 # No, if it was selected, cover_count[target] would be >= 2 (since remove_idx is also selected and covers it).
                                 # Wait, if cover_count[target] == 1, then ONLY remove_idx covers it.
                                 # So no other selected block covers it. So add_idx cannot be in cur_selected.
                                 
                    # Score the add_idx: How many of the broken_tsubs does it cover?
                    score = 0
                    a_tsubs = block_tsubs[add_idx]
                    for bt in broken_tsubs:
                        if bt in a_tsubs: # This is O(N) check. 
                            # Optimization: Precompute set? Or just check if cover_count[bt] is affected.
                            # Actually, we just want to know if it covers the broken ones.
                            pass # This inner loop is too slow if done naively for 364 candidates * 364 checks.
                            
                # Faster scoring:
                # For each candidate, count how many t-subsets in its coverage list have cur_cover == 1 (and are in broken_tsubs? No, just generally low count).
                # Actually, the primary goal is to cover the broken ones.
                # Let's just pick a random candidate from those that cover the target, 
                # or one that minimizes the "newly created" uncovered t-subsets?
                # Simpler: Pick the one that covers the most t-subsets currently with count == 1.
                
                best_add_idx = -1
                best_gain = -1
                for add_idx in candidates:
                    if add_idx in cur_selected:
                        continue
                    gain = 0
                    for t_idx in block_tsubs[add_idx]:
                        if cur_cover[t_idx] == 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_add_idx = add_idx
                        
                if best_add_idx == -1:
                    continue # No good candidate, skip move
                    
                # Perform Swap: Remove remove_idx, Add best_add_idx
                # Calculate Energy Change
                # Remove remove_idx:
                for t_idx in b_tsubs:
                    cur_cover[t_idx] -= 1
                # Add best_add_idx:
                add_tsubs = block_tsubs[best_add_idx]
                for t_idx in add_tsubs:
                    cur_cover[t_idx] += 1
                    
                # Recalculate uncovered count
                # This is O(num_tsubs) if done naively. 
                # We can update cur_uncovered incrementally.
                # But we already modified cur_cover. 
                # Let's just recount for the affected t-subsets?
                # No, cur_uncovered is global.
                # We know which t-subsets changed: union of b_tsubs and add_tsubs.
                # Let's store old counts for these.
                
                # This incremental update is getting complex and error-prone in the loop.
                # Given the small size (num_tsubs ~ 11k), a full recount of uncovered is ~11k ops.
                # 50k iters * 11k ops = 550M ops. Might be too slow in Python.
                # Let's optimize: Only update cur_uncovered based on the specific t-subsets changed.
                
                # Let's revert to a simpler, safer approach for the SWAP move:
                # 1. Tentatively apply swap.
                # 2. Count new_uncovered for the affected set.
                # 3. Calculate delta.
                
                # Let's restart the move logic with a cleaner incremental update.
                
                # REVERT the tentative changes we just made for calculation purposes?
                # No, let's just do the calculation properly.
                
                # Actually, let's just use the full recount if it's fast enough in NumPy?
                # cur_cover is a numpy array.
                # new_uncovered = np.sum(cur_cover == 0)
                # This is fast (C loop).
                
                # Apply changes:
                # (We already applied them in the code above? No, I commented out the application in the "Fast scoring" part, 
                #  but I did apply them in the "Perform Swap" section just before. 
                #  Wait, I applied them. Let's verify state.)
                
                # State is now: remove_idx removed, best_add_idx added.
                # cur_cover is updated.
                # cur_selected needs update.
                
                new_uncovered = int(np.sum(cur_cover == 0))
                new_energy = len(cur_selected) + 1 # We are swapping, so len stays same? 
                # Wait, I haven't updated cur_selected yet.
                # cur_selected currently contains remove_idx.
                # Let's update cur_selected.
                cur_selected.discard(remove_idx)
                cur_selected.add(best_add_idx)
                
                # len(cur_selected) is same as before (swap).
                new_energy = len(cur_selected) + 1000 * new_uncovered
                
                # Acceptance
                delta = new_energy - cur_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    cur_uncovered = new_uncovered
                    cur_energy = new_energy
                    if cur_energy < best_energy:
                        best_energy = cur_energy
                        best_selected = set(cur_selected)
                        best_cover = cur_cover.copy()
                        best_uncovered = cur_uncovered
                else:
                    # Reject: Revert
                    # Re-add remove_idx
                    for t_idx in b_tsubs:
                        cur_cover[t_idx] += 1
                    # Remove best_add_idx
                    for t_idx in add_tsubs:
                        cur_cover[t_idx] -= 1
                    cur_selected.discard(best_add_idx)
                    cur_selected.add(remove_idx)
                    # cur_uncovered and cur_energy remain unchanged (from before move)
                    
            T *= alpha
            
        return list(best_selected), best_cover, best_uncovered

    # Execution
    # 1. Greedy Init
    selected_list, cover_count, uncovered_count = greedy_init()
    
    # 2. Reduce
    selected_list, cover_count = reduce_blocks(selected_list, cover_count)
    
    # 3. WalkSAT
    # Run multiple times with different seeds/temps if time permits
    best_list = selected_list
    best_len = len(selected_list)
    
    # Use time budget for multiple SA runs
    for i in range(10):
        if check_time(20.0):
            break
        # Reset to best found so far for this restart? Or fresh greedy?
        # Fresh greedy might be slower. Let's use the current best as start.
        # But we need a valid cover_count.
        # Let's just use the initial reduced list as a robust start.
        
        # Actually, let's just run WalkSAT on the current best_list.
        # We need cover_count for best_list.
        # If we don't have it, we have to rebuild it.
        
        # To keep it simple and robust:
        # Rebuild cover_count from best_list
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in best_list:
            cover_count[block_tsubs[b_idx]] += 1
        uncovered_count = int(np.sum(cover_count == 0))
        
        res_list, res_cover, res_uncovered = walk_sat(best_list, cover_count, uncovered_count)
        
        if res_uncovered == 0 and len(res_list) < best_len:
            best_len = len(res_list)
            best_list = res_list
            
    # Final reduction
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in best_list:
        cover_count[block_tsubs[b_idx]] += 1
    best_list, _ = reduce_blocks(best_list, cover_count)
    
    # Convert indices to blocks
    return [list(all_blocks[b]) for b in best_list]

if __name__ == "__main__":
    # Quick test
    blocks = construct(19, 8, 5)
    print(f"Found covering with {len(blocks)} blocks")
