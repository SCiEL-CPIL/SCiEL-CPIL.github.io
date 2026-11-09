import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using a randomized greedy 
    set cover approach followed by local search for block removal and exchange.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def time_ok():
        return time.time() < deadline - 5.0

    # Precompute all t-subsets (the universe to be covered)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_covered_t = [None] * num_blocks
    block_covered_t_set = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices
        block_covered_t_set[i] = set(covered_indices)
    
    # Step 1: Remove Redundant Blocks
    def try_remove_blocks(blocks):
        if not blocks:
            return blocks
        
        # Build coverage counts
        cover_count = [0] * num_t_subsets
        for b_idx in blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        # Greedy removal:
        # Iterate through blocks. If a block is redundant, remove it.
        # Repeat until no more can be removed?
        # Or just one pass? One pass is faster but less effective.
        
        # Let's do a multi-pass removal.
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(len(blocks)):
                if not time_ok():
                    return blocks
                
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block i
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop since indices changed
        
        return blocks
    
    # Step 2: 1-Exchange Local Search
    def one_exchange_search(blocks, max_iters=200):
        if not blocks:
            return blocks
        
        current_blocks = list(blocks)
        
        cover_count = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        selected_set = set(current_blocks)
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iters and time_ok():
            improved = False
            iterations += 1
            
            # First pass: remove redundant blocks
            removed_any = False
            for i in range(len(current_blocks)):
                if not time_ok():
                    break
                b_idx = current_blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    selected_set.discard(b_idx)
                    improved = True
                    removed_any = True
                    break
            
            if removed_any:
                continue
            
            # Try 1-exchanges with smarter candidate selection
            for i in range(len(current_blocks)):
                if not time_ok():
                    break
                
                b_idx = current_blocks[i]
                
                # Find essential t-subsets (only covered by this block)
                essential_t = [t_idx for t_idx in block_covered_t[b_idx] if cover_count[t_idx] == 1]
                
                if not essential_t:
                    # Block is redundant, remove it
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    selected_set.discard(b_idx)
                    improved = True
                    break
                
                # Build candidate set from blocks that cover ALL essential t-subsets
                # Use the most constrained essential t-subset to start
                # Sort essential_t by rarity (fewest blocks covering them)
                essential_t_sorted = sorted(essential_t, key=lambda x: len(t_to_blocks[x]))
                
                candidate_set = set(t_to_blocks[essential_t_sorted[0]])
                
                # Filter to only candidates that cover all essential t-subsets
                filtered_candidates = []
                for cand in candidate_set:
                    if cand in selected_set:
                        continue
                    all_covered = True
                    for t_idx in essential_t_sorted[1:]:
                        if t_idx not in block_covered_t_set[cand]:
                            all_covered = False
                            break
                    if all_covered:
                        filtered_candidates.append(cand)
                
                if not filtered_candidates:
                    continue
                
                # Limit candidates for efficiency
                if len(filtered_candidates) > 500:
                    random.shuffle(filtered_candidates)
                    filtered_candidates = filtered_candidates[:500]
                
                best_swap_block = -1
                best_swap_metric = float('-inf')
                
                # Precompute current block's contribution
                old_block_t_set = block_covered_t_set[b_idx]
                
                for cand_idx in filtered_candidates:
                    cand_t_set = block_covered_t_set[cand_idx]
                    
                    # Calculate metric: net new coverage minus overlap penalty
                    new_cover = 0
                    overlap_penalty = 0
                    for t_idx in block_covered_t[cand_idx]:
                        if cover_count[t_idx] == 0:
                            new_cover += 1
                        elif cover_count[t_idx] == 1:
                            # This t-subset is only covered by the block we're removing
                            # So it will become uncovered unless covered by another block
                            # But wait, if cover_count is 1, it means ONLY the current block covers it.
                            # If we remove the current block, it becomes 0.
                            # If the new block covers it, it becomes 1.
                            # So this is not a penalty, it's a necessity.
                            # The penalty is for t-subsets that are covered by multiple blocks,
                            # where adding the new block increases redundancy unnecessarily?
                            # No, the goal is to minimize block count.
                            # If we swap, the count stays the same.
                            # We want to maximize the number of t-subsets that become "more redundant"
                            # or minimize the number that become "less redundant"?
                            
                            # Actually, the standard metric for 1-exchange in set cover is:
                            # Gain = (# of t-subsets covered by New Block but not Old Block) 
                            #       - (# of t-subsets covered by Old Block but not New Block)
                            # This doesn't directly account for redundancy.
                            
                            # Let's use a simpler heuristic:
                            # We must cover all essential_t.
                            # The "cost" of the new block is how many *other* t-subsets it covers
                            # that are already well-covered (cover_count > 1).
                            # The "benefit" is how many t-subsets it covers that are NOT covered by the old block.
                            
                            # Let's refine:
                            # Unique to Old: t in Old, not in New. If cover_count[t] == 1, it becomes uncovered (BAD).
                            # But we already ensured New covers all essential_t (where cover_count==1).
                            # So unique_to_old with cover_count==1 is empty.
                            # Unique to Old with cover_count>1: becomes cover_count-1. (Neutral/Good if it reduces redundancy)
                            # Unique to New: t in New, not in Old.
                            #   If cover_count[t] == 0: becomes 1. (Good, covers new ground? No, universe is fixed. 
                            #   Wait, cover_count is based on CURRENT blocks.
                            #   If t is not in Old, and not in any other block, cover_count[t] would be 0?
                            #   No, if t is uncovered, the solution is invalid. 
                            #   So all t have cover_count >= 1.
                            #   If t is not in Old, cover_count[t] is the count from OTHER blocks.
                            #   If we add New, cover_count[t] increases.
                            #   This increases redundancy.
                            
                            # So the metric should be:
                            # Minimize: (# of t in New \ Old where cover_count[t] >= 1) 
                            # This is just the number of t-subsets the new block covers that are not in the old block.
                            # Since all t-subsets are already covered, adding a new block only adds redundancy.
                            # We want the new block to add the LEAST redundancy.
                            
                            # Redundancy added = sum(1 for t in New \ Old)
                            # This is just len(New \ Old).
                            
                            # Is that it?
                            # If we swap Old for New:
                            # - t in Old \ New: cover_count decreases by 1.
                            # - t in New \ Old: cover_count increases by 1.
                            # - t in Old ∩ New: no change.
                            
                            # We need to ensure no t becomes uncovered.
                            # t in Old \ New becomes uncovered only if cover_count[t] was 1.
                            # We already checked essential_t (cover_count==1 in Old).
                            # If New covers all essential_t, then for any t in Old \ New with cover_count[t]==1,
                            # t must be in New (contradiction, t in Old \ New means t not in New).
                            # Wait.
                            # essential_t = {t in Old | cover_count[t] == 1}.
                            # If t in Old \ New, and t in essential_t, then t is not in New.
                            # So t becomes uncovered.
                            # So we MUST require New covers all essential_t.
                            # This implies Old \ New does NOT contain any essential_t.
                            # So for all t in Old \ New, cover_count[t] >= 2.
                            # So removing Old is safe.
                            
                            # So the only cost is the redundancy added by New \ Old.
                            # We want to minimize len(New \ Old).
                            
                            # Let's use that.
                            pass # Placeholder for metric update
                
                # Let's implement the metric: minimize len(New \ Old)
                # But we also want to prioritize blocks that cover many "rare" t-subsets?
                # No, all t-subsets are covered.
                # Minimizing overlap with existing coverage (excluding Old) is the goal.
                
                # Re-evaluating:
                # We want to replace Old with New.
                # Constraint: New covers all essential_t of Old.
                # Objective: Minimize "waste".
                # Waste = number of t-subsets covered by New that are already covered by other blocks.
                # Since all t-subsets are covered, Waste = len(New \ Old).
                # Wait, if t is in New \ Old, it is covered by other blocks (since it's not in Old, and it's covered).
                # So yes, Waste = len(New \ Old).
                
                # Let's try this simple metric first.
                
                for cand_idx in filtered_candidates:
                    # Calculate waste
                    waste = 0
                    for t_idx in block_covered_t[cand_idx]:
                        if t_idx not in old_block_t_set:
                            waste += 1
                    
                    if waste < best_swap_metric: # We want min waste, so init to inf
                         best_swap_metric = waste
                         best_swap_block = cand_idx
                
                # Wait, I initialized best_swap_metric to -inf.
                # Let's fix that.
                
                # Restarting the inner loop logic properly
                
                best_swap_block = -1
                min_waste = float('inf')
                
                for cand_idx in filtered_candidates:
                    waste = 0
                    for t_idx in block_covered_t[cand_idx]:
                        if t_idx not in old_block_t_set:
                            waste += 1
                    
                    if waste < min_waste:
                        min_waste = waste
                        best_swap_block = cand_idx
                
                if best_swap_block != -1:
                    # Perform swap
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_covered_t[best_swap_block]:
                        cover_count[t_idx] += 1
                    
                    current_blocks[i] = best_swap_block
                    selected_set.discard(b_idx)
                    selected_set.add(best_swap_block)
                    improved = True
                    break # Restart loop after swap
            
        return current_blocks

    # Step 3: Randomized Greedy
    def run_greedy(seed):
        random.seed(seed)
        uncovered_set = set(range(num_t_subsets))
        selected = []
        
        while uncovered_set:
            if not time_ok():
                break
            
            # Pick a random uncovered t-subset to focus on
            # This reduces the search space from all blocks to blocks covering this t-subset
            target_t_idx = random.choice(list(uncovered_set))
            candidates = t_to_blocks[target_t_idx]
            
            best_block = -1
            best_gain = -1
            
            for b_idx in candidates:
                # Calculate gain: number of uncovered t-subsets covered by this block
                gain = 0
                for t_idx in block_covered_t[b_idx]:
                    if t_idx in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break # Should not happen if uncovered_set is non-empty and valid input
                
            selected.append(best_block)
            
            # Update uncovered set
            for t_idx in block_covered_t[best_block]:
                if t_idx in uncovered_set:
                    uncovered_set.remove(t_idx)
        
        return selected
    
    # Step 4: Main Search Loop
    best_blocks = None
    best_len = float('inf')
    
    # Try a few deterministic seeds first
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5]
    
    for seed in seeds:
        if not time_ok():
            break
        
        greedy_res = run_greedy(seed)
        if not greedy_res:
            continue
            
        # Post-process: Remove redundant blocks
        processed_res = try_remove_blocks(greedy_res)
        
        # Post-process: 1-Exchange Search
        exchanged_res = one_exchange_search(processed_res)
        
        if len(exchanged_res) < best_len:
            best_len = len(exchanged_res)
            best_blocks = exchanged_res
            
        # Check if we reached the theoretical lower bound or a known good bound?
        # Lower bound for C(24,7,3) is 69.
        # If we hit 69, we can stop.
        if best_len <= 69:
            break

    # If we have time, try more random seeds
    if time_ok():
        while time_ok():
            seed = random.randint(0, 1000000)
            greedy_res = run_greedy(seed)
            if not greedy_res:
                continue
                
            processed_res = try_remove_blocks(greedy_res)
            exchanged_res = one_exchange_search(processed_res)
            
            if len(exchanged_res) < best_len:
                best_len = len(exchanged_res)
                best_blocks = exchanged_res
                
            if best_len <= 69:
                break

    if best_blocks is None:
        # Fallback: return an empty list or a single block? 
        # The problem requires a valid covering. 
        # If greedy failed (shouldn't happen for these params), we need a fallback.
        # But for v=24, k=7, t=3, greedy will always find a solution.
        return []
        
    # Convert block indices back to the actual blocks (lists of points)
    # The evaluator expects a list of blocks, where each block is a list of k integers.
    # all_blocks is a list of tuples.
    
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    # Ensure distinctness (should be guaranteed by indices) and sorting
    for b in final_blocks:
        b.sort()
        
    return final_blocks
