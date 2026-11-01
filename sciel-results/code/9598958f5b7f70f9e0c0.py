import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # Map t-subset to index for quick lookup? 
    # Actually, we just need to check membership in the uncovered set.
    # So we can just keep the tuples.
    
    # Precompute the list of t-subsets for each k-subset
    k_sub_t_sets = []
    for ksub in all_k_subs:
        k_sub_t_sets.append(list(itertools.combinations(ksub, t)))
        
    # Initial state: all t-subsets are uncovered
    uncovered = set(all_t_subs)
    
    # Greedy Set Cover
    selected = []
    selected_set = set() # indices of selected k-subs
    
    while uncovered:
        best_idx = -1
        best_gain = 0
        
        for i in range(n_k_subs):
            if i in selected_set:
                continue
            
            # Calculate gain: how many uncovered t-subs does this k-sub cover?
            # k_sub_t_sets[i] is a list of tuples
            gain = 0
            for tsub in k_sub_t_sets[i]:
                if tsub in uncovered:
                    gain += 1
                    if gain <= best_gain:
                        # Optimization: if we already have a better gain, we can stop counting early?
                        # No, we need to know if it's strictly better.
                        # But if gain is already <= best_gain, it can't be the best.
                        # Wait, if gain is 0, we stop.
                        # If gain is 1 and best_gain is 5, we stop.
                        pass
            
            # Corrected early stopping logic
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update uncovered
        for tsub in k_sub_t_sets[best_idx]:
            if tsub in uncovered:
                uncovered.remove(tsub)
                
    current_blocks = selected[:]
    
    # Local Search: Try to remove redundant blocks
    # This is O(N^2 * 35) which is 150^2 * 35 ~ 800k ops. Fast.
    improved = True
    while improved:
        improved = False
        for i in range(len(current_blocks)):
            # Check if block i is redundant
            # Compute union of all other blocks
            # Instead of union, check if any t-sub is ONLY covered by block i.
            # If no t-sub is uniquely covered by block i, it's redundant.
            
            # Efficient check:
            # For each t-sub in block i, check if it is covered by any other block.
            # If all t-subs in block i are covered by others, remove it.
            
            is_redundant = True
            for tsub in k_sub_t_sets[current_blocks[i]]:
                covered_by_other = False
                for j in range(len(current_blocks)):
                    if i == j:
                        continue
                    if tsub in k_sub_t_sets[current_blocks[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    is_redundant = False
                    break
            
            if is_redundant:
                current_blocks.pop(i)
                improved = True
                break # Restart loop as list changed
                
    # Local Search: Try swaps
    # For each block i, try to replace it with a block j not in current_blocks
    # such that the new set still covers everything.
    # This is harder to check efficiently.
    # Heuristic: Only try swaps if the block has "low" unique coverage.
    
    # Given time constraints, the greedy + redundancy removal might be sufficient to beat the seed.
    # The seed likely did a simpler greedy or no local search.
    # The current program (score 1.1071) does bitmask greedy + local search.
    # My new approach is Set-based Greedy + Local Search.
    # It should be comparable in quality but potentially faster, allowing for more local search iterations or a better greedy.
    
    # Let's add a simple swap phase to improve further.
    # We will try to replace a block with another if the new block covers all the "unique" t-subs of the old block.
    
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 100.0
    
    # To make swaps faster, we can precompute for each t-sub, which blocks cover it.
    # But that's memory intensive.
    
    # Let's stick to the redundancy removal for now, as it's very effective.
    # The greedy algorithm usually leaves some redundancy.
    
    return [all_k_subs[i] for i in current_blocks]
