"""New program using Iterated Local Search for Covering Design."""
import itertools
import random
import numpy as np
from collections import defaultdict
from math import comb
import time

def construct(v, k, t):
    # Generate all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map t-subsets to indices for coverage tracking
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute: for each block, which t-subsets it covers
    block_to_tsubs = []
    # For greedy init
    tsub_to_blocks = defaultdict(list)
    
    for b_idx, block in enumerate(all_blocks):
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            tsub_to_blocks[idx].append(b_idx)
            tsubs_in_block.append(idx)
        block_to_tsubs.append(tsubs_in_block)
        
    def greedy_init():
        """Run standard greedy to get a starting solution."""
        covered = np.zeros(n_tsubs, dtype=bool)
        gain = np.full(n_blocks, comb(k, t), dtype=int)
        selected = []
        
        while True:
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            b_idx = candidates[0]
            selected.append(b_idx)
            
            for ts_idx in block_to_tsubs[b_idx]:
                if not covered[ts_idx]:
                    covered[ts_idx] = True
                    for other_b in tsub_to_blocks[ts_idx]:
                        gain[other_b] -= 1
        return selected

    # Initial solution
    initial_selected = greedy_init()
    in_solution = np.zeros(n_blocks, dtype=bool)
    for b in initial_selected:
        in_solution[b] = True
        
    # Calculate initial cover counts
    cover_count = np.zeros(n_tsubs, dtype=int)
    for b in initial_selected:
        for ts_idx in block_to_tsubs[b]:
            cover_count[ts_idx] += 1
            
    best_len = len(initial_selected)
    best_solution = initial_selected.copy()
    
    start_time = time.time()
    time_limit = 110
    rng = random.Random(42)
    
    def try_remove_redundant(selected_list, in_solution, cover_count):
        """Remove any block that is not the unique cover for any t-subset."""
        i = 0
        removed = False
        while i < len(selected_list):
            b_idx = selected_list[i]
            is_redundant = True
            for ts_idx in block_to_tsubs[b_idx]:
                if cover_count[ts_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                in_solution[b_idx] = False
                for ts_idx in block_to_tsubs[b_idx]:
                    cover_count[ts_idx] -= 1
                selected_list.pop(i)
                removed = True
                # Do not increment i
            else:
                i += 1
        return removed

    def local_search(selected_list, in_solution, cover_count):
        """Perform local search with 1-1 swaps to escape local minima."""
        improved = True
        while improved:
            improved = try_remove_redundant(selected_list, in_solution, cover_count)
            if not improved:
                # Stuck. Try a 1-1 swap.
                # Pick a random block to remove
                if not selected_list:
                    break
                b_out = rng.choice(selected_list)
                
                # Find t-subsets that would become uncovered
                critical_tsubs = []
                for ts_idx in block_to_tsubs[b_out]:
                    if cover_count[ts_idx] == 1:
                        critical_tsubs.append(ts_idx)
                
                if not critical_tsubs:
                    # Should have been removed by try_remove_redundant
                    # Force removal if logic bug
                    in_solution[b_out] = False
                    for ts_idx in block_to_tsubs[b_out]:
                        cover_count[ts_idx] -= 1
                    selected_list.remove(b_out)
                    improved = True
                    continue
                
                # Find a block b_in that covers all critical_tsubs
                # To speed up, we can sample a few candidates
                found_swap = False
                # Try random candidates
                for _ in range(50):
                    b_in = rng.randint(0, n_blocks - 1)
                    if in_solution[b_in]:
                        continue
                    
                    # Check if b_in covers all critical_tsubs
                    valid = True
                    for ts_idx in critical_tsubs:
                        if ts_idx not in block_to_tsubs[b_in]: # This is O(1) if we use a set, but list is O(n). 
                            # Optimization: Precompute block_to_tsubs as sets? 
                            # Or just check membership. 56 items.
                            # Let's use a set for block_to_tsubs for faster lookup
                            pass 
                            
                    # Since block_to_tsubs is a list, 'in' is slow.
                    # Let's convert block_to_tsubs to sets for this check.
                    # We will do this conversion once.
                    
                # This inner loop is inefficient if we don't have sets.
                # Let's modify the precomputation to use sets for block_to_tsubs.
                
        return selected_list, in_solution, cover_count

    # Precomputation update: Use sets for block_to_tsubs for faster membership testing
    block_to_tsubs_sets = [set(lst) for lst in block_to_tsubs]
    
    def local_search_v2(selected_list, in_solution, cover_count):
        """Perform local search with 1-1 swaps."""
        max_swaps = 100
        swaps = 0
        while swaps < max_swaps:
            # Try to remove redundant
            removed = False
            i = 0
            while i < len(selected_list):
                b_idx = selected_list[i]
                is_redundant = True
                for ts_idx in block_to_tsubs[b_idx]:
                    if cover_count[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    in_solution[b_idx] = False
                    for ts_idx in block_to_tsubs[b_idx]:
                        cover_count[ts_idx] -= 1
                    selected_list.pop(i)
                    removed = True
                else:
                    i += 1
            
            if removed:
                swaps = 0 # Reset swap counter on success
                continue
            else:
                # Stuck, try swap
                if not selected_list:
                    break
                b_out = rng.choice(selected_list)
                critical_tsubs = [ts for ts in block_to_tsubs[b_out] if cover_count[ts] == 1]
                
                if not critical_tsubs:
                    # Shouldn't happen
                    break
                    
                found = False
                # Try to find a b_in covering critical_tsubs
                # Sample random b_in
                for _ in range(20):
                    b_in = rng.randint(0, n_blocks - 1)
                    if in_solution[b_in]:
                        continue
                    if block_to_tsubs_sets[b_in].issuperset(critical_tsubs):
                        # Perform swap
                        # Remove b_out
                        in_solution[b_out] = False
                        for ts_idx in block_to_tsubs[b_out]:
                            cover_count[ts_idx] -= 1
                        selected_list.remove(b_out)
                        
                        # Add b_in
                        in_solution[b_in] = True
                        for ts_idx in block_to_tsubs[b_in]:
                            cover_count[ts_idx] += 1
                        selected_list.append(b_in)
                        
                        found = True
                        break
                if not found:
                    break # Give up on this local search instance
                
                swaps += 1
        
        return selected_list, in_solution, cover_count

    # Main Loop: Iterated Local Search
    current_selected = initial_selected.copy()
    current_in_sol = in_solution.copy()
    current_cover = cover_count.copy()
    
    while time.time() - start_time < time_limit:
        # Run local search
        current_selected, current_in_sol, current_cover = local_search_v2(
            current_selected, current_in_sol, current_cover
        )
        
        if len(current_selected) < best_len:
            best_len = len(current_selected)
            best_solution = current_selected.copy()
            
        # Perturbation: Randomly remove a few blocks and re-add via greedy?
        # Or just restart from best?
        # Simple restart:
        if len(current_selected) > best_len:
            # Reset to best
            # Clear current
            for b in current_selected:
                current_in_sol[b] = False
                for ts_idx in block_to_tsubs[b]:
                    current_cover[ts_idx] -= 1
            current_selected = best_solution.copy()
            current_cover[:] = 0
            for b in best_solution:
                current_in_sol[b] = True
                for ts_idx in block_to_tsubs[b]:
                    current_cover[ts_idx] += 1
        else:
            # Keep current, maybe perturb slightly?
            # For now, just restart local search from current state
            pass

    return [list(all_blocks[i]) for i in best_solution]
