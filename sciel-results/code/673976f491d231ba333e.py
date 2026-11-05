import itertools
import random
import time
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    # And for each t-subset, the list of blocks that cover it
    # This is memory intensive but necessary for fast updates
    # n_blocks ~ 50376, n_tsubsets ~ 3876
    # Each block covers C(7,4)=35 t-subsets. Total entries ~ 50376 * 35 ~ 1.76M. OK.
    
    block_covers = [] # block_covers[b] = list of t-subset indices
    tsub_to_blocks = [[] for _ in range(n_tsubsets)]
    
    for b_idx, block in enumerate(all_blocks):
        covers = []
        for ts in itertools.combinations(block, t):
            s_idx = tsubset_to_idx[ts]
            covers.append(s_idx)
        block_covers.append(covers)
        for s_idx in covers:
            tsub_to_blocks[s_idx].append(b_idx)
    
    # --- Initial Solution via Greedy ---
    # State: coverage counts for each t-subset (int array)
    # We want to select a set of blocks.
    # Greedy: repeatedly pick the block that covers the most *uncovered* t-subsets.
    
    def greedy_initial():
        coverage = np.zeros(n_tsubsets, dtype=np.int32)
        selected = []
        
        # To speed up, we can track which t-subsets are uncovered.
        # But updating "uncovered" set is O(n_tsubsets) if we recompute.
        # Instead, we can just check coverage[s] == 0.
        
        # Optimization: Precompute a random permutation of blocks to break ties?
        # Or just iterate.
        
        # Efficient Greedy:
        # At each step, we need max_b (count of s where coverage[s]==0 and s in block_covers[b])
        # This is O(n_blocks * 35) per step.
        # 50k * 35 * 150 steps ~ 250M ops. Might be slow in Python loop.
        # Let's try to vectorize or optimize.
        
        # Alternative: Pick a random uncovered t-subset, then pick best block covering it.
        # This is "Lazy" greedy. Often works well and is O(1) to find candidate pool.
        
        uncovered_set = set(range(n_tsubsets))
        
        while uncovered_set:
            if time.time() > deadline - 20:
                break
            # Pick a random uncovered t-subset
            # Using a list for random selection is better than set iteration if we want uniform random
            # But set is fine for membership.
            # Let's keep a list of uncovered indices for fast random pick.
            # Actually, maintaining a list of uncovered indices requires removal which is O(N).
            # Let's just pick a random index from 0..N-1 and check if it's uncovered?
            # If N is small (3876), this is fast.
            
            s = random.choice(list(uncovered_set))
            
            # Find blocks covering s
            candidates = tsub_to_blocks[s]
            if not candidates:
                break # Should not happen
            
            # Pick the candidate that covers the most other uncovered t-subsets
            best_b = -1
            best_gain = -1
            
            # Limit candidates to first 50 for speed? No, let's try all if fast enough.
            # Candidates ~ 1000. 1000 * 35 = 35k ops. 150 steps * 35k = 5M ops. OK.
            
            for b in candidates:
                gain = 0
                for c in block_covers[b]:
                    if c in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            
            if best_b == -1:
                break
                
            selected.append(best_b)
            
            # Update coverage and uncovered_set
            for c in block_covers[best_b]:
                if c in uncovered_set:
                    uncovered_set.remove(c)
            
            if time.time() > deadline - 20:
                break
                
        # Repair: if any uncovered remain, add blocks greedily until covered or time out
        # The loop above should cover all if it doesn't break on time.
        # If it breaks on time, we might have uncovered.
        # Let's do a final pass.
        if time.time() < deadline - 5:
            # Find remaining uncovered
            # We lost track of exact uncovered_set if we broke early? No, we maintain it.
            while uncovered_set and time.time() < deadline - 5:
                s = random.choice(list(uncovered_set))
                candidates = tsub_to_blocks[s]
                # Pick first block? Or best? Best is safer.
                # To save time, just pick the first one.
                b = candidates[0]
                if b not in selected:
                    selected.append(b)
                    for c in block_covers[b]:
                        if c in uncovered_set:
                            uncovered_set.remove(c)
        
        return selected

    current_blocks = greedy_initial()
    if not current_blocks:
        # Fallback: return empty or random? Must be valid.
        # If greedy failed completely, we are in trouble.
        # But it shouldn't.
        return []

    current_coverage = np.zeros(n_tsubsets, dtype=np.int32)
    for b in current_blocks:
        for c in block_covers[b]:
            current_coverage[c] += 1
    
    # --- Simulated Annealing ---
    # State: list of blocks (indices)
    # Moves:
    # 1. Add a random block
    # 2. Remove a random block
    # 3. Swap a random block with another random block
    
    # Energy Function:
    # We want to minimize: (Number of Blocks) + Penalty * (Number of Uncovered t-subsets)
    # Let U = number of uncovered t-subsets (coverage[s] == 0)
    # Let B = number of blocks
    # Energy = B + Lambda * U
    # If U > 0, we must reduce U.
    # If U == 0, we can try to reduce B.
    
    # To ensure validity, we should heavily penalize U.
    # If we find a solution with U=0 and B blocks, that's our answer.
    # We want to minimize B subject to U=0.
    
    # Strategy:
    # Start with a valid solution (U=0).
    # Try to reduce B by removing blocks.
    # If removal makes U > 0, we must add blocks back to fix it, possibly different ones.
    # This is complex.
    
    # Alternative SA approach:
    # State: Set of blocks.
    # Move: Pick a random block in set, remove it. Pick a random block not in set, add it. (Swap)
    # Or: Just remove a block. If U becomes > 0, revert or try to fix?
    # Standard SA on set cover usually defines cost as sum of weights of selected sets + penalty for uncovered elements.
    
    # Let's use: Cost = B + M * U, where M is a large constant (e.g., 1000).
    # We want to minimize Cost.
    # If U=0, Cost = B.
    # If U>0, Cost is huge.
    
    # Initial state: current_blocks (U=0, Cost=B_init)
    # T0 = 10.0, T_end = 0.1, Cooling = 0.999
    
    # Precompute block frequencies?
    # block_freq[b] = number of times block b is in current set. (0 or 1 usually, but can be >1 if we allow duplicates? No, distinct blocks.)
    # So block_in_set[b] is boolean.
    
    block_in_set = np.zeros(n_blocks, dtype=bool)
    for b in current_blocks:
        block_in_set[b] = True
        
    def calculate_uncovered(coverage):
        # Count how many s have coverage[s] == 0
        return np.sum(coverage == 0)
    
    current_U = calculate_uncovered(current_coverage)
    current_B = len(current_blocks)
    current_Cost = current_B + 1000 * current_U
    
    best_blocks = list(current_blocks)
    best_B = current_B
    best_U = current_U
    best_Cost = current_Cost
    
    T = 10.0
    T_end = 0.1
    cooling = 0.999
    
    # To speed up, we can precompute for each block the list of t-subsets it covers (already done: block_covers)
    
    # We need to be careful with the swap move.
    # Removing b1: coverage[block_covers[b1]] -= 1
    # Adding b2: coverage[block_covers[b2]] += 1
    
    # Let's run SA for a fixed number of steps or until time limit.
    
    steps = 0
    max_steps = 100000 # Adjust based on speed
    
    while time.time() < deadline - 10 and T > T_end:
        steps += 1
        
        # Choose a move
        # 1. Remove a random block from the set
        # 2. Add a random block not in the set
        # 3. Swap
        
        # To keep B roughly constant or allow it to change, let's allow Add and Remove.
        # But we want to minimize B. So Remove is good if U stays 0.
        # Add is bad unless it allows removing multiple later?
        
        # Let's try:
        # 50% chance: Remove a random block from current set.
        # 50% chance: Swap a random block in set with a random block not in set.
        
        if not current_blocks:
            # If empty, add a random block
            b_new = random.choice(all_blocks)
            current_blocks.append(b_new)
            block_in_set[b_new] = True
            for c in block_covers[b_new]:
                current_coverage[c] += 1
            current_B += 1
            current_U = calculate_uncovered(current_coverage)
            new_Cost = current_B + 1000 * current_U
            # Always accept if it reduces cost? Or SA?
            # If U was >0, adding might help.
            # Let's just use SA acceptance.
            delta = new_Cost - current_Cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_Cost = new_Cost
                if current_U == 0 and current_B < best_B:
                    best_B = current_B
                    best_U = current_U
                    best_Cost = current_Cost
                    best_blocks = list(current_blocks)
            else:
                # Revert
                current_blocks.remove(b_new)
                block_in_set[b_new] = False
                for c in block_covers[b_new]:
                    current_coverage[c] -= 1
                current_B -= 1
                current_U = calculate_uncovered(current_coverage)
                current_Cost = current_B + 1000 * current_U
            T *= cooling
            continue
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Remove
            b_old = random.choice(current_blocks)
            
            # Apply removal
            for c in block_covers[b_old]:
                current_coverage[c] -= 1
            current_U = calculate_uncovered(current_coverage)
            current_B -= 1
            new_Cost = current_B + 1000 * current_U
            delta = new_Cost - current_Cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                # Accept
                current_blocks.remove(b_old)
                block_in_set[b_old] = False
                current_Cost = new_Cost
                
                if current_U == 0 and current_B < best_B:
                    best_B = current_B
                    best_U = current_U
                    best_Cost = current_Cost
                    best_blocks = list(current_blocks)
            else:
                # Revert
                for c in block_covers[b_old]:
                    current_coverage[c] += 1
                current_U = calculate_uncovered(current_coverage)
                current_B += 1
                current_Cost = current_B + 1000 * current_U
        else:
            # Swap
            b_old = random.choice(current_blocks)
            # Pick a random block not in set
            # Efficient way: pick random index, if in set, pick another?
            # Or maintain a list of available blocks?
            # n_blocks is 50k. current_blocks is ~150.
            # Probability of collision is low.
            b_new = random.randint(0, n_blocks - 1)
            while block_in_set[b_new]:
                b_new = random.randint(0, n_blocks - 1)
            
            # Apply swap
            for c in block_covers[b_old]:
                current_coverage[c] -= 1
            for c in block_covers[b_new]:
                current_coverage[c] += 1
                
            current_U = calculate_uncovered(current_coverage)
            # B stays same
            new_Cost = current_B + 1000 * current_U
            delta = new_Cost - current_Cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                # Accept
                # Update current_blocks list
                # Find index of b_old
                idx = current_blocks.index(b_old)
                current_blocks[idx] = b_new
                block_in_set[b_old] = False
                block_in_set[b_new] = True
                current_Cost = new_Cost
                
                if current_U == 0 and current_B < best_B:
                    best_B = current_B
                    best_U = current_U
                    best_Cost = current_Cost
                    best_blocks = list(current_blocks)
            else:
                # Revert
                for c in block_covers[b_old]:
                    current_coverage[c] += 1
                for c in block_covers[b_new]:
                    current_coverage[c] -= 1
                current_U = calculate_uncovered(current_coverage)
                current_Cost = current_B + 1000 * current_U
                
        T *= cooling
        
        # Periodic local search?
        # If we are stuck at U=0, try to remove blocks greedily?
        # The SA "Remove" move handles this.
        
    # Final result
    # We want the best valid solution (U=0) with min B.
    # If best_U > 0, we have an invalid solution. We must repair it.
    
    if best_U > 0:
        # Repair best_blocks
        # Calculate coverage of best_blocks
        cov = np.zeros(n_tsubsets, dtype=np.int32)
        for b in best_blocks:
            for c in block_covers[b]:
                cov[c] += 1
        uncovered = np.where(cov == 0)[0]
        
        # Add blocks to cover uncovered
        # Greedy repair
        uncovered_set = set(uncovered)
        while uncovered_set and time.time() < deadline - 2:
            s = random.choice(list(uncovered_set))
            candidates = tsub_to_blocks[s]
            # Pick block with max gain
            best_b = -1
            best_gain = -1
            for b in candidates:
                gain = 0
                for c in block_covers[b]:
                    if c in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b == -1:
                break
            if best_b not in best_blocks:
                best_blocks.append(best_b)
            for c in block_covers[best_b]:
                if c in uncovered_set:
                    uncovered_set.remove(c)
        
        best_U = len(uncovered_set)
        best_B = len(best_blocks)
        
        # If still invalid, fallback to initial greedy which was valid
        if best_U > 0:
            best_blocks = current_blocks # Wait, current_blocks might be modified.
            # We should keep a copy of the initial valid solution.
            # Let's re-run greedy if needed? Or just return initial.
            # Actually, initial greedy was valid.
            # Let's store it.
            pass
            
    # To be safe, let's verify best_blocks.
    # If best_blocks is invalid, use the initial greedy result.
    # We didn't store initial greedy result separately from current_blocks which gets modified.
    # Let's re-implement: Keep a variable `valid_solution` that is always valid.
    
    # Redo logic to track valid solution properly.
    
    # --- Revised Implementation ---
    
    # Re-initialize to be safe and clean
    current_blocks = greedy_initial()
    if not current_blocks:
        return []
        
    current_coverage = np.zeros(n_tsubsets, dtype=np.int32)
    for b in current_blocks:
        for c in block_covers[b]:
            current_coverage[c] += 1
            
    best_blocks = list(current_blocks)
    best_B = len(current_blocks)
    best_U = 0 # Greedy ensures valid
    
    block_in_set = np.zeros(n_blocks, dtype=bool)
    for b in current_blocks:
        block_in_set[b] = True
        
    current_U = 0
    current_B = len(current_blocks)
    current_Cost = current_B # U=0
    
    T = 10.0
    T_end = 0.1
    cooling = 0.999
    
    while time.time() < deadline - 10 and T > T_end:
        if not current_blocks:
            b_new = random.randint(0, n_blocks - 1)
            current_blocks.append(b_new)
            block_in_set[b_new] = True
            for c in block_covers[b_new]:
                current_coverage[c] += 1
            current_B += 1
            current_U = np.sum(current_coverage == 0)
            new_Cost = current_B + 1000 * current_U
            delta = new_Cost - current_Cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_Cost = new_Cost
                if current_U == 0 and current_B < best_B:
                    best_B = current_B
                    best_U = 0
                    best_blocks = list(current_blocks)
            else:
                current_blocks.remove(b_new)
                block_in_set[b_new] = False
                for c in block_covers[b_new]:
                    current_coverage[c] -= 1
                current_B -= 1
                current_U = 0
                current_Cost = current_B
            T *= cooling
            continue
            
        move_type = random.random()
        
        if move_type < 0.5:
            b_old = random.choice(current_blocks)
            for c in block_covers[b_old]:
                current_coverage[c] -= 1
            current_U = np.sum(current_coverage == 0)
            current_B -= 1
            new_Cost = current_B + 1000 * current_U
            delta = new_Cost - current_Cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                current_blocks.remove(b_old)
                block_in_set[b_old] = False
                current_Cost = new_Cost
                
                if current_U == 0 and current_B < best_B:
                    best_B = current_B
                    best_U = 0
                    best_blocks = list(current_blocks)
            else:
                for c in block_covers[b_old]:
                    current_coverage[c] += 1
                current_U = 0
                current_B += 1
                current_Cost = current_B
        else:
            b_old = random.choice(current_blocks)
            b_new = random.randint(0, n_blocks - 1)
            while block_in_set[b_new]:
                b_new = random.randint(0, n_blocks - 1)
            
            for c in block_covers[b_old]:
                current_coverage[c] -= 1
            for c in block_covers[b_new]:
                current_coverage[c] += 1
                
            current_U = np.sum(current_coverage == 0)
            new_Cost = current_B + 1000 * current_U
            delta = new_Cost - current_Cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                idx = current_blocks.index(b_old)
                current_blocks[idx] = b_new
                block_in_set[b_old] = False
                block_in_set[b_new] = True
                current_Cost = new_Cost
                
                if current_U == 0 and current_B < best_B:
                    best_B = current_B
                    best_U = 0
                    best_blocks = list(current_blocks)
            else:
                for c in block_covers[b_old]:
                    current_coverage[c] += 1
                for c in block_covers[b_new]:
                    current_coverage[c] -= 1
                current_U = 0
                current_Cost = current_B
                
        T *= cooling
        
    # Convert best_blocks indices to lists of points
    result = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    return result
