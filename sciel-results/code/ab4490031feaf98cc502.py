import itertools
import sys
import random
import math

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute coverage indices for each block
    # block_indices[i] is a list of indices of t-subsets covered by k_blocks[i]
    block_indices = []
    for block in k_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(sub_to_idx[t_sub])
        block_indices.append(indices)
        
    # 3. Initial Solution via Greedy Set Cover
    uncovered_mask = (1 << n_t_subs) - 1
    # For speed in greedy, we might use bitmasks, but we need indices for SA.
    # Let's stick to the greedy logic from before but store indices.
    
    # We need to find the block with max gain.
    # Gain = number of uncovered t-subs covered by block.
    # We can maintain a 'coverage_count' array for greedy too?
    # Or just use bitmasks for greedy, then convert to SA state.
    
    # Greedy Phase
    selected_indices = []
    uncovered = (1 << n_t_subs) - 1
    
    # Precompute masks for greedy
    block_masks = [0] * n_k_blocks
    for i, indices in enumerate(block_indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        block_masks[i] = mask
        
    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # Convert to SA State
    selected_set = set(selected_indices)
    coverage = [0] * n_t_subs
    for idx in selected_set:
        for t_idx in block_indices[idx]:
            coverage[t_idx] += 1
            
    # Calculate initial uncovered count (should be 0)
    current_uncovered = sum(1 for c in coverage if c == 0)
    current_blocks = len(selected_set)
    
    best_blocks = current_blocks
    best_solution = list(selected_set)
    
    # 4. Simulated Annealing
    # Parameters
    M = 1000  # Penalty for uncovered subsets
    T_start = 10.0
    T_end = 0.01
    steps = 50000
    
    # Precompute random block selection lists? No, just use random.choice.
    
    for step in range(steps):
        # Calculate temperature
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Choose a move
        move_type = random.random()
        if move_type < 0.5:
            # Remove
            if not selected_set:
                continue
            b_out = random.choice(list(selected_set))
            b_in = None
        elif move_type < 0.8:
            # Add
            if len(selected_set) == n_k_blocks:
                continue
            # Find a block not in selected_set
            # This is slow if we check every time.
            # Optimization: Maintain a list of available blocks?
            # Or just pick random and check.
            # Given n_k_blocks ~ 75k and selected ~ 300, probability of collision is low.
            b_in = random.randint(0, n_k_blocks - 1)
            if b_in in selected_set:
                # Retry or skip
                continue
            b_out = None
        else:
            # Swap
            if not selected_set or len(selected_set) == n_k_blocks:
                continue
            b_out = random.choice(list(selected_set))
            b_in = random.randint(0, n_k_blocks - 1)
            if b_in == b_out or b_in in selected_set:
                continue
        
        # Calculate Delta E
        delta_blocks = 0
        if b_out is not None: delta_blocks -= 1
        if b_in is not None: delta_blocks += 1
        
        delta_uncovered = 0
        
        # Simulate changes on coverage
        # We need to know how many t-subs go from 0->1 (decrease uncovered) or 1->0 (increase uncovered)
        
        if b_out is not None:
            for t_idx in block_indices[b_out]:
                if coverage[t_idx] == 1:
                    # Will become 0
                    delta_uncovered += 1
                # If > 1, becomes > 0, no change in uncovered count
                # If 0, impossible since it's in selected_set
        
        if b_in is not None:
            for t_idx in block_indices[b_in]:
                # Note: If b_out and b_in share t_idx, we need to be careful.
                # If t_idx is in both:
                #   coverage[t_idx] decreases by 1 (from b_out) and increases by 1 (from b_in). Net 0.
                #   However, the intermediate state matters if we check "was it 0 or 1".
                #   Let's handle intersections carefully.
                pass
        
        # Correct Delta Calculation for Swap/Add/Remove:
        # It is safer to just apply the move to a temporary copy of the affected coverage counts?
        # Or calculate logically:
        
        # Let's re-calculate delta_uncovered properly.
        # For Remove (b_out only):
        #   delta_uncovered = count(t in b_out | coverage[t] == 1)
        
        # For Add (b_in only):
        #   delta_uncovered = -count(t in b_in | coverage[t] == 0)
        
        # For Swap (b_out, b_in):
        #   Let A = set(t in b_out), B = set(t in b_in)
        #   For t in A \ B: coverage decreases by 1.
        #       If coverage[t] was 1, becomes 0. delta += 1.
        #   For t in B \ A: coverage increases by 1.
        #       If coverage[t] was 0, becomes 1. delta -= 1.
        #   For t in A \cap B: coverage unchanged.
        
        delta_uncovered = 0
        if b_out is not None and b_in is None:
            for t_idx in block_indices[b_out]:
                if coverage[t_idx] == 1:
                    delta_uncovered += 1
        elif b_out is None and b_in is not None:
            for t_idx in block_indices[b_in]:
                if coverage[t_idx] == 0:
                    delta_uncovered -= 1
        elif b_out is not None and b_in is not None:
            # Swap
            # To avoid O(56*56) intersection check, we can use sets or just iterate and mark.
            # Given small size, simple logic:
            # Create a set of t_idx in b_out
            out_set = set(block_indices[b_out])
            in_set = set(block_indices[b_in])
            
            # t in out but not in in
            for t_idx in out_set - in_set:
                if coverage[t_idx] == 1:
                    delta_uncovered += 1
                    
            # t in in but not in out
            for t_idx in in_set - out_set:
                if coverage[t_idx] == 0:
                    delta_uncovered -= 1
            
            # t in both: no change in coverage count, so no change in uncovered status.
            
        delta_E = M * delta_uncovered + delta_blocks
        
        # Acceptance
        if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
            # Apply Move
            if b_out is not None:
                selected_set.remove(b_out)
                for t_idx in block_indices[b_out]:
                    coverage[t_idx] -= 1
            
            if b_in is not None:
                selected_set.add(b_in)
                for t_idx in block_indices[b_in]:
                    coverage[t_idx] += 1
            
            # Update current state
            current_uncovered += delta_uncovered
            current_blocks += delta_blocks
            
            # Check if valid and better
            if current_uncovered == 0 and current_blocks < best_blocks:
                best_blocks = current_blocks
                best_solution = list(selected_set)
                
    # Final Greedy Reduction on best_solution to ensure no redundant blocks
    # Re-construct coverage for best_solution
    final_selected = set(best_solution)
    final_coverage = [0] * n_t_subs
    for idx in final_selected:
        for t_idx in block_indices[idx]:
            final_coverage[t_idx] += 1
            
    # Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for idx in list(final_selected):
            # Check if idx is redundant
            redundant = True
            for t_idx in block_indices[idx]:
                if final_coverage[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                final_selected.remove(idx)
                for t_idx in block_indices[idx]:
                    final_coverage[t_idx] -= 1
                changed = True
                
    return [list(k_blocks[i]) for i in final_selected]
