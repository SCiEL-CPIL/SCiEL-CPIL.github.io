import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_covers, n_t, deadline):
    """Remove redundant blocks from selected set."""
    if not selected:
        return selected
    
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
            
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            redundant = True
            for s in block_covers[i]:
                if final_counts[s] <= 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(i)
                for s in block_covers[i]:
                    final_counts[s] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    return selected

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=100):
    """Try to improve the solution by swapping blocks."""
    if not selected or len(selected) < 2:
        return selected
    
    # Build coverage counts
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    # Build set of selected blocks for quick lookup
    selected_set = set(selected)
    
    # Try multiple random restarts of local search
    best_selected = set(selected)
    best_size = len(best_selected)
    
    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break
        
        # Pick a random block from the current solution
        if not best_selected:
            break
        block_to_remove = random.choice(list(best_selected))
        
        # Find uncovered subsets if we remove this block
        uncovered_if_removed = []
        for s in block_covers[block_to_remove]:
            if counts[s] <= 1:
                uncovered_if_removed.append(s)
        
        if not uncovered_if_removed:
            # Block is redundant, remove it
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_size -= 1
            continue
        
        # Try to find a replacement block that covers all uncovered_if_removed
        best_replacement = None
        
        # Get candidate blocks from the most constrained uncovered subset
        min_count = float('inf')
        target_sub = -1
        for s in uncovered_if_removed:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            continue
        
        candidates = sub_to_blocks[target_sub]
        
        for b in candidates:
            if b in best_selected:
                continue
            
            # Check if this block covers all uncovered_if_removed
            covers_all = True
            for s in uncovered_if_removed:
                if s not in block_covers[b]:
                    covers_all = False
                    break
            
            if not covers_all:
                continue
            
            if best_replacement is None:
                best_replacement = b
        
        if best_replacement is not None:
            # Perform the swap
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_selected.add(best_replacement)
            for s in block_covers[best_replacement]:
                counts[s] += 1
    
    return best_selected

def _walksat_local_search(selected, block_covers, n_t, sub_to_blocks, deadline, 
                          max_steps=5000, penalty_weight=1.0):
    """WalkSAT-inspired local search for minimizing block count while maintaining coverage.
    
    State: which blocks are selected.
    Objective: minimize (uncovered_count * 100 + selected_count * penalty_weight)
    Move: flip a block (add or remove) if it improves or has a small chance of worsening.
    """
    if not selected:
        return selected
    
    n_blocks = len(block_covers)
    selected_set = set(selected)
    
    # Coverage counts
    counts = [0] * n_t
    for i in selected_set:
        for s in block_covers[i]:
            counts[s] += 1
    
    uncovered_count = sum(1 for c in counts if c == 0)
    current_cost = uncovered_count * 100 + len(selected_set) * penalty_weight
    
    best_selected = set(selected_set)
    best_cost = current_cost
    
    step = 0
    no_improve = 0
    max_no_improve = 200
    
    while step < max_steps and _deadline_ok(deadline):
        step += 1
        no_improve += 1
        
        # Choose a move: either remove a selected block or add an unselected block
        # Weight towards moves that might help
        if random.random() < 0.6 and selected_set:
            # Try removing a block
            block = random.choice(list(selected_set))
            # Check if removal keeps coverage valid
            would_uncover = 0
            for s in block_covers[block]:
                if counts[s] == 1:
                    would_uncover += 1
            
            if would_uncover == 0:
                # Safe removal
                selected_set.remove(block)
                for s in block_covers[block]:
                    counts[s] -= 1
                uncovered_count = sum(1 for c in counts if c == 0)
                current_cost = uncovered_count * 100 + len(selected_set) * penalty_weight
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_selected = set(selected_set)
                    no_improve = 0
            else:
                # Risky removal - only do with small probability (WalkSAT-style)
                if random.random() < 0.05:
                    selected_set.remove(block)
                    for s in block_covers[block]:
                        counts[s] -= 1
                    uncovered_count = sum(1 for c in counts if c == 0)
                    current_cost = uncovered_count * 100 + len(selected_set) * penalty_weight
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = set(selected_set)
                        no_improve = 0
                    else:
                        # Undo
                        selected_set.add(block)
                        for s in block_covers[block]:
                            counts[s] += 1
                        uncovered_count = sum(1 for c in counts if c == 0)
                        current_cost = uncovered_count * 100 + len(selected_set) * penalty_weight
        else:
            # Try adding a block - focus on blocks that cover uncovered subsets
            uncovered_subs = [s for s in range(n_t) if counts[s] == 0]
            if not uncovered_subs:
                # No uncovered subsets, try removing
                if selected_set:
                    block = random.choice(list(selected_set))
                    would_uncover = 0
                    for s in block_covers[block]:
                        if counts[s] == 1:
                            would_uncover += 1
                    if would_uncover == 0:
                        selected_set.remove(block)
                        for s in block_covers[block]:
                            counts[s] -= 1
                        uncovered_count = sum(1 for c in counts if c == 0)
                        current_cost = uncovered_count * 100 + len(selected_set) * penalty_weight
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_selected = set(selected_set)
                            no_improve = 0
                continue
            
            # Pick a random uncovered subset
            target_sub = random.choice(uncovered_subs)
            candidates = sub_to_blocks[target_sub]
            if not candidates:
                continue
            
            # Among candidates, pick one that covers the most uncovered subsets
            best_block = None
            best_gain = -1
            # Sample up to 20 candidates for speed
            sample_size = min(20, len(candidates))
            for b in random.sample(candidates, sample_size):
                gain = 0
                for s in block_covers[b]:
                    if counts[s] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is not None:
                selected_set.add(best_block)
                for s in block_covers[best_block]:
                    counts[s] += 1
                uncovered_count = sum(1 for c in counts if c == 0)
                current_cost = uncovered_count * 100 + len(selected_set) * penalty_weight
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_selected = set(selected_set)
                    no_improve = 0
        
        if no_improve >= max_no_improve:
            # Reset to best and try a different strategy
            selected_set = set(best_selected)
            counts = [0] * n_t
            for i in selected_set:
                for s in block_covers[i]:
                    counts[s] += 1
            uncovered_count = sum(1 for c in counts if c == 0)
            current_cost = uncovered_count * 100 + len(selected_set) * penalty_weight
            no_improve = 0
    
    # Final prune
    best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    return best_selected

def _ilp_cutting_plane(selected, block_covers, n_t, sub_to_blocks, deadline, max_rounds=50):
    """ILP-inspired cutting-plane heuristic: iteratively identify uncovered t-subsets
    and add blocks to cover them, while trying to minimize total block count.
    
    This models the problem as: minimize sum(x_i) subject to for each t-subset s,
    sum_{i: s in covers[i]} x_i >= 1. We use a greedy cutting-plane approach where
    we start with a feasible solution and try to remove blocks, adding cutting planes
    (constraints) when infeasibility is detected.
    """
    if not selected:
        return selected
    
    selected_set = set(selected)
    
    # Coverage counts
    counts = [0] * n_t
    for i in selected_set:
        for s in block_covers[i]:
            counts[s] += 1
    
    best_selected = set(selected_set)
    best_size = len(best_selected)
    
    for round_num in range(max_rounds):
        if not _deadline_ok(deadline):
            break
        
        # Try to remove each block one by one (in random order)
        blocks_to_try = list(selected_set)
        random.shuffle(blocks_to_try)
        
        for block in blocks_to_try:
            if not _deadline_ok(deadline):
                break
            
            if block not in selected_set:
                continue
            
            # Check if removing this block would leave any t-subset uncovered
            would_uncover = []
            for s in block_covers[block]:
                if counts[s] == 1:
                    would_uncover.append(s)
            
            if not would_uncover:
                # Safe removal
                selected_set.remove(block)
                for s in block_covers[block]:
                    counts[s] -= 1
                if len(selected_set) < best_size:
                    best_size = len(selected_set)
                    best_selected = set(selected_set)
            else:
                # Cutting plane: we need to find a replacement
                # Find a block that covers all would_uncover subsets
                best_replacement = None
                best_overlap = -1
                
                # Use the most constrained uncovered subset
                target_sub = would_uncover[0]
                candidates = sub_to_blocks[target_sub]
                
                for b in candidates:
                    if b in selected_set:
                        continue
                    
                    # Check coverage of all would_uncover subsets
                    covers_all = True
                    overlap = 0
                    for s in would_uncover:
                        if s in block_covers[b]:
                            overlap += 1
                        else:
                            covers_all = False
                            break
                    
                    if covers_all and overlap > best_overlap:
                        best_overlap = overlap
                        best_replacement = b
                    
                    if best_overlap == len(would_uncover):
                        break
                
                if best_replacement is not None:
                    # Swap
                    selected_set.remove(block)
                    for s in block_covers[block]:
                        counts[s] -= 1
                    selected_set.add(best_replacement)
                    for s in block_covers[best_replacement]:
                        counts[s] += 1
                    if len(selected_set) < best_size:
                        best_size = len(selected_set)
                        best_selected = set(selected_set)
    
    return best_selected

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy: at each step, pick randomly among top-k candidates
        uncovered = set(range(n_t))
        selected = set()
        
        while uncovered:
            if not _deadline_ok(deadline):
                break
            
            # Find the uncovered subset with fewest candidate blocks
            min_count = float('inf')
            target_sub = -1
            for s in uncovered:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
            
            if target_sub == -1:
                break
            
            candidates = sub_to_blocks[target_sub]
            
            # Calculate gains for all candidates
            gains = []
            for b in candidates:
                gain = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        gain += 1
                gains.append((gain, b))
            
            # Sort by gain descending
            gains.sort(reverse=True)
            
            # Pick randomly among top candidates (top 10 or all if fewer)
            top_k = min(10, len(gains))
            if restart == 0:
                # First restart: deterministic greedy
                best_block = gains[0][1]
            else:
                # Randomized: pick from top-k
                best_block = random.choice(gains[:top_k])[1]
            
            selected.add(best_block)
            
            for s in block_covers[best_block]:
                uncovered.discard(s)
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline)
        
        # WalkSAT-inspired local search
        if _deadline_ok(deadline):
            selected = _walksat_local_search(selected, block_covers, n_t, sub_to_blocks, 
                                             deadline, max_steps=3000)
        
        # ILP cutting-plane approach
        if _deadline_ok(deadline):
            selected = _ilp_cutting_plane(selected, block_covers, n_t, sub_to_blocks, deadline)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _walksat_local_search(best_selected, block_covers, n_t, 
                                                  sub_to_blocks, deadline, max_steps=2000)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
            # Also try ILP on best
            if _deadline_ok(deadline):
                best_selected = _ilp_cutting_plane(best_selected, block_covers, n_t, sub_to_blocks, deadline)
                if len(best_selected) < best_size:
                    best_size = len(best_selected)
    
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
