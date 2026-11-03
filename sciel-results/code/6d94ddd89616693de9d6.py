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

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=200):
    """Try to improve the solution by swapping blocks."""
    if not selected or len(selected) < 2:
        return selected
    
    # Build coverage counts
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    best_selected = set(selected)
    best_size = len(best_selected)
    
    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break
        
        if not best_selected:
            break
        
        # Try to remove a random block
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
        
        # Try candidates, prefer those that cover more uncovered subsets
        best_gain = -1
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
            
            # Count how many currently covered subsets this block would add
            gain = 0
            for s in block_covers[b]:
                if counts[s] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
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

def _walksat_cover(block_covers, n_t, sub_to_blocks, deadline, max_steps=50000):
    """
    WalkSAT-style local search for set cover.
    Objective: minimize number of selected blocks while covering all t-subsets.
    
    State: boolean vector of length n_blocks (selected or not).
    Constraints: for each t-subset s, at least one block covering s is selected.
    Objective: minimize sum of selected blocks.
    
    We use a weighted score: primary = number of uncovered subsets (must be 0),
    secondary = number of selected blocks (to minimize).
    """
    n_blocks = len(block_covers)
    
    # Initialize with greedy solution
    uncovered = set(range(n_t))
    selected = set()
    
    while uncovered:
        if not _deadline_ok(deadline):
            break
        
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
        
        gains = []
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += 1
            gains.append((gain, b))
        
        gains.sort(reverse=True)
        best_block = gains[0][1]
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            uncovered.discard(s)
    
    if not uncovered:
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
    
    if not _deadline_ok(deadline):
        return selected
    
    # Now do WalkSAT-style local search
    # Build coverage counts
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    # Precompute: for each block, which subsets it covers (as a set for fast lookup)
    # We already have block_covers as lists
    
    best_selected = set(selected)
    best_score = len(best_selected)  # primary: 0 uncovered, secondary: size
    
    # WalkSAT parameters
    p_flip = 0.45  # probability of flipping a random block (vs. best)
    
    for step in range(max_steps):
        if not _deadline_ok(deadline):
            break
        
        # Check if all covered
        uncovered_count = 0
        for s in range(n_t):
            if counts[s] == 0:
                uncovered_count += 1
                break
        
        if uncovered_count == 0:
            # All covered, try to reduce size
            current_size = len(best_selected)
            
            # Try removing a random block
            if current_size > 1:
                b = random.choice(list(best_selected))
                
                # Check if removal would cause uncovered
                would_uncover = False
                for s in block_covers[b]:
                    if counts[s] <= 1:
                        would_uncover = True
                        break
                
                if not would_uncover:
                    # Safe removal
                    best_selected.remove(b)
                    for s in block_covers[b]:
                        counts[s] -= 1
                    best_score = len(best_selected)
                    continue
            else:
                break
        
        # Not all covered or can't reduce - do a flip
        # Compute "score" for each potential flip
        # We want to find a flip that either:
        # 1. Reduces uncovered count (if any)
        # 2. Keeps uncovered count same but reduces selected count
        # 3. Keeps both same (neutral)
        
        # For efficiency, pick a random block to consider flipping
        # Or pick based on what would help most
        
        # Strategy: pick a random block from all blocks
        b = random.randrange(n_blocks)
        
        if b in best_selected:
            # Consider unselecting b
            # Would this cause any subset to become uncovered?
            would_uncover = []
            for s in block_covers[b]:
                if counts[s] <= 1:
                    would_uncover.append(s)
            
            if not would_uncover:
                # Safe to unselect
                best_selected.remove(b)
                for s in block_covers[b]:
                    counts[s] -= 1
                best_score = len(best_selected)
            else:
                # This would cause uncovered subsets.
                # With probability (1-p_flip), try to fix by selecting a block covering the most constrained uncovered subset
                if random.random() < (1 - p_flip):
                    # Find best block to add that covers would_uncover
                    # Most constrained subset in would_uncover
                    min_c = float('inf')
                    target = -1
                    for s in would_uncover:
                        c = len(sub_to_blocks[s])
                        if c < min_c:
                            min_c = c
                            target = s
                            if c == 1:
                                break
                    
                    if target != -1:
                        cands = sub_to_blocks[target]
                        # Pick candidate that covers most of would_uncover
                        best_cand = -1
                        best_cover = -1
                        for cb in cands:
                            if cb in best_selected:
                                continue
                            cov = 0
                            for s in block_covers[cb]:
                                if s in would_uncover:
                                    cov += 1
                            if cov > best_cover:
                                best_cover = cov
                                best_cand = cb
                        
                        if best_cand != -1 and best_cover > 0:
                            # Do the swap: remove b, add best_cand
                            best_selected.remove(b)
                            for s in block_covers[b]:
                                counts[s] -= 1
                            best_selected.add(best_cand)
                            for s in block_covers[best_cand]:
                                counts[s] += 1
                else:
                    # Random flip: just unselect b anyway (accepting temporary uncovered)
                    # Actually, let's not do this - it might diverge
                    # Instead, try a different block
                    pass
        else:
            # Consider selecting b
            # This might help cover some uncovered subsets
            helps = 0
            for s in block_covers[b]:
                if counts[s] == 0:
                    helps += 1
            
            if helps > 0:
                # Select b
                best_selected.add(b)
                for s in block_covers[b]:
                    counts[s] += 1
            else:
                # Selecting b doesn't help with uncovered, skip
                # With small probability, still select to explore
                if random.random() < 0.05:
                    best_selected.add(b)
                    for s in block_covers[b]:
                        counts[s] += 1
    
    # Final prune
    best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    return best_selected

def _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """Greedy set cover with optional randomization."""
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
        
        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            uncovered.discard(s)
    
    return selected

def _branch_and_bound(uncovered, block_covers, sub_to_blocks, best_size, deadline):
    """
    Branch and bound solver for Set Cover.
    Returns a set of block indices if a better solution is found, else None.
    """
    if not uncovered:
        return set()
    
    if not _deadline_ok(deadline):
        return None
    
    # Simple LB
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
        return None
        
    max_cov = 0
    for b in sub_to_blocks[target_sub]:
        cov = 0
        for s in block_covers[b]:
            if s in uncovered:
                cov += 1
        if cov > max_cov:
            max_cov = cov
            
    if max_cov == 0:
        return None
        
    lb = (len(uncovered) + max_cov - 1) // max_cov
    
    if lb >= best_size:
        return None
        
    # Branch on target_sub
    candidates = sub_to_blocks[target_sub]
    
    # Sort candidates by coverage descending to find good solutions earlier
    sorted_candidates = []
    for b in candidates:
        cov = 0
        for s in block_covers[b]:
            if s in uncovered:
                cov += 1
        sorted_candidates.append((cov, b))
    sorted_candidates.sort(reverse=True)
    
    for cov, b in sorted_candidates:
        if not _deadline_ok(deadline):
            return None
            
        # Try adding b
        new_uncovered = set(uncovered)
        for s in block_covers[b]:
            new_uncovered.discard(s)
            
        # Recurse
        result = _branch_and_bound(new_uncovered, block_covers, sub_to_blocks, best_size - 1, deadline)
        
        if result is not None:
            result.add(b)
            return result
            
    return None

def _polish_pass(selected, block_covers, n_t, sub_to_blocks, deadline, max_rounds=50):
    """
    High-precision polish: iteratively try to remove blocks, allowing temporary
    uncovered subsets that can be fixed by adding other blocks. Uses a soft
    penalty approach to escape local optima.
    """
    if not selected:
        return selected
    
    best_selected = set(selected)
    best_size = len(best_selected)
    
    # Build coverage counts
    counts = [0] * n_t
    for i in best_selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    for rnd in range(max_rounds):
        if not _deadline_ok(deadline):
            break
        
        improved = False
        
        # Try each block in random order
        for b in random.sample(list(best_selected), len(best_selected)):
            if not _deadline_ok(deadline):
                break
            
            # What would become uncovered if we remove b?
            would_uncover = []
            for s in block_covers[b]:
                if counts[s] <= 1:
                    would_uncover.append(s)
            
            if not would_uncover:
                # Safe removal
                best_selected.remove(b)
                for s in block_covers[b]:
                    counts[s] -= 1
                best_size -= 1
                improved = True
                continue
            
            # Try to find a replacement that covers would_uncover
            # Use the most constrained subset
            min_c = float('inf')
            target = -1
            for s in would_uncover:
                c = len(sub_to_blocks[s])
                if c < min_c:
                    min_c = c
                    target = s
                    if c == 1:
                        break
            
            if target == -1:
                continue
            
            cands = sub_to_blocks[target]
            best_cand = -1
            best_net_gain = -1
            
            for cb in cands:
                if cb in best_selected:
                    continue
                
                # Check coverage of would_uncover
                covers_all = True
                for s in would_uncover:
                    if s not in block_covers[cb]:
                        covers_all = False
                        break
                
                if not covers_all:
                    continue
                
                # Net gain: how many new subsets this block covers (currently count==0)
                new_cover = 0
                for s in block_covers[cb]:
                    if counts[s] == 0:
                        new_cover += 1
                
                # Prefer blocks that cover more currently-uncovered subsets
                if new_cover > best_net_gain:
                    best_net_gain = new_cover
                    best_cand = cb
            
            if best_cand != -1:
                # Perform swap
                best_selected.remove(b)
                for s in block_covers[b]:
                    counts[s] -= 1
                best_selected.add(best_cand)
                for s in block_covers[best_cand]:
                    counts[s] += 1
                # Size stays same, but we may have improved structure
                improved = True
        
        if not improved:
            break
    
    # Final prune
    best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
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
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy
        selected = _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, 
                                  randomize=(restart > 0), top_k=10)
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
                
    # 4. WalkSAT-style local search (Max-SAT formulation)
    if best_selected is not None and _deadline_ok(deadline):
        walksat_result = _walksat_cover(block_covers, n_t, sub_to_blocks, deadline, max_steps=30000)
        if walksat_result is not None and len(walksat_result) < best_size:
            best_size = len(walksat_result)
            best_selected = walksat_result
    
    # 5. Branch and Bound attempt
    if best_selected is not None and _deadline_ok(deadline):
        node_count = [0]
        max_nodes = 10000
        
        def _bndb(uncovered, best_limit):
            if not _deadline_ok(deadline) or node_count[0] > max_nodes:
                return None
            node_count[0] += 1
            
            if not uncovered:
                return set()
                
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
                return None
                
            max_cov = 0
            for b in sub_to_blocks[target_sub]:
                cov = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    
            if max_cov == 0:
                return None
                
            lb = (len(uncovered) + max_cov - 1) // max_cov
            
            if lb >= best_limit:
                return None
                
            candidates = sub_to_blocks[target_sub]
            
            sorted_candidates = []
            for b in candidates:
                cov = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        cov += 1
                sorted_candidates.append((cov, b))
            sorted_candidates.sort(reverse=True)
            
            for cov, b in sorted_candidates:
                if not _deadline_ok(deadline) or node_count[0] > max_nodes:
                    return None
                    
                new_uncovered = set(uncovered)
                for s in block_covers[b]:
                    new_uncovered.discard(s)
                
                result = _bndb(new_uncovered, best_limit - 1)
                
                if result is not None:
                    result.add(b)
                    return result
                    
            return None
            
        bnb_result = _bndb(set(range(n_t)), best_size)
        
        if bnb_result is not None and len(bnb_result) < best_size:
            best_selected = bnb_result
            best_size = len(best_selected)
            
    # 6. Final high-precision polish pass
    if best_selected is not None and _deadline_ok(deadline):
        polished = _polish_pass(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_rounds=50)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished
    
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
