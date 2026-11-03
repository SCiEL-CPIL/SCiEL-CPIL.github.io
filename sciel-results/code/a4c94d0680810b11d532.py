import itertools
import time
import random
import sys
import numpy as np
from math import comb

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

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=500):
    """Try to improve the solution by swapping blocks."""
    if not selected or len(selected) < 2:
        return selected
    
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    best_selected = set(selected)
    best_size = len(best_selected)
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break
        
        if not best_selected:
            break
        
        if no_improve_count > 100:
            break
        
        block_to_remove = random.choice(list(best_selected))
        
        uncovered_if_removed = []
        for s in block_covers[block_to_remove]:
            if counts[s] <= 1:
                uncovered_if_removed.append(s)
        
        if not uncovered_if_removed:
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_size -= 1
            no_improve_count = 0
            continue
        
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
            no_improve_count += 1
            continue
        
        candidates = sub_to_blocks[target_sub]
        
        best_gain = -1
        best_replacement = None
        for b in candidates:
            if b in best_selected:
                continue
            
            covers_all = True
            for s in uncovered_if_removed:
                if s not in block_covers[b]:
                    covers_all = False
                    break
            
            if not covers_all:
                continue
            
            gain = 0
            for s in block_covers[b]:
                if counts[s] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_replacement = b
        
        if best_replacement is not None:
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_selected.add(best_replacement)
            for s in block_covers[best_replacement]:
                counts[s] += 1
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_selected

def _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """Greedy set cover with optional randomization."""
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
        
        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            uncovered.discard(s)
    
    return selected

def _dual_weighted_greedy(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """
    Greedy set cover with dual variable updates (subgradient-inspired weighting).
    
    Instead of just picking the block that covers the most uncovered t-subsets,
    we maintain a dual weight for each t-subset that reflects how "deficient" it is.
    The score for a block is the sum of dual weights of uncovered t-subsets it covers.
    After each selection, we update dual weights to prioritize still-uncovered subsets.
    """
    uncovered = set(range(n_t))
    selected = set()
    
    # Dual variables: weight for each t-subset, initialized to 1
    dual_weights = np.ones(n_t, dtype=np.float64)
    
    while uncovered:
        if not _deadline_ok(deadline):
            break
        
        # Find the most constrained uncovered t-subset (fewest covering blocks)
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
        
        # Score each candidate by sum of dual weights of uncovered t-subsets it covers
        gains = []
        for b in candidates:
            gain = 0.0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += dual_weights[s]
            gains.append((gain, b))
        
        gains.sort(reverse=True)
        
        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        # Update uncovered set
        newly_covered = []
        for s in block_covers[best_block]:
            if s in uncovered:
                uncovered.discard(s)
                newly_covered.append(s)
        
        # Update dual weights: increase weights of still-uncovered subsets
        # This creates a feedback loop where harder-to-cover subsets get more weight
        for s in uncovered:
            dual_weights[s] *= 1.05  # Small multiplicative increase
        
        # Decay weights of covered subsets to 0
        for s in newly_covered:
            dual_weights[s] = 0.0
    
    return selected

def _alternating_greedy(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """
    Greedy set cover that alternates between two strategies:
    1. Pick the block covering the most uncovered t-subsets (standard greedy)
    2. Pick the block that covers the most "rare" uncovered t-subsets (those with fewest covering blocks)
    
    This helps escape local optima by sometimes prioritizing structural constraints
    over immediate coverage gain.
    """
    uncovered = set(range(n_t))
    selected = set()
    
    # Precompute rarity: number of blocks covering each t-subset
    rarity = np.array([len(sub_to_blocks[s]) for s in range(n_t)], dtype=np.int32)
    
    step = 0
    while uncovered:
        if not _deadline_ok(deadline):
            break
        
        # Alternate between strategies
        use_rarity = (step % 3 == 2)  # Every third step, use rarity-based selection
        
        if use_rarity:
            # Find uncovered t-subset with minimum rarity
            min_rarity = float('inf')
            target_sub = -1
            for s in uncovered:
                r = rarity[s]
                if r < min_rarity:
                    min_rarity = r
                    target_sub = s
                    if r == 1:
                        break
            
            if target_sub == -1:
                break
            
            candidates = sub_to_blocks[target_sub]
            
            # Among candidates, pick the one that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            for b in candidates:
                gain = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
        else:
            # Standard greedy: pick block covering most uncovered
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
            
            if not randomize:
                best_block = gains[0][1]
            else:
                top_k_actual = min(top_k, len(gains))
                best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            uncovered.discard(s)
        
        step += 1
    
    return selected

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

    # 3. Multiple greedy strategies with local search refinement
    best_selected = None
    best_size = float('inf')
    
    # Strategy 1: Standard greedy (deterministic)
    if _deadline_ok(deadline):
        selected = _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, randomize=False)
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=300)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
    
    # Strategy 2: Dual-weighted greedy
    if _deadline_ok(deadline):
        selected = _dual_weighted_greedy(block_covers, n_t, sub_to_blocks, deadline, randomize=False)
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=300)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
    
    # Strategy 3: Alternating greedy
    if _deadline_ok(deadline):
        selected = _alternating_greedy(block_covers, n_t, sub_to_blocks, deadline, randomize=False)
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=300)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
    
    # Randomized restarts with mixed strategies
    num_restarts = 8
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Alternate between strategies
        if restart % 3 == 0:
            selected = _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, randomize=True, top_k=15)
        elif restart % 3 == 1:
            selected = _dual_weighted_greedy(block_covers, n_t, sub_to_blocks, deadline, randomize=True, top_k=15)
        else:
            selected = _alternating_greedy(block_covers, n_t, sub_to_blocks, deadline, randomize=True, top_k=15)
        
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=250)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # Additional local search on best
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=150)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
    
    # 4. Final refinement passes
    if best_selected is not None and _deadline_ok(deadline):
        for _ in range(4):
            if not _deadline_ok(deadline):
                break
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=200)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
            best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
