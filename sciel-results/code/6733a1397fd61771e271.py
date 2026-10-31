import time
import random
import itertools
import numpy as np

_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_subsets = list(itertools.combinations(points, t))
    
    n_blocks = len(all_blocks)
    n_subsets = len(all_subsets)
    
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # For each block, list of subset indices it covers
    block_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_idx[s])
        block_subsets.append(subs)
        
    # For each subset, list of block indices that cover it
    blocks_covering_subset = [[] for _ in range(n_subsets)]
    for i, subs in enumerate(block_subsets):
        for s_idx in subs:
            blocks_covering_subset[s_idx].append(i)
            
    _CACHE[key] = (all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets)
    return _CACHE[key]

def _greedy_construct(v, k, t, deadline, rng, randomize=True):
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    uncovered = [True] * n_subsets
    counts = [0] * n_subsets
    selected = []
    selected_set = set()
    
    # Track how many uncovered subsets each block covers (for fast updates)
    # We'll recompute gain each iteration for simplicity, but with numpy acceleration
    
    while True:
        if not any(uncovered):
            break
        if time.time() >= deadline:
            break
        
        best_block = -1
        best_gain = -1
        best_tiebreak = -1
        
        # Compute gain for each block: number of currently uncovered subsets it covers
        # Use a simple loop but break early if we find a block covering all remaining
        remaining = sum(uncovered)
        
        for i in range(n_blocks):
            if i in selected_set:
                continue
            subs = block_subsets[i]
            gain = 0
            for s_idx in subs:
                if uncovered[s_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = i
                best_tiebreak = i
                if gain == remaining:
                    # Can't do better than covering all remaining
                    break
            elif gain == best_gain and gain > 0:
                # Tie-break: prefer block that covers more subsets with count 0 (new)
                # or just pick the one with smaller index for determinism in non-randomized
                if not randomize:
                    best_block = i
                    best_tiebreak = i
        
        if best_block == -1 or best_gain == 0:
            break
        
        if randomize and best_gain > 0:
            # With some probability, pick a random block among those with gain >= best_gain * 0.8
            # This introduces diversity
            if rng.random() < 0.15:
                # Pick a random block with gain >= 0.7 * best_gain
                threshold = int(best_gain * 0.7)
                candidates = []
                for i in range(n_blocks):
                    if i in selected_set:
                        continue
                    subs = block_subsets[i]
                    gain = 0
                    for s_idx in subs:
                        if uncovered[s_idx]:
                            gain += 1
                    if gain >= threshold:
                        candidates.append(i)
                if candidates:
                    best_block = rng.choice(candidates)
        
        selected.append(best_block)
        selected_set.add(best_block)
        
        for s_idx in block_subsets[best_block]:
            counts[s_idx] += 1
            if counts[s_idx] == 1:
                uncovered[s_idx] = False
                
    return selected, counts, selected_set

def _local_search(selected, counts, selected_set, block_subsets, blocks_covering_subset, n_subsets, deadline, rng):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks (blocks that don't cover any uniquely-covered subset)
        i = 0
        while i < len(selected):
            b = selected[i]
            is_redundant = True
            for s_idx in block_subsets[b]:
                if counts[s_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for s_idx in block_subsets[b]:
                    counts[s_idx] -= 1
                selected.pop(i)
                selected_set.remove(b)
                improved = True
            else:
                i += 1
                
        if time.time() >= deadline:
            break
            
        # Phase 2: Try to find a better replacement for each selected block
        # For each block A in selected, find if there's a block B not in selected
        # such that B covers all subsets that A uniquely covers, and B is "better"
        
        # To be efficient, we process blocks in order of "criticality" (most unique subsets first)
        # but that's expensive. Instead, we try a random subset of blocks each pass.
        
        n_selected = len(selected)
        if n_selected == 0:
            break
        
        # Try to improve by replacing blocks
        # We'll try to replace a random block with a better one
        max_attempts = min(50, n_selected)
        
        for _ in range(max_attempts):
            if time.time() >= deadline:
                break
            
            # Pick a random block to try to replace
            idx = rng.randrange(len(selected))
            A = selected[idx]
            
            # Find all subsets that A uniquely covers
            unique_subs = [s_idx for s_idx in block_subsets[A] if counts[s_idx] == 1]
            
            if not unique_subs:
                # A is redundant, should have been removed in phase 1
                # But just in case, remove it
                for s_idx in block_subsets[A]:
                    counts[s_idx] -= 1
                selected.pop(idx)
                selected_set.remove(A)
                improved = True
                break
            
            # Find candidate blocks B that cover all of unique_subs
            # Start with the rarest unique subset
            unique_subs_sorted = sorted(unique_subs, key=lambda s: len(blocks_covering_subset[s]))
            
            candidates = set(blocks_covering_subset[unique_subs_sorted[0]])
            for s_idx in unique_subs_sorted[1:]:
                candidates &= set(blocks_covering_subset[s_idx])
            
            # Remove currently selected blocks
            candidates -= selected_set
            
            if not candidates:
                continue
            
            # Score each candidate B
            # We want B to:
            # 1. Cover all unique subsets of A (guaranteed by construction)
            # 2. Cover as many other subsets as possible (especially those with low count)
            # 3. Not create too much redundancy
            
            best_B = -1
            best_score = -1
            
            for B in candidates:
                score = 0
                for s_idx in block_subsets[B]:
                    c = counts[s_idx]
                    if c == 0:
                        score += 10
                    elif c == 1:
                        score += 5
                    elif c == 2:
                        score += 2
                    else:
                        score -= 1
                # Bonus if B covers fewer subsets that are already well-covered
                # Penalty for blocks that overlap too much with existing selection
                if score > best_score:
                    best_score = score
                    best_B = B
            
            if best_B == -1:
                continue
            
            # Check: does replacing A with B actually help?
            # Compute the "net benefit": subsets that become uncovered vs subsets that gain coverage
            # Since B covers all unique subsets of A, no subset becomes uncovered.
            # The benefit is: B may cover additional subsets, reducing their count pressure.
            
            # Perform the swap
            for s_idx in block_subsets[A]:
                counts[s_idx] -= 1
            selected_set.remove(A)
            selected[idx] = best_B
            selected_set.add(best_B)
            for s_idx in block_subsets[best_B]:
                counts[s_idx] += 1
            
            improved = True
            
    return selected, counts, selected_set

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin (150s limit)
    
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    rng = random.Random(42)
    
    # Strategy: Run multiple randomized greedy constructions, keep the best,
    # then do extensive local search on the best solution.
    
    best_selected = None
    best_counts = None
    best_selected_set = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy constructions with increasing randomization
    # Use first 40% of time for greedy constructions
    
    greedy_deadline = t0 + (deadline - t0) * 0.40
    
    n_greedy = 5  # Number of greedy passes
    greedy_time_per = (greedy_deadline - t0) / n_greedy
    
    for g in range(n_greedy):
        g_deadline = t0 + (g + 1) * greedy_time_per
        # Increase randomization with each pass
        selected, counts, selected_set = _greedy_construct(v, k, t, g_deadline, rng, randomize=(g > 0))
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_selected = selected
            best_counts = counts
            best_selected_set = selected_set
    
    if best_selected is None:
        # Fallback: pure greedy
        selected, counts, selected_set = _greedy_construct(v, k, t, deadline - 5.0, rng, randomize=False)
        best_selected = selected
        best_counts = counts
        best_selected_set = selected_set
        best_len = len(selected)
    
    # Phase 2: Local search on the best solution
    # Use remaining time for local search
    ls_deadline = deadline - 5.0
    
    best_selected, best_counts, best_selected_set = _local_search(
        best_selected, best_counts, best_selected_set,
        block_subsets, blocks_covering_subset,
        n_subsets, ls_deadline, rng
    )
    
    # Phase 3: If time remains, try local search again (may find more improvements)
    if time.time() < deadline - 10.0:
        best_selected, best_counts, best_selected_set = _local_search(
            best_selected, best_counts, best_selected_set,
            block_subsets, blocks_covering_subset,
            n_subsets, deadline - 5.0, rng
        )
    
    # Final: one more pass of redundancy removal
    i = 0
    while i < len(best_selected):
        b = best_selected[i]
        is_redundant = True
        for s_idx in block_subsets[b]:
            if best_counts[s_idx] == 1:
                is_redundant = False
                break
        if is_redundant:
            for s_idx in block_subsets[b]:
                best_counts[s_idx] -= 1
            best_selected.pop(i)
            best_selected_set.remove(b)
        else:
            i += 1
    
    result = [list(all_blocks[i]) for i in best_selected]
    
    return result
