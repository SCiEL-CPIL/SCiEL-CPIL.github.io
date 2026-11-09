import itertools
import random
import time
import sys
import numpy as np

_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Precompute block_triple_indices and triple_to_blocks
    block_triple_indices = [None] * n_blocks
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        indices = []
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            indices.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = indices
        
    # Build bitmask array for fast gain computation in greedy
    # Each block maps to a bitmask of which triples it covers
    n_words = (n_triples + 63) // 64
    block_bitmasks = np.zeros((n_blocks, n_words), dtype=np.uint64)
    for i in range(n_blocks):
        for idx in block_triple_indices[i]:
            word = idx >> 6
            bit = idx & 63
            block_bitmasks[i, word] |= (1 << bit)
    
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, block_bitmasks)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, seed):
    rng = random.Random(seed)
    
    # Use a set for uncovered triples for O(1) membership checks
    uncovered_set = set(range(n_triples))
    
    blocks_used = []
    used_blocks_set = set()
    
    # Precompute constraint (number of blocks covering each triple) for "Most Constrained" heuristic
    triple_constraints = [len(triple_to_blocks[i]) for i in range(n_triples)]
    
    # Get bitmask array from cache
    cache_key = (v, k, t)
    block_bitmasks = _CACHE[cache_key][5]
    n_words = block_bitmasks.shape[1]
    
    # Maintain a bitmask of uncovered triples
    uncovered_mask = np.zeros(n_words, dtype=np.uint64)
    for i in range(n_triples):
        word = i >> 6
        bit = i & 63
        uncovered_mask[word] |= (1 << bit)
    
    while uncovered_set:
        # Pick the uncovered triple with the minimum number of covering blocks
        min_count = float('inf')
        candidates_for_tri = []
        
        for tri in uncovered_set:
            count = triple_constraints[tri]
            if count < min_count:
                min_count = count
                candidates_for_tri = [tri]
            elif count == min_count:
                candidates_for_tri.append(tri)
                
        if not candidates_for_tri:
            break
            
        target_tri = candidates_for_tri[rng.randint(0, len(candidates_for_tri) - 1)]
        
        # Get all blocks covering this triple
        candidate_blocks = triple_to_blocks[target_tri]
        
        if not candidate_blocks:
            break
            
        # Evaluate gain for each candidate block using bitmask
        best_block = None
        max_gain = -1
        
        for b_idx in candidate_blocks:
            if b_idx in used_blocks_set:
                continue
                
            # Gain = number of set bits in (block_bitmask & uncovered_mask)
            bm = block_bitmasks[b_idx]
            gain = 0
            for w in range(n_words):
                gain += bin(bm[w] & uncovered_mask[w]).count('1')
                    
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
            elif gain == max_gain and best_block is not None:
                if rng.random() < 0.5:
                    best_block = b_idx
                    
        if best_block is None:
            break
            
        blocks_used.append(best_block)
        used_blocks_set.add(best_block)
        
        # Update uncovered_mask and uncovered_set
        bm = block_bitmasks[best_block]
        for w in range(n_words):
            to_clear = bm[w] & uncovered_mask[w]
            if to_clear:
                uncovered_mask[w] &= ~to_clear
                # Clear bits from uncovered_set
                for bit in range(64):
                    if to_clear & (1 << bit):
                        t_idx = (w << 6) | bit
                        if t_idx < n_triples:
                            uncovered_set.discard(t_idx)
                
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, deadline):
    current_blocks = list(initial_blocks)
    
    # Precompute coverage counts
    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    num_uncovered = sum(1 for c in triple_counts if c == 0)
    if num_uncovered != 0:
        return current_blocks
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    rng = random.Random(42)
    
    # Simulated annealing parameters
    temperature = 3.0
    cooling_rate = 0.995
    min_temperature = 0.05
    
    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Phase 1: Delete all redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_out = current_blocks[i]
                redundant = True
                for idx in block_triple_indices[b_out]:
                    if triple_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    del current_blocks[i]
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break
        
        if not current_blocks:
            break
            
        # Phase 2: Try moves with simulated annealing
        move_type = rng.random()
        
        if move_type < 0.7:
            # 1-exchange: remove one block, add one block
            pos = rng.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[pos]
            
            uncovered_triples = []
            for idx in block_triple_indices[b_out]:
                if triple_counts[idx] == 1:
                    uncovered_triples.append(idx)
                    
            if not uncovered_triples:
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                continue

            min_triple = min(uncovered_triples, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            
            for tri in uncovered_triples:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                candidate_list = list(candidates)
                rng.shuffle(candidate_list)
                
                # Try up to 3 candidates
                max_tries = min(3, len(candidate_list))
                best_candidate = None
                best_gain = -1
                
                for b_in in candidate_list[:max_tries]:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if triple_counts[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_in
                
                if best_candidate is not None:
                    # Compute quality: penalize unique coverage, reward redundancy
                    old_metric = 0
                    for idx in block_triple_indices[b_out]:
                        c = triple_counts[idx]
                        if c >= 2:
                            old_metric += 1
                        elif c == 1:
                            old_metric -= 0.5
                    for idx in block_triple_indices[best_candidate]:
                        if idx not in block_triple_indices[b_out]:
                            c = triple_counts[idx]
                            if c >= 2:
                                old_metric += 1
                            elif c == 1:
                                old_metric -= 0.5
                            
                    new_metric = 0
                    affected = set(block_triple_indices[b_out]) | set(block_triple_indices[best_candidate])
                    for idx in affected:
                        c = triple_counts[idx]
                        if idx in block_triple_indices[b_out]:
                            c -= 1
                        if idx in block_triple_indices[best_candidate]:
                            c += 1
                        if c >= 2:
                            new_metric += 1
                        elif c == 1:
                            new_metric -= 0.8 # Heavier penalty for unique coverage
                    
                    delta_quality = new_metric - old_metric
                    
                    if delta_quality >= 0 or rng.random() < np.exp(delta_quality / temperature):
                        del current_blocks[pos]
                        for idx in block_triple_indices[b_out]:
                            triple_counts[idx] -= 1
                        
                        current_blocks.append(best_candidate)
                        for idx in block_triple_indices[best_candidate]:
                            triple_counts[idx] += 1
            
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
        elif move_type < 0.9:
            # 2-exchange: remove two blocks, add one block
            if len(current_blocks) < 2:
                continue
                
            pos1 = rng.randint(0, len(current_blocks) - 1)
            pos2 = rng.randint(0, len(current_blocks) - 1)
            if pos1 == pos2:
                continue
                
            b_out1 = current_blocks[pos1]
            b_out2 = current_blocks[pos2]
            
            # Temporarily remove both
            temp_counts = list(triple_counts)
            for idx in block_triple_indices[b_out1]:
                temp_counts[idx] -= 1
            for idx in block_triple_indices[b_out2]:
                temp_counts[idx] -= 1
            
            # Find uncovered triples
            uncovered = [i for i in range(n_triples) if temp_counts[i] == 0]
            if not uncovered:
                # Both are redundant
                # Remove the one with lower index first to avoid issues
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                    b_out1, b_out2 = b_out2, b_out1
                del current_blocks[pos1]
                for idx in block_triple_indices[b_out1]:
                    triple_counts[idx] -= 1
                del current_blocks[pos2 - 1]
                for idx in block_triple_indices[b_out2]:
                    triple_counts[idx] -= 1
                continue
            
            # Find a block that covers all uncovered triples
            min_triple = min(uncovered, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            
            for tri in uncovered:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                candidate_list = list(candidates)
                rng.shuffle(candidate_list)
                
                max_tries = min(3, len(candidate_list))
                best_candidate = None
                best_gain = -1
                
                for b_in in candidate_list[:max_tries]:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if temp_counts[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_in
                
                if best_candidate is not None:
                    # This reduces block count by 1, so always accept
                    # Remove both blocks
                    if pos1 > pos2:
                        pos1, pos2 = pos2, pos1
                        b_out1, b_out2 = b_out2, b_out1
                    del current_blocks[pos1]
                    for idx in block_triple_indices[b_out1]:
                        triple_counts[idx] -= 1
                    del current_blocks[pos2 - 1]
                    for idx in block_triple_indices[b_out2]:
                        triple_counts[idx] -= 1
                    
                    current_blocks.append(best_candidate)
                    for idx in block_triple_indices[best_candidate]:
                        triple_counts[idx] += 1
                    
                    temperature *= cooling_rate
                    if temperature < min_temperature:
                        temperature = min_temperature
    
    return best_blocks

def _ilp_polish(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, deadline):
    """
    Uses a branch-and-bound heuristic guided by ILP relaxation insights.
    We try to remove blocks that are 'fractionally' less important in a covering sense.
    Specifically, we iterate through blocks and check if they can be swapped for a better set.
    This is essentially a more aggressive local search that mimics LP relaxation variables.
    """
    current_blocks = list(initial_blocks)
    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    if any(c == 0 for c in triple_counts):
        return current_blocks
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    rng = random.Random(123)
    
    # We will perform a series of "exchange" moves, but with a focus on global optimality hints.
    # We'll use a priority queue of potential removals based on 'slack' or 'redundancy'.
    
    while time.time() < deadline:
        # 1. Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_out = current_blocks[i]
                redundant = True
                for idx in block_triple_indices[b_out]:
                    if triple_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    del current_blocks[i]
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break
        
        if not current_blocks:
            break
            
        # 2. Try to find a better exchange using a more global heuristic
        # We select a block to remove. We prefer blocks with high "redundancy" (covered by many others).
        # Then we try to cover the unique triples of that block with a single new block.
        
        # Calculate "redundancy score" for each block: sum of 1/(count-1) for its triples? 
        # Or simply: prefer removing blocks that share many triples with other blocks.
        
        # Let's try a random block, but weighted by how "safe" it is to remove.
        # Safety = number of triples it covers that have count > 1.
        
        scores = []
        for i, b in enumerate(current_blocks):
            safe_triples = 0
            for idx in block_triple_indices[b]:
                if triple_counts[idx] > 1:
                    safe_triples += 1
            scores.append(safe_triples)
            
        # Pick a block with high score (safe to remove)
        # But also, we want to find a block that is "hard" to replace if we do remove it.
        # Actually, we want to remove a block and find a replacement.
        # If we remove a block with many unique triples, it's hard to replace.
        # So we should remove blocks with FEW unique triples.
        
        unique_counts = []
        for i, b in enumerate(current_blocks):
            unique = 0
            for idx in block_triple_indices[b]:
                if triple_counts[idx] == 1:
                    unique += 1
            unique_counts.append(unique)
            
        # Sort by unique count ascending, then randomize within top candidates
        indices = list(range(len(current_blocks)))
        indices.sort(key=lambda i: unique_counts[i])
        
        # Try the top 5 candidates for removal
        candidates_to_try = indices[:5]
        rng.shuffle(candidates_to_try)
        
        found_improvement = False
        for pos in candidates_to_try:
            if pos >= len(current_blocks):
                continue
                
            b_out = current_blocks[pos]
            
            # Identify triples that would become uncovered
            critical_triples = []
            for idx in block_triple_indices[b_out]:
                if triple_counts[idx] == 1:
                    critical_triples.append(idx)
                    
            if not critical_triples:
                # Redundant, remove it
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                found_improvement = True
                break
            
            # Find a block that covers all critical_triples
            # Start with the most constrained critical triple
            min_tri = min(critical_triples, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_tri])
            
            for tri in critical_triples:
                if tri == min_tri:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if not candidates:
                continue
                
            # Evaluate candidates
            best_candidate = None
            best_score = -1
            
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            
            for b_in in cand_list[:10]:
                # Score: how many other triples does it cover that are currently under-covered?
                # Or simply: does it cover any triple with count 0? (No, we checked critical)
                # Let's use: number of triples it covers with count == 1 (increases redundancy)
                score = 0
                for idx in block_triple_indices[b_in]:
                    if triple_counts[idx] == 1:
                        score += 1
                    elif triple_counts[idx] == 0:
                        score += 10 # Should not happen if valid
                    else:
                        score -= 0.1 # Slight penalty for over-covering? No, we want to minimize blocks.
                        # Actually, covering more is good for flexibility, but we want to keep counts low?
                        # Let's just prefer blocks that cover the critical ones and maybe some others.
                        
                if score > best_score:
                    best_score = score
                    best_candidate = b_in
                    
            if best_candidate is not None:
                # Perform exchange
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                current_blocks.append(best_candidate)
                for idx in block_triple_indices[best_candidate]:
                    triple_counts[idx] += 1
                found_improvement = True
                break
                
        if not found_improvement:
            # If no improvement found in this pass, break to avoid infinite loop
            # Or we could randomize more.
            break
            
    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, block_bitmasks = _precompute(v, k, t)
    
    best_blocks_global = None
    best_len_global = float('inf')
    
    seed = 0
    while time.time() < deadline - 10:
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        # Allocate time for greedy, local search, and ILP polish
        # Greedy is fast, Local Search is medium, ILP Polish is medium/slow
        # Let's give 50% to LS, 50% to Polish
        
        ls_deadline = time.time() + remaining * 0.5
        polish_deadline = deadline - 2
        
        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, seed)
        seed += 1
        
        if not greedy_blocks:
            continue
            
        ls_blocks = _local_search(
            v, k, t, 
            greedy_blocks, 
            block_triple_indices, 
            triple_to_blocks, 
            full_mask, 
            n_triples, 
            ls_deadline
        )
        
        # Polish the result of local search
        polished_blocks = _ilp_polish(
            v, k, t,
            ls_blocks,
            block_triple_indices,
            triple_to_blocks,
            full_mask,
            n_triples,
            polish_deadline
        )
        
        # Also try polishing the greedy result directly (sometimes helps)
        if time.time() < deadline - 2:
            polished_greedy = _ilp_polish(
                v, k, t,
                greedy_blocks,
                block_triple_indices,
                triple_to_blocks,
                full_mask,
                n_triples,
                deadline - 1
            )
            if len(polished_greedy) < len(polished_blocks):
                polished_blocks = polished_greedy
        
        if len(polished_blocks) < best_len_global:
            best_len_global = len(polished_blocks)
            best_blocks_global = polished_blocks

    if best_blocks_global is None:
        if time.time() < deadline - 2:
            greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, 999)
            best_blocks_global = greedy_blocks
        else:
            best_blocks_global = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, 42)

    result = [list(all_blocks[idx]) for idx in best_blocks_global]
    return result
