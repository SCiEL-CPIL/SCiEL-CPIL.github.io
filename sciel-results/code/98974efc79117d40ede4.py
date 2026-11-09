import itertools
import random
import time
import sys

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Generate all triples
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Precompute block_triple_indices: list of triple indices for each block
    block_triple_indices = [None] * n_blocks
    
    # Precompute triple_to_blocks: list of block indices for each triple
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        indices = []
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            indices.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = indices
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, seed):
    rng = random.Random(seed)
    
    # Use a set for uncovered triples for O(1) membership checks
    uncovered_set = set(range(n_triples))
    
    blocks_used = []
    
    while uncovered_set:
        # Pick a random uncovered triple
        tri_list = list(uncovered_set)
        target_tri = tri_list[rng.randint(0, len(tri_list) - 1)]
        
        # Get all blocks covering this triple
        candidate_blocks = triple_to_blocks[target_tri]
        
        if not candidate_blocks:
            break # Should not happen
            
        # Evaluate gain for each candidate block
        best_block = None
        max_gain = -1
        
        for b_idx in candidate_blocks:
            gain = 0
            for t_idx in block_triple_indices[b_idx]:
                if t_idx in uncovered_set:
                    gain += 1
                    
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
            elif gain == max_gain and best_block is not None:
                # Tie-breaking: random
                if rng.random() < 0.5:
                    best_block = b_idx
                    
        if best_block is None:
            break
            
        blocks_used.append(best_block)
        
        # Update uncovered_set
        for t_idx in block_triple_indices[best_block]:
            if t_idx in uncovered_set:
                uncovered_set.remove(t_idx)
                
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, deadline):
    # State: list of block indices
    current_blocks = list(initial_blocks)
    
    # Precompute coverage counts
    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    # Verify initial validity
    num_uncovered = sum(1 for c in triple_counts if c == 0)
    if num_uncovered != 0:
        return current_blocks
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Try to delete
        if current_blocks:
            pos = rng.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[pos]
            
            # Check if b_out is redundant
            redundant = True
            for idx in block_triple_indices[b_out]:
                if triple_counts[idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                continue # Immediately try again
        
        # If no deletion possible, try a swap
        if not current_blocks:
            break
            
        pos = rng.randint(0, len(current_blocks) - 1)
        b_out = current_blocks[pos]
        
        # Determine which triples become uncovered if we remove b_out
        uncovered_triples = []
        for idx in block_triple_indices[b_out]:
            if triple_counts[idx] == 1:
                uncovered_triples.append(idx)
                
        if not uncovered_triples:
            # Redundant, delete
            del current_blocks[pos]
            for idx in block_triple_indices[b_out]:
                triple_counts[idx] -= 1
            continue

        # Find a replacement block that covers all uncovered_triples
        if uncovered_triples:
            # Start with the triple with fewest covering blocks
            min_triple = min(uncovered_triples, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            
            for tri in uncovered_triples:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            # Filter out blocks already in current_blocks
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                b_in = rng.choice(list(candidates))
                
                # Perform swap
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                    
                current_blocks.append(b_in)
                for idx in block_triple_indices[b_in]:
                    triple_counts[idx] += 1

    return best_blocks

def _try_pairwise_merge(blocks, block_triple_indices, triple_to_blocks, n_triples, deadline):
    """Try to replace two blocks with one that covers their union of unique triples."""
    current = list(blocks)
    n = len(current)
    
    # Build coverage counts
    triple_counts = [0] * n_triples
    for b in current:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                
                b1 = current[i]
                b2 = current[j]
                
                # Find triples uniquely covered by b1 or b2
                unique_to_merge = []
                for idx in block_triple_indices[b1]:
                    if triple_counts[idx] == 1:
                        unique_to_merge.append(idx)
                for idx in block_triple_indices[b2]:
                    if triple_counts[idx] == 1:
                        unique_to_merge.append(idx)
                
                if not unique_to_merge:
                    # Both are redundant? Just remove one
                    continue
                
                # Find blocks that cover all unique_to_merge
                min_tri = min(unique_to_merge, key=lambda x: len(triple_to_blocks[x]))
                candidates = set(triple_to_blocks[min_tri])
                for tri in unique_to_merge:
                    if tri == min_tri:
                        continue
                    candidates.intersection_update(triple_to_blocks[tri])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                current_set = set(current)
                candidates.difference_update(current_set)
                
                if not candidates:
                    continue
                
                # Pick a candidate that maximizes overlap with existing blocks
                best_cand = None
                best_overlap = -1
                for cand in candidates:
                    overlap = 0
                    for idx in block_triple_indices[cand]:
                        if triple_counts[idx] > 0:
                            overlap += 1
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_cand = cand
                
                if best_cand is None:
                    continue
                
                # Perform the merge: remove b1 and b2, add best_cand
                new_current = [x for x in current if x != b1 and x != b2]
                new_current.append(best_cand)
                
                # Verify validity
                new_counts = [0] * n_triples
                for b in new_current:
                    for idx in block_triple_indices[b]:
                        new_counts[idx] += 1
                
                if all(c > 0 for c in new_counts):
                    # Check that we actually reduced count
                    if len(new_current) < len(current):
                        current = new_current
                        improved = True
                        break  # Restart outer loop with new state
            
            if improved:
                break
    
    return current

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples = _precompute(v, k, t)
    
    best_blocks_global = None
    best_len_global = float('inf')
    
    seed = 0
    # Run multiple greedy + local search instances
    while time.time() < deadline - 20:  # Leave 20s for final polish
        # Greedy
        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, seed)
        seed += 1
        
        if not greedy_blocks:
            continue
            
        # Local Search
        ls_deadline = time.time() + 6
        
        ls_blocks = _local_search(
            v, k, t, 
            greedy_blocks, 
            block_triple_indices, 
            triple_to_blocks, 
            full_mask, 
            n_triples, 
            ls_deadline
        )
        
        if len(ls_blocks) < best_len_global:
            best_len_global = len(ls_blocks)
            best_blocks_global = ls_blocks
            
    # Final polish pass: try pairwise merges on the best solution
    if best_blocks_global is not None and time.time() < deadline - 2:
        polished = _try_pairwise_merge(
            best_blocks_global,
            block_triple_indices,
            triple_to_blocks,
            n_triples,
            deadline - 2
        )
        if len(polished) < best_len_global:
            best_len_global = len(polished)
            best_blocks_global = polished

    if best_blocks_global is None:
        # Fallback
        if time.time() < deadline - 2:
            greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, 999)
            best_blocks_global = greedy_blocks
        else:
            # Emergency: run a quick greedy
            best_blocks_global = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, 42)

    result = [list(all_blocks[idx]) for idx in best_blocks_global]
    return result
