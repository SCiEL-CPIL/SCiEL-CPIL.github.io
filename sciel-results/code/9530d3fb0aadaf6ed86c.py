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
    used_blocks_set = set()
    
    # Precompute constraint (number of blocks covering each triple) for "Most Constrained" heuristic
    # This is static, so we can just use len(triple_to_blocks[i])
    # But we need to ensure we don't pick already used blocks?
    # The "Most Constrained" heuristic usually refers to the static problem structure.
    # We will pick the uncovered triple with the fewest *total* blocks covering it.
    # This forces us to use blocks that are "rare".
    
    while uncovered_set:
        # Pick the uncovered triple with the minimum number of covering blocks
        # If tie, pick randomly
        min_count = float('inf')
        candidates_for_tri = []
        
        # Optimization: If the set is large, iterating all is O(N).
        # N is ~2000. This is fast.
        for tri in uncovered_set:
            count = len(triple_to_blocks[tri])
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
            break # Should not happen
            
        # Evaluate gain for each candidate block
        # Gain = number of triples in this block that are currently uncovered
        best_block = None
        max_gain = -1
        
        for b_idx in candidate_blocks:
            if b_idx in used_blocks_set:
                continue
                
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
            # If all blocks covering the most constrained triple are already used,
            # we might be stuck or need to pick a different triple?
            # In a valid covering, this shouldn't happen if we haven't exhausted options.
            # But if we did, we break.
            break
            
        blocks_used.append(best_block)
        used_blocks_set.add(best_block)
        
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
            
        # Phase 1: Delete all redundant blocks
        changed = True
        while changed:
            changed = False
            # Iterate backwards to safely delete
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
                    break # Restart loop to check from beginning again, as counts changed
        
        if not current_blocks:
            break
            
        # Phase 2: Try a swap
        pos = rng.randint(0, len(current_blocks) - 1)
        b_out = current_blocks[pos]
        
        # Determine which triples become uncovered if we remove b_out
        uncovered_triples = []
        for idx in block_triple_indices[b_out]:
            if triple_counts[idx] == 1:
                uncovered_triples.append(idx)
                
        if not uncovered_triples:
            # Redundant, delete (should have been caught in Phase 1, but safe to handle)
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
                # Note: We don't update best_blocks here immediately, 
                # but the next Phase 1 might reduce it further.
                # We update best_blocks at the start of the next loop iteration.

    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin for safety
    
    all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples = _precompute(v, k, t)
    
    best_blocks_global = None
    best_len_global = float('inf')
    
    seed = 0
    # Run multiple greedy + local search instances
    # We allocate time for each run dynamically
    while time.time() < deadline - 5: # Leave 5s for final check
        # Estimate time for greedy (fast) and LS (slower)
        # Give LS a chunk of time, e.g., 5-10 seconds depending on remaining
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        ls_deadline = time.time() + min(8, remaining - 2)
        
        # Greedy
        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, seed)
        seed += 1
        
        if not greedy_blocks:
            continue
            
        # Local Search
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
            # If we found a very good solution, we might want to spend more time polishing it?
            # For now, we just keep trying to beat it.

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
