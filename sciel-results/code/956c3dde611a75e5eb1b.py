import itertools
import random
import time
import sys

def construct(v, k, t):
    # Specific optimization for the known task parameters
    if v == 24 and k == 7 and t == 3:
        return _construct_24_7_3()
    else:
        # Fallback generic greedy
        return _construct_generic(v, k, t)

def _construct_generic(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        # Pick a random uncovered triple to start the block
        start_triple = random.choice(list(uncovered))
        block = list(start_triple)
        # Fill the rest of the block greedily
        while len(block) < k:
            def gain(x):
                if x in block:
                    return -1
                new_block = block + [x]
                return sum(1 for rest in itertools.combinations(new_block, t) if tuple(rest) in uncovered)
            # Find best x
            best_x = max((x for x in range(v) if x not in block), key=gain)
            block.append(best_x)
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks

def _construct_24_7_3():
    v, k, t = 24, 7, 3
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Precomputation
    # Total triples
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Generate all blocks (7-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block index to list of triple indices
    # This is memory intensive but fast. 
    # 346k blocks * 35 triples = 12M ints. ~50MB.
    block_triples = [None] * num_blocks
    for i, block in enumerate(all_blocks):
        tris = [triple_to_idx[tri] for tri in itertools.combinations(block, t)]
        block_triples[i] = tris
        
    # Map triple index to list of block indices
    # 2024 triples * 5985 blocks = 12M ints.
    triple_blocks = [[] for _ in range(num_triples)]
    for i, tris in enumerate(block_triples):
        for tri_idx in tris:
            triple_blocks[tri_idx].append(i)
            
    best_blocks = None
    best_size = float('inf')
    
    def greedy_cover(random_seed):
        random.seed(random_seed)
        uncovered = set(range(num_triples))
        blocks = []
        cover_count = [0] * num_triples
        
        while uncovered:
            # Pick a random uncovered triple
            # To be more efficient, we could pick the one with fewest options, 
            # but random is fast and works well with restarts.
            target_triple = random.choice(tuple(uncovered))
            
            # Find the block containing target_triple that covers the most uncovered triples
            candidate_blocks = triple_blocks[target_triple]
            best_block = -1
            max_gain = -1
            
            # Optimization: We don't need to check all 5985 if we find a good one?
            # No, we need the max to ensure quality.
            # But 5985 * 35 is ~200k ops.
            
            for b_idx in candidate_blocks:
                tris = block_triples[b_idx]
                gain = 0
                for tri_idx in tris:
                    if tri_idx in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    if gain == len(tris): # Perfect gain, can't do better
                        break
            
            if best_block == -1:
                # Should not happen if uncovered is not empty and data is consistent
                break
                
            blocks.append(all_blocks[best_block])
            
            # Update uncovered and cover_count
            for tri_idx in block_triples[best_block]:
                cover_count[tri_idx] += 1
                if cover_count[tri_idx] == 1:
                    uncovered.discard(tri_idx)
                    
        # Local Search: Remove redundant blocks
        # We can iterate until no more removals are possible
        changed = True
        while changed:
            changed = False
            # Try to remove any block
            # To be fast, iterate in random order or just standard
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                # Check if all triples in b are covered by other blocks
                redundant = True
                for tri_idx in block_triples[all_blocks.index(b)]: # This index lookup is SLOW.
                    # We need to map block tuple to index or store index in blocks list
                    pass
            # Re-implement remove phase with indices
            # Let's store block indices in the greedy phase
        return blocks

    # Re-implementing with block indices for speed
    
    def greedy_cover_idx(random_seed):
        random.seed(random_seed)
        uncovered = set(range(num_triples))
        block_indices = []
        cover_count = [0] * num_triples
        in_block = [False] * num_blocks # Not strictly needed if we just track indices
        
        while uncovered:
            target_triple = random.choice(tuple(uncovered))
            candidate_blocks = triple_blocks[target_triple]
            
            best_block = -1
            max_gain = -1
            
            # Sample if list is too long? 5985 is manageable.
            for b_idx in candidate_blocks:
                tris = block_triples[b_idx]
                gain = 0
                # Use local variable for speed
                unc = uncovered
                for tri_idx in tris:
                    if tri_idx in unc:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    if gain == len(tris):
                        break
            
            if best_block == -1:
                break
                
            block_indices.append(best_block)
            
            for tri_idx in block_triples[best_block]:
                cover_count[tri_idx] += 1
                if cover_count[tri_idx] == 1:
                    uncovered.discard(tri_idx)
                    
        # Local Search: Remove
        # Convert to set for O(1) lookup of block indices if needed, but we iterate list
        # We need to check redundancy.
        # Redundant if for all T in B, cover_count[T] > 1.
        
        # We maintain cover_count.
        # To remove block B:
        # 1. Check if all T in B have cover_count[T] > 1.
        # 2. If yes, decrement cover_count[T] for all T in B.
        # 3. Remove B from list.
        
        changed = True
        while changed:
            changed = False
            # Iterate backwards to avoid index shifting issues
            for i in range(len(block_indices) - 1, -1, -1):
                b_idx = block_indices[i]
                tris = block_triples[b_idx]
                redundant = True
                for tri_idx in tris:
                    if cover_count[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for tri_idx in tris:
                        cover_count[tri_idx] -= 1
                    block_indices.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        # Local Search: Swap
        # Try to swap a block with another to enable removal?
        # This is complex. Let's stick to Greedy + Remove for now, with many restarts.
        # Swap is often needed to escape local minima.
        
        # Simple Swap:
        # For each block B in set:
        #   Let Unique(B) be triples covered only by B.
        #   If Unique(B) is not empty:
        #      Find a block B' not in set such that Unique(B) is subset of B'.
        #      If found, swap B with B'. Then run Remove phase.
        #      If size decreased, keep. Else revert.
        
        # We'll implement a limited swap search.
        
        improved = True
        while improved:
            improved = False
            # Copy state for rollback
            original_indices = list(block_indices)
            original_cover_count = list(cover_count)
            
            for i in range(len(block_indices)):
                b_idx = block_indices[i]
                tris = block_triples[b_idx]
                
                # Find unique triples
                unique_tris = [tri for tri in tris if cover_count[tri] == 1]
                
                if not unique_tris:
                    # Should have been removed in previous phase
                    continue
                    
                # Find candidate replacement
                # Intersection of triple_blocks for all unique_tris
                # Start with list of blocks for first unique triple
                candidates = set(triple_blocks[unique_tris[0]])
                
                for u_tri in unique_tris[1:]:
                    candidates &= set(triple_blocks[u_tri])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                    
                # Remove current block from candidates (we don't want to swap with self)
                candidates.discard(b_idx)
                
                # Also, we shouldn't swap with a block already in the set (unless we allow duplicates? No, blocks must be distinct)
                # Actually, if we swap B with B', B' must not be in the set.
                current_set = set(block_indices)
                valid_candidates = [c for c in candidates if c not in current_set]
                
                if not valid_candidates:
                    continue
                
                # Pick a random valid candidate
                new_b_idx = random.choice(valid_candidates)
                
                # Perform swap
                # Remove b_idx
                for tri in tris:
                    cover_count[tri] -= 1
                block_indices[i] = new_b_idx
                # Add new_b_idx
                for tri in block_triples[new_b_idx]:
                    cover_count[tri] += 1
                    
                # Run Remove phase
                changed = True
                while changed:
                    changed = False
                    for j in range(len(block_indices) - 1, -1, -1):
                        b_j = block_indices[j]
                        tris_j = block_triples[b_j]
                        redundant = True
                        for tri_idx in tris_j:
                            if cover_count[tri_idx] == 1:
                                redundant = False
                                break
                        if redundant:
                            for tri_idx in tris_j:
                                cover_count[tri_idx] -= 1
                            block_indices.pop(j)
                            changed = True
                            break
                            
                # Check if size decreased
                if len(block_indices) < len(original_indices):
                    improved = True
                    # Keep the new state
                    # Do not break, continue trying to improve
                else:
                    # Revert
                    block_indices[:] = original_indices
                    cover_count[:] = original_cover_count
                    break # Move to next i? Or break entire while?
                    # If swap didn't help, try next block.
                    
        return block_indices

    # Main Loop
    # Run multiple iterations
    # Estimate time per iteration.
    # Greedy: ~80 steps * 200k ops = 16M ops.
    # Remove: ~80 * 35 * 80 (avg) = 224k ops? No, remove is fast.
    # Swap: Expensive.
    
    # Let's run until time limit.
    num_iterations = 0
    while time.time() - start_time < time_limit:
        idx = greedy_cover_idx(num_iterations)
        if len(idx) < best_size:
            best_size = len(idx)
            best_blocks = [all_blocks[i] for i in idx]
        num_iterations += 1
        # If we found the theoretical lower bound or known best, we could stop?
        # Lower bound is 69. Best known 76.
        # If we find 76, it's good. If we find <76, it's a breakthrough.
        
    if best_blocks is None:
        # Fallback if no time was spent or error
        return _construct_generic(v, k, t)
        
    return best_blocks

if __name__ == "__main__":
    # Test
    blocks = construct(24, 7, 3)
    print(f"Found {len(blocks)} blocks")
    # Verify
    covered = set()
    for b in blocks:
        for tri in itertools.combinations(b, 3):
            covered.add(tri)
    all_tris = set(itertools.combinations(range(24), 3))
    print(f"Covered {len(covered)} / {len(all_tris)}")
