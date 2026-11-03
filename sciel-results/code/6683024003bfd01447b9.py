import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Shape: (n_blocks, C(k, t))
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Precompute triple_to_blocks: for each triple, which blocks cover it?
    # This is useful for local search replacements.
    # Since n_blocks is large (C(24,7) = 346104), storing lists for 2024 triples is feasible.
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx].append(i)

    # Compute how many blocks cover each triple (for weighted greedy)
    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1
    # Weight: rare triples get higher weight
    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)
    
    def greedy_cover(random_tiebreak=False):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Weighted gain: sum of weights of uncovered triples in block
            gains = np.sum(uncovered[block_triples].astype(np.float64) * triple_weights[block_triples], axis=1)
            
            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        
        return selected
    
    def remove_redundant(block_indices):
        """Remove blocks that are not needed to cover all triples."""
        if len(block_indices) == 0:
            return block_indices
        
        # For each block, check if removing it leaves all triples covered
        # Use a coverage count approach
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                bi = result[i]
                # Check if any triple in this block would become uncovered
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                # Safe to remove
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break
        
        return result

    def local_search(block_indices, max_iterations=200):
        """
        Try to replace blocks with alternatives that maintain coverage.
        Uses triple_to_blocks for efficient candidate finding.
        """
        if len(block_indices) == 0:
            return block_indices
        
        # Build coverage count
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        selected_set = set(current)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Shuffle order to avoid bias? Or just iterate.
            # We look for a block that can be removed or swapped.
            
            # First, try simple removals again? remove_redundant does this, but maybe we can do it faster inline?
            # Let's focus on swaps.
            
            # Identify blocks with critical triples (coverage == 1)
            # If a block has no critical triples, it might be removable (handled by remove_redundant usually, but let's check).
            
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                
                # Find critical triples for this block
                critical_triples = block_triples[bi]
                critical_mask = coverage[critical_triples] <= 1
                n_critical = np.sum(critical_mask)
                
                if n_critical == 0:
                    # This block is redundant? 
                    # If we remove it, does it break anything?
                    # If n_critical == 0, all its triples are covered by >= 2 blocks.
                    # So we can just remove it.
                    coverage[critical_triples] -= 1
                    current.pop(i)
                    selected_set.remove(bi)
                    improved = True
                    break
                
                # If there are critical triples, we must find a replacement block B' 
                # that covers ALL critical_triples.
                # B' must be in the intersection of triple_to_blocks for each critical triple.
                
                critical_list = critical_triples[critical_mask].tolist()
                
                # Start with the list of blocks covering the first critical triple
                # Filter out the current block
                candidates = [b for b in triple_to_blocks[critical_list[0]] if b != bi]
                
                if not candidates:
                    continue
                
                # Intersect with lists for other critical triples
                # Optimization: If the list gets too large, we might just sample.
                # But usually, critical triples are rare, so the list should shrink.
                
                for ct in critical_list[1:]:
                    if not candidates:
                        break
                    # Intersect candidates with triple_to_blocks[ct]
                    # Use set for intersection if lists are large?
                    # triple_to_blocks[ct] is a list.
                    # candidates is a list.
                    # Let's use a set for candidates for O(1) lookup.
                    cand_set = set(candidates)
                    candidates = [b for b in triple_to_blocks[ct] if b in cand_set]
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # We have valid candidates. Pick one.
                # To be safe, pick the one that maximizes additional coverage?
                # Or just pick the first one.
                # Let's pick the one that covers the most *other* uncovered triples?
                # Actually, any valid swap maintains validity.
                # We might prefer a swap that allows future removals.
                # For now, pick the first candidate.
                new_bi = candidates[0]
                
                # Perform swap
                coverage[block_triples[bi]] -= 1
                coverage[block_triples[new_bi]] += 1
                current[i] = new_bi
                selected_set.remove(bi)
                selected_set.add(new_bi)
                improved = True
                break
                
            if not improved:
                # No simple swap found.
                # We could try more complex moves (2-swaps) or break.
                # Given time constraints, breaking is safer if we've done many iterations.
                # But maybe we should just break if no improvement in a full pass?
                # The loop above breaks on first improvement. 
                # If no improvement in a full pass, we are stuck in a local minimum for 1-swaps.
                break
        
        return current
    
    # First greedy pass
    selected = greedy_cover(random_tiebreak=False)
    selected = remove_redundant(selected)
    selected = local_search(selected, max_iterations=500)
    selected = remove_redundant(selected) # Cleanup after swaps
    
    best_solution = [list(blocks[i]) for i in selected]
    best_size = len(best_solution)
    
    # Multiple random restarts
    num_restarts = 10
    for _ in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        selected = greedy_cover(random_tiebreak=True)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=200)
        selected = remove_redundant(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]
    
    return best_solution
