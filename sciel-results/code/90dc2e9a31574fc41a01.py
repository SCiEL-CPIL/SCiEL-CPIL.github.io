import itertools
import random

def construct(v, k, t):
    # Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block masks and block-triple lists
    block_masks = [0] * num_blocks
    block_triples = [[] for _ in range(num_blocks)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        triples_in_block = []
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
            triples_in_block.append(idx)
        block_masks[b_idx] = mask
        block_triples[b_idx] = triples_in_block
        
    # Precompute which blocks contain each triple (for swap heuristic)
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx in range(num_blocks):
        for t_idx in block_triples[b_idx]:
            triple_blocks[t_idx].append(b_idx)

    def greedy_construct():
        uncovered_mask = (1 << num_triples) - 1
        blocks = []
        while uncovered_mask:
            best_b_idx = -1
            max_covered = -1
            # Scan all blocks to find the one covering most uncovered triples
            for b_idx in range(num_blocks):
                # If this block covers no uncovered triples, skip
                # (block_masks[b_idx] & uncovered_mask) will be 0
                covered = block_masks[b_idx] & uncovered_mask
                if covered:
                    count = covered.bit_count()
                    if count > max_covered:
                        max_covered = count
                        best_b_idx = b_idx
            if best_b_idx == -1:
                break # Should not happen if valid input
            blocks.append(best_b_idx)
            uncovered_mask &= ~block_masks[best_b_idx]
        return blocks

    def local_search(blocks):
        # blocks is a list of block indices
        solution_set = set(blocks)
        
        # Initialize counts
        counts = [0] * num_triples
        for b_idx in blocks:
            for t_idx in block_triples[b_idx]:
                counts[t_idx] += 1
                
        improved = True
        # Multiple passes of removal and swap
        while improved:
            improved = False
            # Try removals
            new_blocks = []
            for b_idx in blocks:
                # Check if any triple in b_idx has count 1
                can_remove = True
                for t_idx in block_triples[b_idx]:
                    if counts[t_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove b_idx
                    for t_idx in block_triples[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
            
            # Try swaps
            for i, b_idx in enumerate(blocks):
                # Find unique triples for b_idx
                unique_triples = []
                for t_idx in block_triples[b_idx]:
                    if counts[t_idx] == 1:
                        unique_triples.append(t_idx)
                
                if not unique_triples:
                    continue # Should have been removed, but maybe counts changed?
                    
                # Try to find a replacement block b_new
                # Heuristic: pick a random unique triple to filter candidates
                # Or use the first one. Let's use the first one for determinism in this pass, 
                # or randomize for diversity.
                # We will try a few candidates from triple_blocks[unique_triples[0]]
                
                # To avoid modifying blocks while iterating, we just mark swap
                # We need to check if a swap is valid.
                # A swap is valid if b_new covers all unique_triples.
                # And b_new is not already in the solution.
                
                found_swap = False
                # Get candidates covering the first unique triple
                candidates = triple_blocks[unique_triples[0]]
                # Shuffle to avoid bias?
                # For speed, just iterate.
                for b_new in candidates:
                    if b_new in solution_set:
                        continue
                    
                    # Check if b_new covers all unique_triples
                    valid = True
                    # Optimization: if b_new doesn't cover t_idx, invalid
                    # We can check intersection of set(unique_triples) and set(block_triples[b_new])
                    # But sets are slow. 
                    # Since unique_triples is small (usually 1 or 2), we can just check each.
                    for t_idx in unique_triples:
                        # Is t_idx in block_triples[b_new]?
                        # block_triples[b_new] is a list. Checking membership is O(L).
                        # L is C(7,3)=35. So O(35) per triple.
                        # unique_triples size is small.
                        if t_idx not in block_triples[b_new]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        # Remove b_idx
                        for t_idx in block_triples[b_idx]:
                            counts[t_idx] -= 1
                        # Add b_new
                        for t_idx in block_triples[b_new]:
                            counts[t_idx] += 1
                        
                        # Update solution
                        solution_set.remove(b_idx)
                        solution_set.add(b_new)
                        blocks[i] = b_new
                        improved = True
                        found_swap = True
                        break # Try next block in solution
                
                if found_swap:
                    # Counts have changed, restart swap loop for this block? 
                    # Or just continue to next block?
                    # Since counts changed, other blocks' unique_triples might change.
                    # It's safer to break out and restart the whole local_search loop?
                    # Or just continue, but be aware counts are stale for subsequent checks in this pass?
                    # The `improved` flag will trigger a new pass anyway.
                    # Let's break the inner loop and let the outer `while improved` handle it.
                    # Actually, breaking the `for i` loop is good.
                    break 

        return blocks

    # Run greedy
    best_blocks = greedy_construct()
    
    # Run local search
    # To ensure we find a good solution, we can run greedy with different random seeds?
    # Greedy is deterministic here. 
    # We can add some randomization to the greedy step: if multiple blocks have same max_covered, pick randomly.
    # But the current greedy picks the first one found.
    
    # Let's try a few random restarts of a randomized greedy
    best_solution = best_blocks
    best_size = len(best_solution)
    
    for run in range(20):
        random.seed(run)
        uncovered_mask = (1 << num_triples) - 1
        blocks = []
        while uncovered_mask:
            best_b_idx = -1
            max_covered = -1
            candidates = []
            for b_idx in range(num_blocks):
                covered = block_masks[b_idx] & uncovered_mask
                if covered:
                    count = covered.bit_count()
                    if count > max_covered:
                        max_covered = count
                        candidates = [b_idx]
                    elif count == max_covered:
                        candidates.append(b_idx)
            if not candidates:
                break
            chosen = random.choice(candidates)
            blocks.append(chosen)
            uncovered_mask &= ~block_masks[chosen]
        
        # Local search
        refined = local_search(blocks)
        if len(refined) < best_size:
            best_size = len(refined)
            best_solution = refined

    # Convert indices back to lists of points
    return [list(all_blocks[i]) for i in best_solution]
