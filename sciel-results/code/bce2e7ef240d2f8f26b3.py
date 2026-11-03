import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)
    
    # C(k, t)
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] contains the indices of the t-subsets covered by blocks[i]
    # Shape: (n_blocks, c_k_t)
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    # Also compute frequency of each triple to use for weighting
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_freq[t_idx] += 1
            idx += 1
            
    # Weights: 1 / frequency. Higher weight for rarer triples.
    weights = 1.0 / np.maximum(triple_freq, 1)
    
    # Helper to check if a set of blocks covers all triples
    # Returns True if valid, False otherwise
    def is_valid_cover(block_indices):
        if not block_indices:
            return n_triples == 0
        covered = np.zeros(n_triples, dtype=bool)
        for b_idx in block_indices:
            covered[block_triples[b_idx]] = True
        return np.all(covered)
    
    def prune_redundant(block_indices):
        """Remove blocks that are completely redundant."""
        if not block_indices:
            return []
        
        # Count coverage for each triple
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_triples[b_idx]] += 1
            
        current_selected = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_selected)):
                b_idx = current_selected[i]
                # If all triples in this block are covered by at least 2 blocks, it's redundant
                # i.e., min cover count for its triples > 1
                if np.min(cover_count[block_triples[b_idx]]) > 1:
                    # Remove it
                    cover_count[block_triples[b_idx]] -= 1
                    current_selected.pop(i)
                    changed = True
                    break # Restart loop as indices/counts changed
                    
        return current_selected

    def local_search_2_exchange(block_indices, max_iters=50):
        """
        Try to improve the solution by replacing 1 block with 2 others.
        If we can replace 1 block with 2, the size increases by 1 (bad).
        If we can replace 2 blocks with 1, the size decreases by 1 (good).
        So we look for: Remove 2 blocks, Add 1 block.
        Condition: The new block must cover all triples that were ONLY covered by the 2 removed blocks.
        """
        current = list(block_indices)
        current_size = len(current)
        
        # Precompute for each triple, which blocks in 'current' cover it?
        # To make this efficient, we can maintain a count of how many blocks in 'current' cover each triple.
        # But checking "only covered by A and B" requires knowing the specific set.
        
        # Simplified 2-1 Exchange:
        # For each pair of blocks (A, B) in current:
        #   Find triples that are covered by A or B but NOT by any other block in current \ {A, B}.
        #   Let S be this set of "exclusive" triples.
        #   If S is empty, we can remove both A and B (size -2). This is handled by prune_redundant if they are redundant individually?
        #   No, prune_redundant removes one by one. If A and B are only redundant together, prune_redundant might not catch it immediately 
        #   unless one becomes redundant after the other is removed.
        #   Actually, if A covers triple t, and B covers triple t, and no other block covers t, then neither A nor B is redundant individually.
        #   But if we remove A, t is still covered by B. So A is redundant? No, "redundant" means removing it doesn't break coverage.
        #   If we remove A, t is covered by B. So A is redundant IF all its other triples are covered by others.
        #   
        #   The 2-1 exchange looks for a single block C (not in current) that covers all triples in S.
        #   If such C exists, we can replace {A, B} with {C}. Size changes from 2 to 1. Net -1.
        
        # This is computationally heavy if done naively.
        # Given time constraints, we will rely heavily on prune_redundant and randomized restarts.
        # A lighter version: Try to swap 1 block with 1 other block if it doesn't increase size (neutral) but helps future pruning?
        # Or just stick to strong pruning and more restarts.
        
        # Let's implement a simple "1-for-1" swap search to break ties or improve structure?
        # Actually, the most effective improvement for greedy set cover is often just **more diverse restarts** 
        # combined with **aggressive pruning**.
        
        return current

    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Standard Weighted Greedy
    def run_greedy(use_weighted=True, random_tie=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if use_weighted:
                # Weighted gain: sum of weights of uncovered triples in block
                # weights are precomputed. 
                # We need sum(weights[j] for j in block if uncovered[j])
                # This is sum(weights * uncovered) over the block's triples.
                # Using numpy: np.sum(uncovered[block_triples[i]] * weights[block_triples[i]])
                # Vectorized:
                gains = np.sum(uncovered[block_triples] * weights[block_triples], axis=1)
            else:
                # Unweighted gain: count of uncovered triples in block
                gains = np.sum(uncovered[block_triples], axis=1)
                
            if gains.max() == 0:
                break
                
            if random_tie:
                max_gain = gains.max()
                candidates = np.where(gains == max_gain)[0]
                best_idx = random.choice(candidates.tolist())
            else:
                best_idx = np.argmax(gains)
                
            selected_blocks.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected_blocks

    # Run initial deterministic strategies
    for strategy in [
        {'use_weighted': True, 'random_tie': False, 'seed': None},
        {'use_weighted': False, 'random_tie': False, 'seed': None},
    ]:
        if time.time() > deadline - 5.0:
            continue
            
        selected = run_greedy(**strategy)
        pruned = prune_redundant(selected)
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(blocks[i]) for i in pruned]
            
        if best_size <= 69:
            break

    # Run randomized restarts
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        seed = restart_count * 104729 + 1
        use_weighted = (restart_count % 2 == 0)
        
        selected = run_greedy(use_weighted=use_weighted, random_tie=True, seed=seed)
        pruned = prune_redundant(selected)
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(blocks[i]) for i in pruned]
            
        # If we find a very good solution, we might spend a bit more time trying to polish it
        # But for now, just keep looking for better ones.
        
        if best_size <= 69:
            break

    # Final Polish: If we have time, try to improve the best solution found
    if best_solution is not None and time.time() < deadline - 2.0:
        # Convert best_solution back to indices for manipulation
        # We need a way to map block lists back to indices quickly.
        # Since blocks are combinations, we can use a dictionary.
        block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
        
        current_indices = []
        for b in best_solution:
            current_indices.append(block_to_idx[tuple(b)])
            
        # Try a few rounds of random swaps or additions/removals?
        # A simple local search: try to remove a block, see if still valid.
        # This is essentially prune_redundant again, but maybe we can try adding a block 
        # that allows removing two others? (The 2-1 exchange mentioned earlier).
        
        # Let's try a limited 2-1 exchange search on the current best solution.
        # For each pair in current_indices:
        #   Identify triples only covered by this pair.
        #   Check if any single block covers all those triples.
        
        # This is O(N^2 * C(k,t)) per pair. N is ~70. Pairs ~2400. C(7,3)=35.
        # 2400 * 35 = 84,000 ops per pair check? No, checking coverage is faster.
        
        # Let's implement a fast check for 2-1 exchange.
        # 1. Compute cover_count for current_indices.
        # 2. For each pair (i, j) in current_indices:
        #    a. Temporarily decrement count for blocks i and j.
        #    b. Find triples where count == 0. These are the "exclusive" triples of {i, j} relative to the rest.
        #    c. If no such triples, we can remove both i and j. (Size -2). This should have been caught by prune_redundant?
        #       prune_redundant removes one if it's redundant. If i and j are mutually redundant but not individually, 
        #       prune_redundant might miss it if it checks sequentially. 
        #       Actually, if i is not redundant (has an exclusive triple t1) and j is not redundant (has exclusive triple t2),
        #       but t1 is covered by j and t2 is covered by i, then removing i leaves t1 covered by j. So i IS redundant?
        #       Yes, if all triples of i are covered by (current \ {i}), then i is redundant.
        #       So if t1 is covered by j, and j is in current \ {i}, then t1 is covered.
        #       So prune_redundant SHOULD catch mutually redundant pairs one by one.
        #       Therefore, if prune_redundant finished, no single block is redundant.
        #       This implies no pair is mutually redundant in a way that allows removal of one.
        #       But 2-1 exchange replaces {i, j} with {k}.
        #       This requires that all triples covered by i or j (but not by others) are covered by k.
        
        # Let's try this.
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current_indices:
            cover_count[block_triples[b_idx]] += 1
            
        improved = False
        # Iterate over pairs
        for i_idx in range(len(current_indices)):
            if time.time() > deadline - 2.0:
                break
            for j_idx in range(i_idx + 1, len(current_indices)):
                if time.time() > deadline - 2.0:
                    break
                
                b1 = current_indices[i_idx]
                b2 = current_indices[j_idx]
                
                # Identify triples that would become uncovered if we remove b1 and b2
                # These are triples where cover_count == 2 AND they are in b1 or b2.
                # Actually, we need to know which triples are covered by b1 or b2.
                # Let's just simulate removal.
                
                # To speed up, we can precompute the set of triples for each block.
                # But we have block_triples matrix.
                
                # Find unique triples of b1 and b2 that are NOT covered by any other block.
                # A triple is "critical" for {b1, b2} if cover_count[triple] == 2 and (triple in b1 or triple in b2).
                # Wait, if cover_count[triple] == 2, and one of the covering blocks is b1 or b2, 
                # then the other covering block must be in current_indices \ {b1, b2}.
                # So if we remove b1 and b2, that triple becomes uncovered (count 0).
                
                # We need to find if there is a block k (not in current_indices) that covers ALL such critical triples.
                
                # Step 1: Find critical triples.
                # This is expensive to do for every pair.
                # Alternative: For each block k not in current, check if it can replace a pair.
                # Too slow.
                
                # Given the time budget, the randomized restarts are likely the most effective improvement 
                # over the base greedy. The 2-1 exchange is a deep local search that might be too slow 
                # to implement efficiently in Python within the remaining time for all pairs.
                # We will rely on the diversity of restarts.
                pass

    if best_solution is None:
        # Fallback: return a valid covering
        # Just return the first greedy result
        selected = run_greedy(use_weighted=True, random_tie=False)
        best_solution = [list(blocks[i]) for i in selected]
        
    return best_solution
