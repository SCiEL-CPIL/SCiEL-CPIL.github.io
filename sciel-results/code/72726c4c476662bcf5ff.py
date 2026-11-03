import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 130.0  # safety margin below 150s kill
    start_time = time.time()
    
    # 1. Precompute all t-subsets (triples) and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    triple_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # 2. Generate a pool of candidate blocks
    # Strategy: Generate a diverse set of blocks using a fast randomized greedy method
    # We want a pool large enough to allow good selection, but small enough to generate quickly.
    # Let's aim for ~20,000 candidates.
    candidate_blocks = set()
    
    def generate_fast_block(rng):
        """
        Fast heuristic to generate a single block.
        Instead of optimizing perfectly, we just pick a random starting triple
        and fill it with random points, biased towards covering common structures?
        Actually, a simple random block is fine for the pool, but biased is better.
        Let's use a light greedy: pick a random uncovered triple, extend to size k.
        """
        # Pick a random triple
        start_sub = rng.choice(all_t_subsets)
        block = list(start_sub)
        block_set = set(block)
        
        # Fill remaining k-3 points randomly from remaining v-3 points
        remaining_points = [x for x in range(v) if x not in block_set]
        rng.shuffle(remaining_points)
        for p in remaining_points:
            if len(block) == k:
                break
            block.append(p)
        
        return tuple(sorted(block))

    # Generate pool
    pool_size_target = 20000
    rng_pool = random.Random(42)
    
    # To ensure diversity, we mix pure random and light greedy
    # Pure random:
    for _ in range(pool_size_target // 2):
        if time.time() > deadline - 60: break
        block = tuple(sorted(rng_pool.sample(range(v), k)))
        candidate_blocks.add(block)
        
    # Light greedy (random start, random fill but maybe sorted?):
    for _ in range(pool_size_target // 2):
        if time.time() > deadline - 60: break
        block = generate_fast_block(rng_pool)
        candidate_blocks.add(block)
        
    # Add some structured blocks? e.g. cyclic shifts of a base block?
    # For v=24, k=7, t=3.
    # Let's just stick to the randomized pool for now, the selection phase is the key.
    
    candidate_blocks = list(candidate_blocks)
    n_candidates = len(candidate_blocks)
    
    if n_candidates == 0:
        # Fallback: simple greedy if pool generation failed
        return _fallback_greedy(v, k, t, deadline)

    # 3. Precompute the triple indices for each candidate block
    # Each block covers C(7,3) = 35 triples.
    # We create a matrix of shape (n_candidates, 35) containing triple indices.
    # Actually, to speed up "newly covered" calculation, we can store the set of indices.
    # But NumPy sets are slow. 
    # Alternative: Bitmask? 2024 bits is ~253 bytes. 20k blocks * 253 bytes = 5MB. Manageable.
    # But bitwise operations on 2024-bit integers in Python are slow if done per block in a loop.
    
    # Better NumPy approach:
    # Maintain a boolean array `covered` of size n_t_subsets.
    # For each selection step, we need to find the block that maximizes sum(1 - covered[indices]).
    # This is equivalent to: block_size - sum(covered[indices]).
    # Since block_size is constant (35), we minimize sum(covered[indices]).
    
    # Let's store the indices for each block in a 2D numpy array.
    # Some blocks might have duplicate triple indices? No, triples are distinct subsets.
    # So each block has exactly 35 unique triple indices.
    block_triple_indices = np.zeros((n_candidates, 35), dtype=np.int32)
    
    for i, block in enumerate(candidate_blocks):
        triples = itertools.combinations(block, t)
        idxs = [triple_to_idx[tup] for tup in triples]
        block_triple_indices[i, :] = idxs
        
    if time.time() > deadline - 30:
        return _fallback_greedy(v, k, t, deadline)

    # 4. Selection Loop (Submodular Greedy with Vectorized Gain)
    covered = np.zeros(n_t_subsets, dtype=bool)
    selected_blocks = []
    remaining_indices = np.arange(n_candidates)
    
    # We will iteratively pick the best block.
    # To speed up, we can re-evaluate gains less frequently or use approximations,
    # but for 20k blocks and ~70 selections, full re-evaluation might be okay if vectorized.
    # Cost per step: 20k * 35 lookups into `covered`. 
    # 20000 * 35 = 700,000 operations. In NumPy, this is a sum over axis 1.
    # sum(covered[block_triple_indices], axis=1)
    # This creates a temporary array of size (20000, 35) of bools, then sums.
    # This should be very fast (< 1ms).
    # 70 steps * 1ms = 70ms. Very feasible.
    
    # Optimization: Remove selected blocks from remaining pool to avoid duplicates?
    # The problem says blocks must be distinct.
    # We can just keep a set of selected block tuples.
    
    selected_set = set()
    
    while time.time() < deadline - 5:
        if len(remaining_indices) == 0:
            break
            
        # Calculate overlap with currently covered triples for all remaining blocks
        # covered_indices = block_triple_indices[remaining_indices] # (N_rem, 35)
        # overlaps = covered[covered_indices].sum(axis=1) # (N_rem,)
        
        # To save memory, we do it in one go
        sub_indices = block_triple_indices[remaining_indices]
        # Map indices to covered status
        # Note: covered is a 1D bool array. 
        # We can index it with sub_indices directly if we flatten? 
        # No, covered[sub_indices] works and returns shape (N_rem, 35)
        overlaps = covered[sub_indices].sum(axis=1)
        
        # Gain = 35 - overlaps
        # We want to maximize gain.
        # Tie-breaking: Random? Or prefer lower index? 
        # Let's add a small random noise to break ties and encourage diversity?
        # Or just pick the first max.
        
        max_gain_idx = np.argmin(overlaps) # Min overlap = Max gain
        best_block_idx = remaining_indices[max_gain_idx]
        
        # Check if this block is already selected (shouldn't happen if we remove from remaining)
        # We are using remaining_indices, so we must update it.
        
        block = candidate_blocks[best_block_idx]
        
        if block in selected_set:
            # Should not happen if we manage remaining_indices correctly
            pass
        else:
            selected_blocks.append(block)
            selected_set.add(block)
            
            # Update covered
            # Get indices of triples in this block
            block_idxs = block_triple_indices[best_block_idx]
            covered[block_idxs] = True
            
            # Remove this block from remaining pool
            # Find position of best_block_idx in remaining_indices
            pos = np.where(remaining_indices == best_block_idx)[0][0]
            remaining_indices = np.delete(remaining_indices, pos)
            
            # Early termination if all covered
            if covered.all():
                break
                
    # 5. Validation and Fallback
    # Check if we have a valid covering
    if not covered.all():
        # If the pool was not sufficient to cover everything, we need to fill gaps.
        # Use a simple greedy on the *uncovered* triples using the *remaining* pool or new blocks.
        # Let's use a standard greedy on the fly for the remaining gaps.
        uncovered = np.where(~covered)[0]
        if len(uncovered) > 0:
            # Map indices back to triples
            uncovered_triples = [all_t_subsets[i] for i in uncovered]
            
            # Simple greedy to finish
            blocks_list = list(selected_blocks)
            covered_set = set()
            for b in blocks_list:
                for sub in itertools.combinations(b, t):
                    covered_set.add(sub)
            
            while uncovered_triples and time.time() < deadline - 1:
                # Pick a random uncovered triple
                start_sub = random.choice(uncovered_triples)
                block = list(start_sub)
                block_set = set(block)
                
                # Greedily add points to cover max uncovered triples
                while len(block) < k:
                    best_gain = -1
                    best_point = -1
                    # Candidates
                    candidates = [x for x in range(v) if x not in block_set]
                    # Only consider points that appear in uncovered triples with current block?
                    # For speed in this small final phase, just check all
                    if len(block) >= t - 1:
                        block_subsets = set(itertools.combinations(block, t - 1))
                    else:
                        block_subsets = set()
                        
                    for x in candidates:
                        count = 0
                        for sub in block_subsets:
                            new_t_sub = tuple(sorted(sub + (x,)))
                            if new_t_sub in uncovered_triples:
                                count += 1
                        if count > best_gain:
                            best_gain = count
                            best_point = x
                            
                    if best_point == -1:
                        best_point = random.choice(candidates) if candidates else -1
                        if best_point == -1: break
                        
                    block.append(best_point)
                    block_set.add(best_point)
                    
                if len(block) == k:
                    block.sort()
                    if tuple(block) not in selected_set:
                        blocks_list.append(block)
                        selected_set.add(tuple(block))
                        
                    # Update uncovered
                    for sub in itertools.combinations(block, t):
                        if sub in uncovered_triples:
                            uncovered_triples.remove(sub)
                            # Also update covered_set if we were tracking it, but we just need final list
                            
            selected_blocks = blocks_list
            # Final check
            final_covered = set()
            for b in selected_blocks:
                for sub in itertools.combinations(b, t):
                    final_covered.add(sub)
            
            if len(final_covered) == n_t_subsets:
                return selected_blocks
            else:
                # Failed to cover all, return best effort (might be invalid, but better than crash)
                # Or force a valid one? The task says "Valid when every...". 
                # If we return invalid, score is 0. 
                # We should try to ensure validity. 
                # The greedy above *should* cover everything if it runs to completion.
                # If it didn't, it's because of timeout.
                # Let's return the selected_blocks. If invalid, score 0. 
                # But we have a deadline guard.
                return selected_blocks
    else:
        return selected_blocks

def _fallback_greedy(v, k, t, deadline):
    """
    Standard greedy if the pool strategy fails or times out early.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline - 2:
        # Pick a random uncovered subset
        start_subset = random.choice(list(uncovered))
        block = list(start_subset)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            best_point = -1
            candidates = [x for x in range(v) if x not in block_set]
            
            if len(block) >= t - 1:
                block_subsets = set(itertools.combinations(block, t - 1))
            else:
                block_subsets = set()
                
            for x in candidates:
                count = 0
                for sub in block_subsets:
                    new_t_sub = tuple(sorted(sub + (x,)))
                    if new_t_sub in uncovered:
                        count += 1
                if count > best_gain:
                    best_gain = count
                    best_point = x
                    
            if best_point == -1:
                best_point = random.choice(candidates) if candidates else -1
                if best_point == -1: break
                
            block.append(best_point)
            block_set.add(best_point)
            
        if len(block) == k:
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
    return blocks
