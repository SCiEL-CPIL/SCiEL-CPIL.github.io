import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 150s limit, aim for 140s
    deadline = start_time + 140.0
    
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Precompute block triples for all possible blocks
    # This is C(24,7) = 346,104 blocks. 
    # Storing the list of triple indices for each block is memory intensive but manageable.
    # 346k blocks * 35 triples/block * 4 bytes (int32) ~ 48 MB. Acceptable.
    blocks_list = list(itertools.combinations(points, k))
    num_blocks = len(blocks_list)
    
    # Map block index to list of triple indices
    block_triples = []
    for b in blocks_list:
        ts = [triple_to_idx[triple] for triple in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    # Map triple index to list of block indices
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)

    # Greedy construction: Point-by-point
    def greedy_pointwise(seed):
        rng = random.Random(seed)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        selected_blocks = []
        
        while is_uncovered.any():
            # Start with a random uncovered triple to seed the block? 
            # Or just start empty and pick best point?
            # Starting empty might lead to a block that doesn't cover the "hardest" triples efficiently if the first points are bad.
            # Better: Pick a random uncovered triple, force its points into the block, then fill the rest greedily.
            
            # Find an uncovered triple
            # To speed up, we can just pick the first one or a random one.
            # Let's pick a random uncovered triple to add diversity.
            # Getting a random index where is_uncovered is True:
            uncovered_indices = np.where(is_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_t_idx = rng.choice(uncovered_indices.tolist())
            start_triple = triples[start_t_idx]
            
            current_block = list(start_triple)
            current_block_set = set(current_block)
            
            # Mark these as "tentatively covered" for gain calculation? 
            # No, gain calculation should be based on the global is_uncovered state.
            # But we must ensure the final block covers the start_triple.
            
            while len(current_block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in points if p not in current_block_set]
                
                # Calculate gain for each candidate
                # Gain(p) = number of uncovered triples {x,y,p} such that {x,y} subset of current_block
                # For t=3, {x,y} is a pair from current_block.
                # We can precompute pairs in current_block.
                pairs_in_block = list(itertools.combinations(current_block, t - 1))
                
                # Optimization: 
                # For each candidate p, we check all pairs (a,b) in current_block.
                # Triple is (a,b,p).
                # We check if triple_to_idx[(a,b,p)] is uncovered.
                
                # To speed this up, we can iterate through uncovered triples? No, that's O(N).
                # Iterating candidates (24-3=21) * pairs (3*2/2=3) = 63 checks per step. Very fast.
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        # Create triple
                        # sorted(pair + (p,))
                        triple = tuple(sorted(pair + (p,)))
                        tid = triple_to_idx[triple]
                        if is_uncovered[tid]:
                            gain += 1
                            
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p
                        # Tie-breaking: if gain == max_gain, we could randomize, 
                        # but for now, first found is fine, or we can collect candidates.
                        # Let's collect candidates with max gain for randomization.
                        # Actually, simple max is often good. Let's add slight randomness.
                        # We'll just pick the first max for speed, or store in a list.
                        # Given the small number of candidates, let's just pick the first max.
                        # To improve diversity, we can add noise.
                        # Let's stick to deterministic max for now to ensure consistent gain.
                        # If max_gain is 0, we pick a random candidate.
                
                if max_gain == 0:
                    best_point = rng.choice(candidates)
                
                current_block.append(best_point)
                current_block_set.add(best_point)
                
            # Sort block for consistent indexing
            current_block.sort()
            b_idx = blocks_list.index(tuple(current_block)) # This is O(N) search. 
            # Better: precompute block tuple to index map.
            
            # Let's create the map now if not already done.
            # block_to_idx = {b: i for i, b in enumerate(blocks_list)}
            
            selected_blocks.append(b_idx)
            
            # Update is_uncovered
            for t_idx in block_triples[b_idx]:
                is_uncovered[t_idx] = False
                
        return selected_blocks

    # Precompute block tuple to index map for fast lookup
    block_to_idx = {b: i for i, b in enumerate(blocks_list)}

    def local_search(selected_blocks):
        # selected_blocks is a list of block indices
        if not selected_blocks:
            return []
            
        # Use a set for O(1) lookup of active blocks? 
        # But we need to iterate. List is fine for ~80 blocks.
        active_blocks = list(selected_blocks)
        
        # cover_count: number of blocks covering each triple
        cover_count = np.zeros(num_triples, dtype=np.int32)
        
        for b_idx in active_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
                
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant blocks
            # A block is redundant if all its triples have cover_count > 1
            new_active_blocks = []
            for b_idx in active_blocks:
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    improved = True
                else:
                    new_active_blocks.append(b_idx)
            active_blocks = new_active_blocks
            
            if not active_blocks:
                break
                
            # 2. Try 2-for-1 swaps
            n = len(active_blocks)
            swap_made = False
            # We need to be careful modifying the list while iterating.
            # We'll iterate over pairs, and if a swap is found, we apply it, update state, and restart the pair loop.
            
            for i in range(n):
                if time.time() > deadline - 5:
                    return active_blocks
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        return active_blocks
                    
                    b1 = active_blocks[i]
                    b2 = active_blocks[j]
                    
                    # Find triples covered ONLY by b1 or b2
                    # i.e., cover_count[t] == 1 and t is in b1 or b2
                    # And t is NOT covered by any other active block.
                    
                    # We need to check triples in b1 and b2.
                    # Union of triples in b1 and b2.
                    # If a triple is in both, cover_count >= 2 (from b1 and b2), so it's not "uniquely" covered by one.
                    # We are looking for triples t such that:
                    # 1. t is in b1 OR t is in b2
                    # 2. cover_count[t] == 1 (meaning only ONE block in the whole cover covers it)
                    #    Wait, if t is in b1 and b2, cover_count is at least 2.
                    #    So if cover_count[t] == 1, it must be in exactly one of b1, b2 (or neither, but we check b1/b2).
                    #    Actually, if t is in b1, and cover_count[t]==1, then no other block covers it.
                    #    If t is in b1 AND b2, cover_count[t] >= 2.
                    
                    unique_triples = []
                    points_needed = set()
                    
                    # Check triples in b1
                    for t_idx in block_triples[b1]:
                        if cover_count[t_idx] == 1:
                            unique_triples.append(t_idx)
                            points_needed.update(triples[t_idx])
                            
                    # Check triples in b2
                    for t_idx in block_triples[b2]:
                        if cover_count[t_idx] == 1:
                            unique_triples.append(t_idx)
                            points_needed.update(triples[t_idx])
                            
                    if not unique_triples:
                        # Both blocks are redundant in the context of each other?
                        # If unique_triples is empty, it means every triple in b1 and b2 is covered by at least one other block.
                        # So we can remove both.
                        for t_idx in block_triples[b1]:
                            cover_count[t_idx] -= 1
                        for t_idx in block_triples[b2]:
                            cover_count[t_idx] -= 1
                        active_blocks = [b for idx, b in enumerate(active_blocks) if idx != i and idx != j]
                        improved = True
                        swap_made = True
                        break # Restart pair loop with new list
                    else:
                        # We need to cover these unique_triples with a single new block.
                        if len(points_needed) <= k:
                            # Create new block
                            new_block_points = list(points_needed)
                            # Pad to size k
                            remaining_points = [p for p in points if p not in points_needed]
                            while len(new_block_points) < k:
                                p = remaining_points.pop()
                                new_block_points.append(p)
                            new_block_points.sort()
                            new_b_idx = block_to_idx[tuple(new_block_points)]
                            
                            # Apply swap
                            # Remove b1, b2
                            for t_idx in block_triples[b1]:
                                cover_count[t_idx] -= 1
                            for t_idx in block_triples[b2]:
                                cover_count[t_idx] -= 1
                            # Add new_b
                            for t_idx in block_triples[new_b_idx]:
                                cover_count[t_idx] += 1
                                
                            active_blocks = [b for idx, b in enumerate(active_blocks) if idx != i and idx != j]
                            active_blocks.append(new_b_idx)
                            improved = True
                            swap_made = True
                            break # Restart pair loop with new list
                            
            if swap_made:
                continue # Restart the while loop to check for new redundancies
                
            # 3. Try 1-for-1 swaps (optional, for shuffling)
            # If no 2-for-1 found, try to replace a block with another that covers its unique triples.
            # This doesn't reduce count, but might enable future 2-for-1s.
            # Given time constraints, we might skip this or do a limited pass.
            # Let's try a few random 1-for-1 swaps if stuck.
            
            # For now, let's just break if no 2-for-1 found.
            
        return active_blocks

    # Precompute block_to_idx
    block_to_idx = {b: i for i, b in enumerate(blocks_list)}
    
    best_cover = None
    best_count = float('inf')
    
    # Run multiple seeds
    num_seeds = 50 # Adjust based on time
    seed = 0
    while True:
        if time.time() > deadline - 10:
            break
            
        # Greedy
        cover = greedy_pointwise(seed)
        
        # Local Search
        cover = local_search(cover)
        
        if len(cover) < best_count:
            best_count = len(cover)
            best_cover = cover
            
        seed += 1
        
    if best_cover is None:
        # Fallback
        cover = greedy_pointwise(0)
        return [list(blocks_list[b_idx]) for b_idx in cover]
        
    return [list(blocks_list[b_idx]) for b_idx in best_cover]
