import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)

    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Map block index to list of triple indices it covers
    # and triple index to list of block indices covering it
    block_triples = []
    triple_blocks = [[] for _ in range(num_triples)]
    
    # Precompute for faster access
    for b_idx, b in enumerate(all_blocks):
        ts = []
        for triple in itertools.combinations(b, t):
            t_idx = triple_to_idx[triple]
            ts.append(t_idx)
            triple_blocks[t_idx].append(b_idx)
        block_triples.append(ts)
        
    # Convert to numpy arrays for faster operations if needed, 
    # but lists are fine for small v=24.
    # num_blocks = C(24,7) = 346104
    # num_triples = C(24,3) = 2024
    
    # Precompute gain (number of triples per block) - constant
    # base_gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
    # All blocks have C(7,3) = 35 triples.
    block_size_triples = math.comb(k, t)

    def greedy_randomized(seed, noise_scale=0.0):
        """
        Greedy set cover with random tie-breaking and optional noise.
        """
        random.seed(seed)
        # gain[i] = number of uncovered triples in block i
        gain = np.full(num_blocks, block_size_triples, dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            if time.time() > deadline - 5:
                break

            # Find max gain
            # Adding noise to break ties or diversify
            if noise_scale > 0:
                # Create a copy with noise
                noisy_gain = gain + np.random.normal(0, noise_scale, num_blocks)
                b_idx = np.argmax(noisy_gain)
            else:
                max_gain = np.max(gain)
                if max_gain == 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                b_idx = random.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)

            # Update gains
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # Decrease gain for all blocks covering this triple
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        """
        Remove redundant blocks.
        """
        if not selected_blocks:
            return []
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        active = [True] * len(selected_blocks)
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True

        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

    def local_search_2for1(cover, seed, time_budget=10.0):
        """
        Local search trying to replace 2 blocks with 1.
        """
        random.seed(seed)
        cover = list(cover)
        
        # Initial reduction
        cover = reduce_cover(cover)
        
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        best_cover = list(cover)
        best_len = len(best_cover)
        
        end_time = time.time() + time_budget
        max_iterations = 5000
        
        for _ in range(max_iterations):
            if time.time() > end_time:
                break
            if time.time() > deadline - 2:
                break

            if len(cover) < 2:
                break

            # Pick two random distinct blocks to remove
            i1, i2 = random.sample(range(len(cover)), 2)
            b1 = cover[i1]
            b2 = cover[i2]
            
            # Temporarily remove
            # To keep indices stable, we can just simulate removal
            # But modifying the list is easier if we're careful.
            # Let's just compute the uncovered set.
            
            # Decrement counts for b1 and b2
            for t_idx in block_triples[b1]:
                cover_count[t_idx] -= 1
            for t_idx in block_triples[b2]:
                cover_count[t_idx] -= 1
            
            uncovered = np.where(cover_count == 0)[0]
            
            if len(uncovered) == 0:
                # Both were redundant? Unlikely if reduce_cover was run, 
                # but possible if they became redundant after previous swaps.
                # Actually, if both are removed and coverage holds, we saved 2 blocks.
                # Remove them from cover
                # We need to remove i1 and i2 carefully
                if i1 < i2:
                    cover.pop(i2)
                    cover.pop(i1)
                else:
                    cover.pop(i1)
                    cover.pop(i2)
                best_cover = list(cover)
                best_len = len(cover)
                continue
            
            # Try to find a single block b3 that covers all uncovered triples
            found_fix = False
            # Sample some candidate blocks
            # Ideally, pick blocks that cover many of the uncovered triples
            # For speed, we just sample random blocks from the universe
            # Or better, pick blocks that intersect the uncovered set significantly.
            
            # Optimization: Pre-compute for each block, the list of triples.
            # We need b3 such that all t in uncovered are in b3.
            # This is equivalent to: intersection of triple_blocks[t] for all t in uncovered is non-empty.
            
            # If uncovered is large, it's unlikely a single block covers it.
            # Max triples in a block is 35.
            # If len(uncovered) > 35, it's impossible.
            if len(uncovered) > block_size_triples:
                # Restore counts
                for t_idx in block_triples[b1]:
                    cover_count[t_idx] += 1
                for t_idx in block_triples[b2]:
                    cover_count[t_idx] += 1
                continue
            
            # Find candidates: intersection of block lists for uncovered triples
            # Start with the triple with fewest covering blocks
            min_count = float('inf')
            min_triple = -1
            for t_idx in uncovered:
                count = len(triple_blocks[t_idx])
                if count < min_count:
                    min_count = count
                    min_triple = t_idx
                    if count == 0:
                        break # Should not happen if cover was valid
            
            if min_triple == -1:
                # Restore
                for t_idx in block_triples[b1]:
                    cover_count[t_idx] += 1
                for t_idx in block_triples[b2]:
                    cover_count[t_idx] += 1
                continue
            
            # Get candidate blocks from the rarest uncovered triple
            candidates = triple_blocks[min_triple]
            
            # Check each candidate
            for b3 in candidates:
                if b3 == b1 or b3 == b2:
                    continue
                
                # Check if b3 covers all uncovered
                # We can check if all t in uncovered are in block_triples[b3]
                # For efficiency, we can check if the set of uncovered is a subset of block_triples[b3]
                # Since len(uncovered) is small (<=35), we can just iterate.
                
                covers_all = True
                for t_idx in uncovered:
                    # Check if t_idx is in block_triples[b3]
                    # block_triples[b3] is a list. 
                    # To make this faster, we could use sets, but lists are small.
                    # Or we can precompute a boolean matrix? Too big.
                    # Let's just check.
                    if t_idx not in block_triples[b3]:
                        covers_all = False
                        break
                
                if covers_all:
                    # Found a valid swap!
                    # Remove b1 and b2, add b3
                    # Update cover_count
                    # b1, b2 already decremented.
                    # We need to add b3.
                    for t_idx in block_triples[b3]:
                        cover_count[t_idx] += 1
                    
                    # Modify cover list
                    # Remove i1 and i2
                    if i1 < i2:
                        cover.pop(i2)
                        cover.pop(i1)
                    else:
                        cover.pop(i1)
                        cover.pop(i2)
                    cover.append(b3)
                    
                    best_cover = list(cover)
                    best_len = len(cover)
                    found_fix = True
                    break
            
            if not found_fix:
                # Restore counts for b1 and b2
                for t_idx in block_triples[b1]:
                    cover_count[t_idx] += 1
                for t_idx in block_triples[b2]:
                    cover_count[t_idx] += 1

        return best_cover

    best_cover = None
    
    # Strategy:
    # 1. Run several greedy instances with different noise levels.
    # 2. Reduce and Local Search the best ones.
    
    # To maximize coverage of the search space, we run a few quick greedy passes
    # and then invest remaining time in local search on the best candidate.
    
    num_restarts = 0
    max_restarts = 50 # We will break based on time
    
    # Use a mix of noise scales
    noise_scales = [0.0, 0.5, 1.0, 2.0, 0.0, 0.5, 1.0, 2.0, 0.0, 1.0]
    
    while True:
        if time.time() > deadline - 10:
            break
            
        seed = num_restarts
        noise = noise_scales[num_restarts % len(noise_scales)]
        
        cover = greedy_randomized(seed, noise_scale=noise)
        cover = reduce_cover(cover)
        
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
            
            # If we found a new best, do a quick local search
            if time.time() < deadline - 15:
                ls_cover = local_search_2for1(best_cover, seed + 1000, time_budget=3.0)
                if len(ls_cover) < len(best_cover):
                    best_cover = ls_cover
        else:
            # Even if not better, if it's close, we might want to keep it?
            # No, just keep the best.
            pass
            
        num_restarts += 1
        if num_restarts > 100:
            break

    # Final local search with remaining time
    if best_cover is not None:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            ls_cover = local_search_2for1(best_cover, seed + 2000, time_budget=remaining_time)
            if len(ls_cover) < len(best_cover):
                best_cover = ls_cover

    if best_cover is None:
        # Fallback
        cover = greedy_randomized(0, noise_scale=0.0)
        return [list(all_blocks[b_idx]) for b_idx in cover]

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
