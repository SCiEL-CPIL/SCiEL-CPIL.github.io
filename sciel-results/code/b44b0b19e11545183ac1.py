import itertools
import random
import time
import math
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)

    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Build incidence matrix A: (num_blocks, num_triples)
    # A[b, t] = 1 if block b covers triple t
    rows = []
    cols = []
    for b_idx, b in enumerate(all_blocks):
        for triple in itertools.combinations(b, t):
            t_idx = triple_to_idx[triple]
            rows.append(b_idx)
            cols.append(t_idx)

    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(num_blocks, num_triples))
    A_T = A.T.tocsr()

    # Precompute list of triple indices for each block (for fast iteration)
    block_triples = []
    for b_idx in range(num_blocks):
        ts = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]].tolist()
        block_triples.append(ts)

    # Precompute list of block indices for each triple
    triple_blocks = []
    for t_idx in range(num_triples):
        bs = A_T.indices[A_T.indptr[t_idx]:A_T.indptr[t_idx+1]].tolist()
        triple_blocks.append(bs)

    # All blocks have C(k, t) triples
    block_size_triples = math.comb(k, t)

    def greedy_randomized(seed, noise_scale=0.0):
        """
        Greedy set cover with random tie-breaking and optional noise.
        """
        rng = random.Random(seed)
        # gain[i] = number of uncovered triples in block i
        gain = np.full(num_blocks, block_size_triples, dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            if time.time() > deadline - 10:
                break

            # Find max gain
            # Adding noise to break ties or diversify
            if noise_scale > 0:
                # Create a copy with noise
                noisy_gain = gain + np.random.normal(0, noise_scale, num_blocks)
                b_idx = int(np.argmax(noisy_gain))
            else:
                max_gain = np.max(gain)
                if max_gain == 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                b_idx = int(rng.choice(candidates))
            
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
        rng = random.Random(seed)
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
        max_iterations = 20000
        
        for _ in range(max_iterations):
            if time.time() > end_time:
                break
            if time.time() > deadline - 2:
                break

            if len(cover) < 2:
                break

            # Pick two random distinct blocks to remove
            i1, i2 = rng.sample(range(len(cover)), 2)
            b1 = cover[i1]
            b2 = cover[i2]
            
            # Decrement counts for b1 and b2
            for t_idx in block_triples[b1]:
                cover_count[t_idx] -= 1
            for t_idx in block_triples[b2]:
                cover_count[t_idx] -= 1
            
            uncovered = np.where(cover_count == 0)[0]
            
            if len(uncovered) == 0:
                # Both were redundant
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
            
            # If uncovered is large, it's unlikely a single block covers it.
            # Max triples in a block is 35.
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
                        break
            
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
                covers_all = True
                for t_idx in uncovered:
                    # Check if t_idx is in block_triples[b3]
                    if t_idx not in block_triples[b3]:
                        covers_all = False
                        break
                
                if covers_all:
                    # Found a valid swap!
                    # Update cover_count
                    for t_idx in block_triples[b3]:
                        cover_count[t_idx] += 1
                    
                    # Modify cover list
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

    def local_search_1for1(cover, seed, time_budget=5.0):
        """
        Local search trying to replace 1 block with 1 different block (move).
        """
        rng = random.Random(seed)
        cover = list(cover)
        
        cover = reduce_cover(cover)
        
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        best_cover = list(cover)
        best_len = len(best_cover)
        
        end_time = time.time() + time_budget
        
        for _ in range(10000):
            if time.time() > end_time:
                break
            if time.time() > deadline - 2:
                break

            if len(cover) == 0:
                break

            # Pick a random block to remove
            i = rng.randrange(len(cover))
            b1 = cover[i]
            
            for t_idx in block_triples[b1]:
                cover_count[t_idx] -= 1
            
            uncovered = np.where(cover_count == 0)[0]
            
            if len(uncovered) == 0:
                # Block was redundant
                cover.pop(i)
                best_cover = list(cover)
                best_len = len(cover)
                continue
            
            if len(uncovered) > block_size_triples:
                for t_idx in block_triples[b1]:
                    cover_count[t_idx] += 1
                continue
            
            # Find candidate b2
            min_count = float('inf')
            min_triple = -1
            for t_idx in uncovered:
                count = len(triple_blocks[t_idx])
                if count < min_count:
                    min_count = count
                    min_triple = t_idx
                    if count == 0:
                        break
            
            if min_triple == -1:
                for t_idx in block_triples[b1]:
                    cover_count[t_idx] += 1
                continue
            
            candidates = triple_blocks[min_triple]
            
            found_fix = False
            for b2 in candidates:
                if b2 == b1:
                    continue
                
                covers_all = True
                for t_idx in uncovered:
                    if t_idx not in block_triples[b2]:
                        covers_all = False
                        break
                
                if covers_all:
                    # Swap b1 with b2
                    for t_idx in block_triples[b2]:
                        cover_count[t_idx] += 1
                    cover[i] = b2
                    # No length change, but maybe better for future swaps?
                    # Actually, 1-for-1 doesn't reduce size. 
                    # But it might allow a subsequent 2-for-1.
                    # However, since we track best_len, we only accept if it leads to reduction later.
                    # For now, let's just accept the swap to explore neighborhood.
                    # But we need to be careful not to loop.
                    # Let's just do the swap and continue.
                    found_fix = True
                    break
            
            if not found_fix:
                for t_idx in block_triples[b1]:
                    cover_count[t_idx] += 1

        return best_cover

    best_cover = None
    
    # Strategy:
    # 1. Run several greedy instances with different noise levels.
    # 2. Reduce and Local Search the best ones.
    
    num_restarts = 0
    max_restarts = 50
    
    noise_scales = [0.0, 0.5, 1.0, 2.0, 0.0, 0.5, 1.0, 2.0, 0.0, 1.0]
    
    while True:
        if time.time() > deadline - 15:
            break
            
        seed = num_restarts
        noise = noise_scales[num_restarts % len(noise_scales)]
        
        cover = greedy_randomized(seed, noise_scale=noise)
        cover = reduce_cover(cover)
        
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
            
            # If we found a new best, do a quick local search
            if time.time() < deadline - 20:
                ls_cover = local_search_2for1(best_cover, seed + 1000, time_budget=3.0)
                if len(ls_cover) < len(best_cover):
                    best_cover = ls_cover
        else:
            pass
            
        num_restarts += 1
        if num_restarts > 100:
            break

    # Final local search with remaining time
    if best_cover is not None:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            # Alternate between 2-for-1 and 1-for-1 to escape local optima
            current = best_cover
            while time.time() < deadline - 2:
                t1 = time.time()
                elapsed = t1 - start_time
                time_left = deadline - t1 - 2
                if time_left < 2:
                    break
                
                # Do a 2-for-1 pass
                ls_cover = local_search_2for1(current, seed + 2000, time_budget=min(5.0, time_left/2))
                if len(ls_cover) < len(current):
                    current = ls_cover
                    seed += 100
                else:
                    # If no improvement, try 1-for-1 to perturb
                    ls_cover = local_search_1for1(current, seed + 3000, time_budget=min(3.0, time_left/2))
                    # 1-for-1 doesn't reduce size, but might enable 2-for-1 later
                    # We don't accept 1-for-1 result as "best" unless it reduces size (which it doesn't)
                    # But we can use it as the starting point for the next 2-for-1
                    current = ls_cover
                    seed += 100

            if len(current) < len(best_cover):
                best_cover = current

    if best_cover is None:
        # Fallback
        cover = greedy_randomized(0, noise_scale=0.0)
        return [list(all_blocks[b_idx]) for b_idx in cover]

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
