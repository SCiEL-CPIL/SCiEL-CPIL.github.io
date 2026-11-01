import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Precompute all t-subsets and map them to indices
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)

    # Precompute for each block, the list of subset indices it covers
    # We'll do this on demand but cache it.
    # To speed up greedy, we can precompute a "signature" or just use the list.
    
    # Optimization: Precompute the list of subsets for each possible block?
    # C(19,7) = 50388. Each has C(7,4)=35 subsets.
    # 50388 * 35 = ~1.7M integers. Memory is fine.
    # Let's precompute all block signatures to speed up the greedy loop significantly.
    # block_sig[block] = list of subset indices
    
    # We need a way to index blocks. We can just use the tuple of the block.
    # But iterating over all blocks in Python is slow.
    # Let's use a different approach for the initial greedy.
    
    # Strategy:
    # 1. Randomized Greedy Construction with "Look-ahead" or "Batch" selection.
    # 2. Local Search: 
    #    a. Remove redundant blocks.
    #    b. Replace blocks to maximize redundancy.
    #    c. 2-for-1 swaps.
    
    # Let's stick to the structure of the current program but optimize the inner loops.
    # The main bottleneck in the current program is the greedy step:
    # "for block in itertools.combinations(range(v), k): ... count uncovered"
    # This is O(C(v,k)) per step. With ~150 steps, that's 7.5M iterations.
    # Each iteration does a set intersection of size 35.
    
    # Improvement 1: Use NumPy for the greedy step?
    # Represent uncovered as a boolean array of size n_subsets.
    # Precompute a matrix? No, too big.
    # Precompute for each block the list of subset indices.
    
    # Let's precompute the mapping from block (tuple) to subset indices.
    # This takes time upfront but speeds up the loop.
    
    # Actually, let's try a different greedy strategy:
    # Instead of scanning ALL blocks, we can sample a subset of blocks?
    # No, greedy set cover relies on finding the MAX.
    
    # However, we can speed up the counting.
    # Let's precompute `block_subsets` for all C(v,k) blocks.
    # This is a list of lists.
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_indices = {}
    for b in all_blocks:
        block_indices[b] = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        
    # This precomputation takes:
    # 50388 blocks * 35 subsets * (lookup + append)
    # Lookup is O(1).
    # This should take a few seconds.
    
    # Now the greedy step:
    # uncovered is a set of indices.
    # We want to find block b maximizing len(uncovered.intersection(block_indices[b]))
    
    # We can represent uncovered as a numpy boolean array for faster intersection?
    # Or just use sets. Python sets are quite fast for small sizes (35).
    
    # Let's try to optimize the local search part, which is where the real gains are.
    # The current local search does:
    # 1. Remove redundant.
    # 2. Try to replace each block with a better one.
    
    # Improvement 2: Better replacement heuristic.
    # Instead of just maximizing "already covered" subsets in the new block,
    # we should consider the "net gain" in redundancy or potential for future removals.
    
    # Improvement 3: Add a "2-for-1" swap.
    # If two blocks can be replaced by one block, do it.
    # This requires checking if the union of critical subsets of B1 and B2 fits in a k-set.
    
    # Let's implement the optimized version.
    
    # Precomputation
    # To save memory and time, we might not precompute ALL blocks if it's too slow.
    # But for v=19, k=7, it's manageable.
    
    # Let's check time for precomputation.
    t_pre = time.time()
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_subs = {}
    for b in all_blocks:
        block_to_subs[b] = [subset_to_idx[s] for s in itertools.combinations(b, t)]
    # print(f"Precomputation took {time.time() - t_pre:.2f}s")
    
    # Greedy Construction
    cover_count = np.zeros(n_subsets, dtype=np.int32)
    blocks = []
    uncovered = set(range(n_subsets))
    
    # Use a list of block tuples for fast iteration
    block_list = list(block_to_subs.keys())
    
    while uncovered and time.time() < deadline:
        best_block = None
        best_count = 0
        
        # Iterate over all blocks
        for b in block_list:
            subs = block_to_subs[b]
            # Count uncovered
            # Using a generator expression might be slower than a loop for small lists?
            # Let's try:
            cnt = 0
            for s in subs:
                if s in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_block = b
                if best_count == len(uncovered):
                    break
        
        if best_block is None or best_count == 0:
            break
            
        blocks.append(best_block)
        subs = block_to_subs[best_block]
        for s in subs:
            cover_count[s] += 1
            if cover_count[s] == 1:
                uncovered.discard(s)
                
    # Local Search
    # We need to be careful with the replacement logic.
    
    def is_redundant(block):
        subs = block_to_subs[block]
        for s in subs:
            if cover_count[s] <= 1:
                return False
        return True

    def remove_block(block):
        nonlocal blocks, cover_count
        subs = block_to_subs[block]
        for s in subs:
            cover_count[s] -= 1
        blocks.remove(block)

    def add_block(block):
        nonlocal blocks, cover_count
        subs = block_to_subs[block]
        for s in subs:
            cover_count[s] += 1
        blocks.append(block)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Reduction Phase
        i = 0
        while i < len(blocks):
            if is_redundant(blocks[i]):
                remove_block(blocks[i])
                improved = True
            else:
                i += 1
                
        # 2. Optimization Phase: Block Replacement
        # Try to replace each block with one that covers its critical subsets
        # and maximizes overlap with other blocks' subsets.
        
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
                
            B = blocks[i]
            subs_B = block_to_subs[B]
            
            # Find critical subsets (covered only by B)
            critical = [s for s in subs_B if cover_count[s] == 1]
            
            if not critical:
                continue
                
            # Union of elements in critical subsets
            U_B = set()
            for s in critical:
                U_B.update(all_subsets[s])
                
            if len(U_B) > k:
                continue
                
            # Find best replacement block B' containing U_B
            # We want to maximize the number of subsets in B' that are "well-covered" (count > 1)
            # to make other blocks redundant.
            
            best_B_prime = None
            best_score = -1
            
            # Generate candidates containing U_B
            remaining_points = [p for p in range(v) if p not in U_B]
            num_to_choose = k - len(U_B)
            
            # To speed up, we can limit the number of candidates or use a heuristic.
            # For now, iterate all.
            for candidate_extra in itertools.combinations(remaining_points, num_to_choose):
                B_prime = tuple(sorted(list(U_B) + list(candidate_extra)))
                
                if B_prime == B:
                    continue
                    
                # Check if B_prime is in block_to_subs (it should be)
                if B_prime not in block_to_subs:
                    continue
                    
                subs_Bp = block_to_subs[B_prime]
                
                # Score: number of subsets in B_prime with cover_count > 1
                # Note: cover_count currently includes B.
                # If we replace B with B_prime:
                # For s in B_prime:
                #   if s in B: new_count = cover_count[s] (unchanged, since -1 + 1)
                #   if s not in B: new_count = cover_count[s] + 1
                # We want to maximize the potential to make other blocks redundant.
                # A simple heuristic: count how many subsets in B_prime have cover_count > 1.
                
                score = 0
                for s in subs_Bp:
                    if cover_count[s] > 1:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_B_prime = B_prime
                    
            if best_B_prime is not None:
                # Simulate replacement
                # Remove B
                for s in subs_B:
                    cover_count[s] -= 1
                    
                # Add B_prime
                subs_Bp = block_to_subs[best_B_prime]
                for s in subs_Bp:
                    cover_count[s] += 1
                    
                blocks[i] = best_B_prime
                
                # Check if any block is now redundant
                redundant_found = False
                for j in range(len(blocks)):
                    if is_redundant(blocks[j]):
                        redundant_found = True
                        break
                        
                if redundant_found:
                    improved = True
                    break
                else:
                    # Revert
                    for s in subs_Bp:
                        cover_count[s] -= 1
                    for s in subs_B:
                        cover_count[s] += 1
                    blocks[i] = B

    return blocks
