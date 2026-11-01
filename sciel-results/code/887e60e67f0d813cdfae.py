import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(all_points, k))
    num_blocks = len(all_blocks)
    
    block_triples = []
    triple_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        triples_in_block = []
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[tuple(triple)]
            triples_in_block.append(t_idx)
            triple_blocks[t_idx].append(b_idx)
        block_triples.append(triples_in_block)
        
    best_blocks = None
    best_len = float('inf')
    
    # Precompute counts for weighted greedy
    triple_counts = np.zeros(num_triples, dtype=np.int32)
    for b_idx in range(num_blocks):
        for t_idx in block_triples[b_idx]:
            triple_counts[t_idx] += 1
    
    def greedy_random():
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks_in_sol = []
        blocks_in_sol_set = set()
        
        while True:
            if not np.any(uncovered):
                break
            
            # Pick random uncovered triple
            seed_idx = None
            while True:
                candidate = random.randint(0, num_triples - 1)
                if uncovered[candidate]:
                    seed_idx = candidate
                    break
            
            candidates = triple_blocks[seed_idx]
            if not candidates:
                break
                
            best_candidate = -1
            best_gain = -1
            
            for b_idx in candidates:
                if b_idx in blocks_in_sol_set:
                    continue
                gain = 0
                for t_idx in block_triples[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b_idx
                    if gain == num_triples:
                        break
            
            if best_candidate == -1:
                best_candidate = candidates[0]
                
            blocks_in_sol.append(best_candidate)
            blocks_in_sol_set.add(best_candidate)
            
            for t_idx in block_triples[best_candidate]:
                uncovered[t_idx] = 0
                
        return blocks_in_sol

    def greedy_weighted():
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks_in_sol = []
        blocks_in_sol_set = set()
        
        # Weights for triples: 1/count
        # We want to cover rare triples first.
        # Gain = sum of weights of uncovered triples in block.
        
        while True:
            if not np.any(uncovered):
                break
            
            best_candidate = -1
            best_gain = -1.0
            
            # Iterate all blocks? Too slow.
            # Pick a random uncovered triple, but choose block with max weighted gain among those covering it?
            # Or just pick the block with max weighted gain globally?
            # Global max requires scanning all 346k blocks.
            # Let's do: Pick random uncovered triple. Among blocks covering it, pick max weighted gain.
            
            seed_idx = None
            while True:
                candidate = random.randint(0, num_triples - 1)
                if uncovered[candidate]:
                    seed_idx = candidate
                    break
            
            candidates = triple_blocks[seed_idx]
            if not candidates:
                break
                
            for b_idx in candidates:
                if b_idx in blocks_in_sol_set:
                    continue
                gain = 0.0
                for t_idx in block_triples[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1.0 / triple_counts[t_idx]
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b_idx
            
            if best_candidate == -1:
                best_candidate = candidates[0]
                
            blocks_in_sol.append(best_candidate)
            blocks_in_sol_set.add(best_candidate)
            
            for t_idx in block_triples[best_candidate]:
                uncovered[t_idx] = 0
                
        return blocks_in_sol

    def local_search(blocks_in_sol):
        if not blocks_in_sol:
            return blocks_in_sol
            
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in blocks_in_sol:
            for t_idx in block_triples[b_idx]:
                counts[t_idx] += 1
                
        sol_set = set(blocks_in_sol)
        improved = True
        
        while improved:
            improved = False
            
            # 1. Remove redundant
            to_remove = []
            for b_idx in blocks_in_sol:
                is_redundant = True
                for t_idx in block_triples[b_idx]:
                    if counts[t_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(b_idx)
                    
            if to_remove:
                remove_set = set(to_remove)
                blocks_in_sol = [b for b in blocks_in_sol if b not in remove_set]
                sol_set = set(blocks_in_sol)
                for b_idx in to_remove:
                    for t_idx in block_triples[b_idx]:
                        counts[t_idx] -= 1
                improved = True
                continue
            
            # 2. Swap
            swaps_made = False
            # Shuffle blocks for randomization
            for b_idx in random.sample(blocks_in_sol, len(blocks_in_sol)):
                unique_triples = [t_idx for t_idx in block_triples[b_idx] if counts[t_idx] == 1]
                if not unique_triples:
                    continue
                
                # Find candidates covering all unique_triples
                # Use the first unique triple to filter
                first_t = unique_triples[0]
                candidates = []
                for cand_b_idx in triple_blocks[first_t]:
                    if cand_b_idx in sol_set:
                        continue
                    # Check if covers all unique_triples
                    valid = True
                    for ut in unique_triples:
                        if ut not in block_triples[cand_b_idx]:
                            valid = False
                            break
                    if valid:
                        candidates.append(cand_b_idx)
                
                if not candidates:
                    continue
                
                random.shuffle(candidates)
                # Try more candidates
                for new_b_idx in candidates[:50]: 
                    # Simulate swap
                    # Check if any block becomes redundant
                    # We need to check blocks that might become redundant.
                    # A block C becomes redundant if all its triples have count >= 2 after swap.
                    
                    # To speed up, we can just check all blocks in sol.
                    # But we need to know the NEW counts.
                    # new_counts = counts - block_triples[b_idx] + block_triples[new_b_idx]
                    
                    # Instead of creating a new array, we can check conditionally.
                    # For each block C in sol:
                    #   For each t in C:
                    #      if t in B and t not in B': new_count = count[t] - 1
                    #      if t in B' and t not in B: new_count = count[t] + 1
                    #      else: new_count = count[t]
                    #      if new_count == 1: C is not redundant.
                    
                    # This is O(N_sol * 35 * 35) ? No.
                    # We can precompute sets of triples for B and B'.
                    b_set = set(block_triples[b_idx])
                    nb_set = set(block_triples[new_b_idx])
                    
                    found_redundant = False
                    redundant_block = -1
                    
                    for c_idx in blocks_in_sol:
                        if c_idx == b_idx:
                            continue # b_idx is being removed
                        is_redundant = True
                        for t_idx in block_triples[c_idx]:
                            # Determine new count for t_idx
                            if t_idx in b_set and t_idx not in nb_set:
                                new_c = counts[t_idx] - 1
                            elif t_idx in nb_set and t_idx not in b_set:
                                new_c = counts[t_idx] + 1
                            else:
                                new_c = counts[t_idx]
                                
                            if new_c == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            found_redundant = True
                            redundant_block = c_idx
                            break
                    
                    if found_redundant:
                        # Commit swap and removal
                        sol_set.remove(b_idx)
                        sol_set.add(new_b_idx)
                        blocks_in_sol.remove(b_idx)
                        blocks_in_sol.append(new_b_idx)
                        
                        for t_idx in block_triples[b_idx]:
                            counts[t_idx] -= 1
                        for t_idx in block_triples[new_b_idx]:
                            counts[t_idx] += 1
                        
                        # Remove redundant block
                        sol_set.remove(redundant_block)
                        blocks_in_sol.remove(redundant_block)
                        for t_idx in block_triples[redundant_block]:
                            counts[t_idx] -= 1
                            
                        improved = True
                        swaps_made = True
                        break # Restart main loop
                    else:
                        # No swap, so no need to revert counts (we didn't change them)
                        pass
                if swaps_made:
                    break
            
        return blocks_in_sol

    # Main loop
    while time.time() < deadline:
        if random.random() < 0.5:
            blocks = greedy_random()
        else:
            blocks = greedy_weighted()
            
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(all_blocks[b_idx]) for b_idx in blocks]
            
        # Optional: If best_len is close to lower bound, stop?
        # LB is 69. If we find 70, we can keep going to see if we find 69 (unlikely) or just return.
        # But 69 is the LB, so 70 is the best possible improvement over 71.
        
    return best_blocks if best_blocks is not None else []
