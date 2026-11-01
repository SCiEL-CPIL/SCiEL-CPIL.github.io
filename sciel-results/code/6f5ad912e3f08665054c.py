import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute for each triple, the list of indices of other triples that share 2 points? 
    # No, let's stick to the greedy but optimize the gain calculation.
    
    # Greedy Construction
    coverage_count = np.zeros(num_triples, dtype=np.int32)
    uncovered = set(range(num_triples))
    blocks = []
    
    # Helper to get triple indices for a block
    def get_block_triples(block):
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
    
    # Greedy loop
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick an uncovered triple
        target_triple_idx = next(iter(uncovered))
        target_triple = all_triples[target_triple_idx]
        
        # Find best block containing target_triple
        remaining_points = [p for p in points if p not in target_triple]
        best_block = None
        best_gain = -1
        
        # Precompute the triples of the target_triple part?
        # A block B = target_triple U combo.
        # Triples in B are:
        # 1. target_triple itself (1)
        # 2. Triples with 2 from target, 1 from combo (C(3,2)*C(k-t,1) = 3*(k-t))
        # 3. Triples with 1 from target, 2 from combo (C(3,1)*C(k-t,2) = 3*C(k-t,2))
        # 4. Triples with 0 from target, 3 from combo (C(k-t,3))
        
        # We only care about uncovered ones.
        # We can precompute the indices of triples that are "compatible" with the target triple?
        
        # Iterate through combinations of remaining points to form the block
        for combo in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(target_triple + combo))
            
            # Calculate gain: number of uncovered triples covered by this block
            gain = 0
            for b_triple in itertools.combinations(block, t):
                if triple_to_idx[b_triple] in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is None:
            break
            
        # Add block
        blocks.append(list(best_block))
        
        # Update coverage
        b_triples = get_block_triples(best_block)
        for idx in b_triples:
            coverage_count[idx] += 1
            if coverage_count[idx] == 1:
                uncovered.discard(idx)
                
    # Initial blocks generated.
    # Optimization Phase
    
    # Convert blocks to tuples for hashing
    block_tuples = [tuple(b) for b in blocks]
    
    # Precompute triples for each block in the current set
    block_triples_map = {b: get_block_triples(b) for b in block_tuples}
    
    # Helper to check if a block can be deleted
    def can_delete(block_idx):
        b = block_tuples[block_idx]
        for t_idx in block_triples_map[b]:
            if coverage_count[t_idx] == 1:
                return False
        return True

    # Helper to get unique triples for a block
    def get_unique_triples(block):
        return [t_idx for t_idx in block_triples_map[block] if coverage_count[t_idx] == 1]

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Deletion
        for i in range(len(block_tuples)):
            if time.time() > deadline:
                break
            if can_delete(i):
                b = block_tuples[i]
                for t_idx in block_triples_map[b]:
                    coverage_count[t_idx] -= 1
                del block_tuples[i]
                del block_triples_map[b]
                improved = True
                break
        
        if time.time() > deadline:
            break
            
        if improved:
            continue
            
        # Pass 2: Exchange
        for i in range(len(block_tuples)):
            if time.time() > deadline:
                break
                
            b = block_tuples[i]
            unique_triples = get_unique_triples(b)
            
            if not unique_triples:
                continue
                
            mandatory_points = set()
            for t_idx in unique_triples:
                mandatory_points.update(all_triples[t_idx])
                
            if len(mandatory_points) > k:
                continue
                
            available_points = [p for p in points if p not in mandatory_points]
            
            num_attempts = 5
            found_replacement = False
            for _ in range(num_attempts):
                if time.time() > deadline:
                    break
                    
                if len(available_points) < k - len(mandatory_points):
                    break
                extra_points = random.sample(available_points, k - len(mandatory_points))
                # FIX: Convert set to list before adding to list
                new_block_tuple = tuple(sorted(list(mandatory_points) + extra_points))
                
                if new_block_tuple == b or new_block_tuple in block_tuples:
                    continue
                    
                new_block_triples = get_block_triples(new_block_tuple)
                
                # Perform swap
                for t_idx in block_triples_map[b]:
                    coverage_count[t_idx] -= 1
                del block_triples_map[b]
                
                block_tuples[i] = new_block_tuple
                block_triples_map[new_block_tuple] = new_block_triples
                for t_idx in new_block_triples:
                    coverage_count[t_idx] += 1
                    
                improved = True
                found_replacement = True
                break
                
            if found_replacement:
                break

    # Convert back to lists
    return [list(b) for b in block_tuples]
