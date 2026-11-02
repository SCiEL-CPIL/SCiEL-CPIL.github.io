import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute for each block (as sorted tuple) its triple indices
    # But we can't precompute all C(24,7) = 346104 blocks upfront due to memory/time
    # Instead, cache on demand
    block_triple_cache = {}
    
    def get_block_triples(block):
        key = tuple(sorted(block))
        if key in block_triple_cache:
            return block_triple_cache[key]
        result = [triple_to_idx[triple] for triple in itertools.combinations(key, t)]
        block_triple_cache[key] = result
        return result
    
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        uncovered = set(range(num_triples))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick an uncovered triple (randomly to allow different paths)
            target_triple_idx = random.choice(list(uncovered))
            target_triple = all_triples[target_triple_idx]
            
            remaining_points = [p for p in points if p not in target_triple]
            best_block = None
            best_gain = -1
            
            # Iterate through combinations of remaining points to form the block
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_triple + combo))
                
                # Calculate gain using cached triple indices
                b_triples = get_block_triples(block)
                gain = sum(1 for idx in b_triples if idx in uncovered)
                            
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if best_gain == 35:  # Maximum possible gain, can't do better
                        break
                    
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            
            # Update coverage
            b_triples = get_block_triples(best_block)
            for idx in b_triples:
                coverage_count[idx] += 1
                if coverage_count[idx] == 1:
                    uncovered.discard(idx)
                    
        return blocks, coverage_count

    def local_search(blocks, coverage_count):
        block_tuples = [tuple(sorted(b)) for b in blocks]
        # Rebuild coverage from scratch to be safe and consistent
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        for b in block_tuples:
            for t_idx in get_block_triples(b):
                coverage_count[t_idx] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Deletion
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                can_delete = True
                for t_idx in get_block_triples(b):
                    if coverage_count[t_idx] == 1:
                        can_delete = False
                        break
                if can_delete:
                    for t_idx in get_block_triples(b):
                        coverage_count[t_idx] -= 1
                    del block_tuples[i]
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
                unique_triples = [t_idx for t_idx in get_block_triples(b) if coverage_count[t_idx] == 1]
                
                if not unique_triples:
                    continue
                
                # Determine mandatory points for replacement
                mandatory_points = set()
                for t_idx in unique_triples:
                    mandatory_points.update(all_triples[t_idx])
                
                if len(mandatory_points) > k:
                    continue
                
                available_points = [p for p in points if p not in mandatory_points]
                num_extra_needed = k - len(mandatory_points)
                
                if num_extra_needed > len(available_points):
                    continue
                
                best_new_block = None
                best_score = -1
                best_redundancy = -1
                
                # To avoid huge search space if num_extra_needed is large,
                # we can limit attempts or use a heuristic.
                # For k=7, t=3, unique triples usually force 3-6 points.
                # If 3 points, C(21,4)=5985.
                # If 6 points, C(18,1)=18.
                
                candidates = list(itertools.combinations(available_points, num_extra_needed))
                # Shuffle to break symmetry / randomize
                random.shuffle(candidates)
                
                for combo in candidates:
                    new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                    if new_block == b:
                        continue
                    
                    new_triples = get_block_triples(new_block)
                    
                    # Calculate gain: uncovered triples covered
                    gain = 0
                    redundancy_gain = 0
                    for t_idx in new_triples:
                        if coverage_count[t_idx] == 0:
                            gain += 1
                        elif coverage_count[t_idx] == 1:
                            redundancy_gain += 1
                            
                    # Primary: gain (covers new uncovered triples)
                    # Secondary: redundancy (enables future deletions)
                    # Tertiary: prefer blocks that share more with existing blocks (less disruptive)
                    if gain > best_score or (gain == best_score and redundancy_gain > best_redundancy):
                        best_score = gain
                        best_redundancy = redundancy_gain
                        best_new_block = new_block
                        if gain > 0: # Found an improvement in coverage
                            break
                            
                if best_new_block is not None and best_score > 0:
                    # Perform swap
                    old_triples = get_block_triples(b)
                    for t_idx in old_triples:
                        coverage_count[t_idx] -= 1
                        
                    new_triples = get_block_triples(best_new_block)
                    for t_idx in new_triples:
                        coverage_count[t_idx] += 1
                        
                    block_tuples[i] = best_new_block
                    improved = True
                    break
                    
        # Final polish pass: more aggressive deletion and exchange
        if time.time() < deadline - 5:
            improved = True
            max_polish_rounds = 3
            polish_round = 0
            while improved and time.time() < deadline - 2 and polish_round < max_polish_rounds:
                improved = False
                polish_round += 1
                
                # Deletion pass
                for i in range(len(block_tuples)):
                    if time.time() > deadline - 2:
                        break
                    b = block_tuples[i]
                    can_delete = True
                    for t_idx in get_block_triples(b):
                        if coverage_count[t_idx] == 1:
                            can_delete = False
                            break
                    if can_delete:
                        for t_idx in get_block_triples(b):
                            coverage_count[t_idx] -= 1
                        del block_tuples[i]
                        improved = True
                        break
                
                if not improved:
                    # Exchange pass
                    for i in range(len(block_tuples)):
                        if time.time() > deadline - 2:
                            break
                        b = block_tuples[i]
                        unique_triples = [t_idx for t_idx in get_block_triples(b) if coverage_count[t_idx] == 1]
                        
                        if not unique_triples:
                            continue
                        
                        mandatory_points = set()
                        for t_idx in unique_triples:
                            mandatory_points.update(all_triples[t_idx])
                        
                        if len(mandatory_points) > k:
                            continue
                        
                        available_points = [p for p in points if p not in mandatory_points]
                        num_extra_needed = k - len(mandatory_points)
                        
                        if num_extra_needed > len(available_points):
                            continue
                        
                        best_new_block = None
                        best_score = -1
                        best_redundancy = -1
                        
                        candidates = list(itertools.combinations(available_points, num_extra_needed))
                        random.shuffle(candidates)
                        
                        for combo in candidates:
                            new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                            if new_block == b:
                                continue
                            
                            new_triples = get_block_triples(new_block)
                            gain = 0
                            redundancy_gain = 0
                            for t_idx in new_triples:
                                if coverage_count[t_idx] == 0:
                                    gain += 1
                                elif coverage_count[t_idx] == 1:
                                    redundancy_gain += 1
                            
                            if gain > best_score or (gain == best_score and redundancy_gain > best_redundancy):
                                best_score = gain
                                best_redundancy = redundancy_gain
                                best_new_block = new_block
                                if gain > 0:
                                    break
                            
                        if best_new_block is not None and best_score > 0:
                            old_triples = get_block_triples(b)
                            for t_idx in old_triples:
                                coverage_count[t_idx] -= 1
                            
                            new_triples = get_block_triples(best_new_block)
                            for t_idx in new_triples:
                                coverage_count[t_idx] += 1
                            
                            block_tuples[i] = best_new_block
                            improved = True
                            break
        
        return [list(b) for b in block_tuples]

    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with adaptive budget
    # Use more restarts if we have time, fewer if we're tight
    elapsed = time.time() - start_time
    remaining_time = deadline - time.time()
    
    # Estimate time per restart based on first restart or use a heuristic
    # For C(24,7,3), greedy takes ~2-5s, local search ~5-15s
    # Let's do at least 3 restarts, up to 15 if time permits
    min_restarts = 3
    max_restarts = 15
    
    # Start with a few quick restarts, then do deeper local search on best
    for i in range(min_restarts):
        if time.time() > deadline:
            break
            
        blocks, _ = greedy_construct(random_seed=i)
        blocks = local_search(blocks, None)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        if best_count <= 70:
            break
    
    # If we have time, do more restarts with different seeds
    if best_blocks is not None and time.time() < deadline - 10:
        for i in range(min_restarts, max_restarts):
            if time.time() > deadline - 5:
                break
                
            blocks, _ = greedy_construct(random_seed=i + 100)
            blocks = local_search(blocks, None)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                
            if best_count <= 70:
                break
    
    # Final: if we have significant time left, do another round of local search on best
    if best_blocks is not None and time.time() < deadline - 15:
        # Try to improve the best solution further with more local search iterations
        improved_blocks = local_search(best_blocks, None)
        if len(improved_blocks) < best_count:
            best_count = len(improved_blocks)
            best_blocks = improved_blocks

    if best_blocks is None:
        # Fallback: return whatever we have or empty if failed
        return []
        
    return best_blocks
