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
    
    # Precompute all C(v, k) blocks and their triple indices for fast lookup
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}
    
    # For each block, precompute its triple indices
    block_triple_indices = np.zeros((num_blocks, 35), dtype=np.int32)  # C(7,3)=35
    for b_idx, block in enumerate(all_blocks):
        block_triple_indices[b_idx] = [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
    
    # Precompute which blocks contain each triple (for fast candidate filtering)
    triple_to_blocks = defaultdict(list)
    for b_idx in range(num_blocks):
        for t_idx in block_triple_indices[b_idx]:
            triple_to_blocks[t_idx].append(b_idx)
    
    def get_block_triples(block):
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
        
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
            
            # Find all blocks containing this triple
            candidate_block_indices = triple_to_blocks[target_triple_idx]
            
            if not candidate_block_indices:
                break
            
            # Vectorized gain computation: for each candidate block, count uncovered triples it covers
            gains = np.zeros(len(candidate_block_indices), dtype=np.int32)
            for ci, b_idx in enumerate(candidate_block_indices):
                block_triples = block_triple_indices[b_idx]
                gains[ci] = sum(1 for t_idx in block_triples if t_idx in uncovered)
            
            # Pick the block with maximum gain
            best_ci = np.argmax(gains)
            best_block = all_blocks[candidate_block_indices[best_ci]]
            
            blocks.append(list(best_block))
            
            # Update coverage
            b_idx = candidate_block_indices[best_ci]
            b_triples = block_triple_indices[b_idx]
            for idx in b_triples:
                coverage_count[idx] += 1
                if coverage_count[idx] == 1:
                    uncovered.discard(idx)
                    
        return blocks, coverage_count

    def local_search(blocks, coverage_count):
        block_tuples = [tuple(b) for b in blocks]
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
            
            # Pass 2: Exchange - use greedy point selection instead of full enumeration
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
                
                # Greedy selection of extra points
                current_block_set = set(mandatory_points)
                extra_points = []
                
                for _ in range(num_extra_needed):
                    best_point = None
                    best_score = -1
                    remaining_available = [p for p in available_points if p not in current_block_set]
                    
                    for p in remaining_available:
                        # Count how many new triples this point would cover
                        test_block = current_block_set | {p}
                        score = 0
                        for triple in itertools.combinations(test_block, t):
                            t_idx = triple_to_idx[triple]
                            if coverage_count[t_idx] == 0:
                                score += 2
                            elif coverage_count[t_idx] == 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_point = p
                    
                    if best_point is not None:
                        extra_points.append(best_point)
                        current_block_set.add(best_point)
                    
                    if time.time() > deadline:
                        break
                
                if len(extra_points) < num_extra_needed:
                    continue
                
                new_block = tuple(sorted(list(mandatory_points) + extra_points))
                if new_block == b:
                    continue
                
                new_triples = get_block_triples(new_block)
                
                gain = 0
                for t_idx in new_triples:
                    if coverage_count[t_idx] == 0:
                        gain += 1
                        
                redundancy_gain = 0
                for t_idx in new_triples:
                    if coverage_count[t_idx] == 1:
                        redundancy_gain += 1
                        
                score = gain * 10 + redundancy_gain
                if score > 0:
                    old_triples = get_block_triples(b)
                    for t_idx in old_triples:
                        coverage_count[t_idx] -= 1
                        
                    for t_idx in new_triples:
                        coverage_count[t_idx] += 1
                        
                    block_tuples[i] = new_block
                    improved = True
                    break
            
            if time.time() > deadline:
                break
            if improved:
                continue
            
            # Pass 3: Single point swap - replace one point in block with one outside
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_set = set(b)
                
                # Check if there's a beneficial single point swap
                best_swap = None
                best_score = -1
                
                for p_in in b:
                    for p_out in points:
                        if p_out in b_set:
                            continue
                        
                        new_block = tuple(sorted(list(b_set - {p_in} | {p_out})))
                        new_triples = get_block_triples(new_block)
                        
                        gain = 0
                        for t_idx in new_triples:
                            if coverage_count[t_idx] == 0:
                                gain += 1
                            
                        if gain > best_score:
                            best_score = gain
                            best_swap = (p_in, p_out)
                        
                        if time.time() > deadline:
                            break
                    
                    if time.time() > deadline:
                        break
                
                if best_swap is not None and best_score > 0:
                    p_in, p_out = best_swap
                    b_set = set(block_tuples[i])
                    b_set.discard(p_in)
                    b_set.add(p_out)
                    new_block = tuple(sorted(b_set))
                    
                    old_triples = get_block_triples(block_tuples[i])
                    for t_idx in old_triples:
                        coverage_count[t_idx] -= 1
                        
                    new_triples = get_block_triples(new_block)
                    for t_idx in new_triples:
                        coverage_count[t_idx] += 1
                        
                    block_tuples[i] = new_block
                    improved = True
                    break
                    
        return [list(b) for b in block_tuples]

    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts - allow more since greedy is faster
    max_restarts = 20
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        blocks, _ = greedy_construct(random_seed=i)
        blocks = local_search(blocks, None)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        if best_count <= 70:
            break

    if best_blocks is None:
        return []
        
    return best_blocks
