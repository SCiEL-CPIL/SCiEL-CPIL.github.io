import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_triples = len(all_t_subsets)
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_len = float('inf')
    
    def fast_greedy_cover():
        uncovered = np.ones(n_triples, dtype=np.int32)
        blocks = []
        
        while uncovered.any():
            # Pick a random uncovered triple
            uncovered_indices = np.where(uncovered > 0)[0]
            seed_idx = random.choice(uncovered_indices.tolist())
            seed_triple = all_t_subsets[seed_idx]
            
            block = list(seed_triple)
            block_set = set(block)
            
            while len(block) < k:
                # For each candidate point, compute how many new triples it covers
                best_gain = -1
                best_points = []
                
                # Precompute pairs in current block
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in all_points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for pair in block_pairs:
                        triple = tuple(sorted((x,) + pair))
                        idx = triple_to_idx.get(triple)
                        if idx is not None and uncovered[idx] > 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain and gain > 0:
                        best_points.append(x)
                
                if best_points:
                    # Random choice among best to add diversity
                    block.append(random.choice(best_points))
                    block_set.add(block[-1])
                else:
                    available = [x for x in all_points if x not in block_set]
                    if available:
                        block.append(random.choice(available))
                        block_set.add(block[-1])
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx.get(triple)
                if idx is not None:
                    uncovered[idx] = 0
        
        return blocks
    
    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        def build_coverage_map(bks):
            coverage = [[] for _ in range(n_triples)]
            for i, block in enumerate(bks):
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(triple)
                    if idx is not None:
                        coverage[idx].append(i)
            return coverage
        
        coverage = build_coverage_map(blocks)
        changed = True
        
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(triple)
                    if idx is not None and len(coverage[idx]) == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                for i, block in enumerate(blocks):
                    if i not in remove_set:
                        new_blocks.append(block)
                
                coverage = build_coverage_map(new_blocks)
                blocks = new_blocks
                changed = True
        
        # Try swap-based local search: replace a block with another that covers more unique triples
        for _ in range(3):  # A few passes of swap improvement
            coverage = build_coverage_map(blocks)
            improved = False
            
            for i, block in enumerate(blocks):
                # Count how many triples are ONLY covered by this block
                unique_triples = 0
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(triple)
                    if idx is not None and len(coverage[idx]) == 1:
                        unique_triples += 1
                
                if unique_triples == 0:
                    continue
                
                # Try replacing block i with a random new block
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    
                    # Generate a random block that covers some of the unique triples
                    # Pick a random uncovered triple and build a block around it
                    unique_list = []
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx.get(triple)
                        if idx is not None and len(coverage[idx]) == 1:
                            unique_list.append(triple)
                    
                    if not unique_list:
                        continue
                    
                    seed_triple = random.choice(unique_list)
                    new_block = list(seed_triple)
                    new_block_set = set(new_block)
                    
                    while len(new_block) < k:
                        best_gain = -1
                        best_pts = []
                        block_pairs = list(itertools.combinations(new_block, t - 1))
                        
                        for x in all_points:
                            if x in new_block_set:
                                continue
                            gain = 0
                            for pair in block_pairs:
                                triple = tuple(sorted((x,) + pair))
                                idx = triple_to_idx.get(triple)
                                if idx is not None and len(coverage[idx]) <= 1:
                                    gain += 1
                            
                            if gain > best_gain:
                                best_gain = gain
                                best_pts = [x]
                            elif gain == best_gain and gain > 0:
                                best_pts.append(x)
                        
                        if best_pts:
                            new_block.append(random.choice(best_pts))
                            new_block_set.add(new_block[-1])
                        else:
                            available = [x for x in all_points if x not in new_block_set]
                            if available:
                                new_block.append(random.choice(available))
                                new_block_set.add(new_block[-1])
                            else:
                                break
                    
                    new_block.sort()
                    
                    # Check if new_block is valid (doesn't leave any triple uncovered)
                    valid = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx.get(triple)
                        if idx is not None and len(coverage[idx]) == 1:
                            # This triple is only covered by block i
                            if triple not in itertools.combinations(new_block, t):
                                valid = False
                                break
                    
                    if valid and new_block not in blocks:
                        # Perform the swap
                        new_blocks = blocks[:]
                        new_blocks[i] = new_block
                        new_coverage = build_coverage_map(new_blocks)
                        
                        # Verify all triples are still covered
                        all_covered = all(len(c) > 0 for c in new_coverage)
                        
                        if all_covered:
                            blocks = new_blocks
                            coverage = new_coverage
                            improved = True
                            break
            
            if not improved:
                break
        
        return blocks
    
    # Main loop
    while time.time() < deadline:
        blocks = fast_greedy_cover()
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we've found a very good solution, do more local search on it
        if best_blocks is not None and time.time() < deadline - 20:
            # Try additional local search passes on the best solution
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                improved_blocks = local_search_remove_redundant(best_blocks[:])
                if len(improved_blocks) < best_len:
                    best_len = len(improved_blocks)
                    best_blocks = improved_blocks
    
    return best_blocks if best_blocks is not None else []
