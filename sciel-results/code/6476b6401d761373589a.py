import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_triples = len(all_t_subsets)
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_len = float('inf')
    
    def fast_greedy_cover(mode='basic'):
        """Greedy covering with different scoring modes."""
        uncovered = np.ones(n_triples, dtype=np.int32)
        blocks = []
        
        # Track pair usage for diversity bonus
        pair_usage = {}
        
        while uncovered.any():
            uncovered_indices = np.where(uncovered > 0)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = random.choice(uncovered_indices.tolist())
            seed_triple = all_t_subsets[seed_idx]
            
            block = list(seed_triple)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_points = []
                
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
                    
                    # Diversity bonus: prefer points that form rare pairs
                    if mode == 'diversity':
                        bonus = 0.0
                        for y in block:
                            p = (min(x, y), max(x, y))
                            cnt = pair_usage.get(p, 0)
                            bonus += 1.0 / (1.0 + cnt)
                        score = gain + 0.5 * bonus
                    else:
                        score = gain
                    
                    if score > best_gain + 1e-9:
                        best_gain = score
                        best_points = [x]
                    elif abs(score - best_gain) < 1e-9 and score > 0:
                        best_points.append(x)
                
                if best_points:
                    chosen = random.choice(best_points)
                    block.append(chosen)
                    block_set.add(chosen)
                    # Update pair usage
                    for y in block[:-1]:
                        p = (min(chosen, y), max(chosen, y))
                        pair_usage[p] = pair_usage.get(p, 0) + 1
                else:
                    available = [x for x in all_points if x not in block_set]
                    if available:
                        chosen = random.choice(available)
                        block.append(chosen)
                        block_set.add(chosen)
                        for y in block[:-1]:
                            p = (min(chosen, y), max(chosen, y))
                            pair_usage[p] = pair_usage.get(p, 0) + 1
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx.get(triple)
                if idx is not None:
                    uncovered[idx] = 0
        
        return blocks
    
    def local_search_remove_redundant(blocks):
        """Remove redundant blocks and perform swap-based local search."""
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
        
        # Swap-based local search: try to replace blocks with better alternatives
        max_swap_passes = 5
        for _ in range(max_swap_passes):
            if time.time() > deadline - 10:
                break
            
            coverage = build_coverage_map(blocks)
            improved = False
            
            # Try each block for potential replacement
            for i, block in enumerate(blocks):
                if time.time() > deadline - 10:
                    break
                
                unique_triples = []
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(triple)
                    if idx is not None and len(coverage[idx]) == 1:
                        unique_triples.append(triple)
                
                if not unique_triples:
                    continue
                
                # Try several replacement candidates
                for attempt in range(8):
                    if time.time() > deadline - 10:
                        break
                    
                    # Build a new block that covers the unique triples
                    seed_triple = random.choice(unique_triples)
                    new_block = list(seed_triple)
                    new_block_set = set(new_block)
                    
                    # Fill the block greedily, prioritizing coverage of unique triples
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
                                if idx is not None:
                                    # Prioritize covering unique triples
                                    if len(coverage[idx]) == 1:
                                        gain += 2
                                    else:
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
                    
                    # Check if the new block covers all unique triples of the old block
                    valid = True
                    new_block_triples = set(itertools.combinations(new_block, t))
                    for triple in unique_triples:
                        if triple not in new_block_triples:
                            valid = False
                            break
                    
                    if valid and new_block not in blocks:
                        # Perform the swap and verify
                        new_blocks = blocks[:]
                        new_blocks[i] = new_block
                        new_coverage = build_coverage_map(new_blocks)
                        
                        all_covered = True
                        for c in new_coverage:
                            if len(c) == 0:
                                all_covered = False
                                break
                        
                        if all_covered:
                            blocks = new_blocks
                            coverage = new_coverage
                            improved = True
                            break
            
            if not improved:
                break
        
        return blocks
    
    # Main search loop with multiple strategies
    strategies = ['basic', 'diversity', 'basic', 'diversity']
    strategy_idx = 0
    
    while time.time() < deadline:
        mode = strategies[strategy_idx % len(strategies)]
        strategy_idx += 1
        
        blocks = fast_greedy_cover(mode)
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we have a good solution, try to improve it further
        if best_blocks is not None and time.time() < deadline - 15:
            # Additional local search passes on the best solution
            for _ in range(2):
                if time.time() > deadline - 10:
                    break
                improved_blocks = local_search_remove_redundant(best_blocks[:])
                if len(improved_blocks) < best_len:
                    best_len = len(improved_blocks)
                    best_blocks = improved_blocks
    
    return best_blocks if best_blocks is not None else []
