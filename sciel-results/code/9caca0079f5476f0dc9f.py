import itertools
import random
import time

def construct(v, k, t):
    # Precompute all blocks and triples
    all_blocks = list(itertools.combinations(range(v), k))
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    num_blocks = len(all_blocks)
    num_triples = len(all_triples)
    
    # For each block, compute which triple indices it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(ts)
    
    # For each triple, compute which blocks cover it
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(i)
    
    # MaxSAT-inspired local search approach:
    # Start with a random set of blocks, then iteratively improve by 
    # removing redundant blocks and adding blocks that cover most uncovered triples.
    
    def local_search(seed, max_iter=5000):
        random.seed(seed)
        
        # Initialize with a random covering (slightly over-cover to be safe)
        # Use greedy first to get an initial solution, then improve
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples]
        selected = set()
        
        # Greedy initialization
        while uncovered:
            max_gain = -1
            best_block = -1
            for i in range(num_blocks):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    best_block = i
            
            if max_gain == 0:
                break
                
            selected.add(best_block)
            
            for t_idx in block_triples[best_block]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
        
        initial_size = len(selected)
        
        # Local search: try to remove blocks and add better ones
        # Key insight: a block is redundant if all its triples are covered by other selected blocks
        
        # Precompute for each selected block, which triples it uniquely covers
        # and which blocks could replace it
        
        for iteration in range(max_iter):
            # Try to find a redundant block to remove
            # A block is redundant if every triple it covers is also covered by at least one other selected block
            
            # Pick a random selected block to try removing
            if not selected:
                break
                
            candidates = list(selected)
            random.shuffle(candidates)
            
            improved = False
            for block_to_remove in candidates:
                # Check if this block is redundant
                redundant = True
                for t_idx in block_triples[block_to_remove]:
                    # Check if any other selected block covers this triple
                    covered_by_other = False
                    for b_prime in triple_to_blocks[t_idx]:
                        if b_prime != block_to_remove and b_prime in selected:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                
                if redundant:
                    # Remove this block
                    selected.remove(block_to_remove)
                    improved = True
                    break
            
            if not improved:
                # Try to find a better replacement
                # Look for a block that covers many uncovered or singly-covered triples
                # and swap with a less useful block
                
                # Count how many times each selected block's triples are covered
                triple_coverage = [0] * num_triples
                for b in selected:
                    for t_idx in block_triples[b]:
                        triple_coverage[t_idx] += 1
                
                # Find blocks with "waste" - triples that are over-covered
                # and blocks that could reduce waste
                
                # Try adding a new block that covers triples currently covered only once
                # and removing a block that has the most redundant coverage
                
                # Simplified: try to find a block that would allow removing another block
                # This is complex, so use a heuristic:
                
                # Find the selected block with the most "unique" coverage (hard to replace)
                # and the one with least unique coverage (easy to replace)
                
                unique_counts = {}
                for b in selected:
                    unique_count = 0
                    for t_idx in block_triples[b]:
                        if triple_coverage[t_idx] == 1:
                            unique_count += 1
                    unique_counts[b] = unique_count
                
                if unique_counts:
                    # Try to replace the block with lowest unique count
                    min_unique_block = min(unique_counts, key=unique_counts.get)
                    min_unique = unique_counts[min_unique_block]
                    
                    if min_unique <= 1:
                        # This block is easy to replace
                        # Find a block that covers its triples well
                        target_triples = set(block_triples[min_unique_block])
                        
                        # Find best replacement block
                        best_replacement = -1
                        best_score = -1
                        for b_new in range(num_blocks):
                            if b_new in selected:
                                continue
                            # Score: how many of target_triples does it cover
                            overlap = 0
                            for t_idx in block_triples[b_new]:
                                if t_idx in target_triples:
                                    overlap += 1
                            # Also consider how many other triples it covers that are singly-covered
                            extra = 0
                            for t_idx in block_triples[b_new]:
                                if t_idx not in target_triples and triple_coverage[t_idx] == 1:
                                    extra += 1
                            score = overlap * 2 + extra
                            if score > best_score:
                                best_score = score
                                best_replacement = b_new
                        
                        if best_replacement >= 0 and best_score >= len(target_triples):
                            # Perform swap
                            selected.remove(min_unique_block)
                            selected.add(best_replacement)
                            improved = True
                            break
            
            if not improved:
                # No improvement found, try perturbation
                if len(selected) > 1:
                    # Remove a random block
                    to_remove = random.choice(list(selected))
                    selected.remove(to_remove)
                    # This may break the covering, but local search can recover
                    # Actually, this is risky. Instead, do a 2-swap perturbation
                    selected.add(to_remove)  # Undo
                    # Try removing two blocks and adding one
                    if len(selected) >= 2:
                        blocks_list = list(selected)
                        random.shuffle(blocks_list)
                        b1, b2 = blocks_list[0], blocks_list[1]
                        # Check if removing both is safe
                        all_triples_to_cover = set()
                        for t_idx in block_triples[b1]:
                            all_triples_to_cover.add(t_idx)
                        for t_idx in block_triples[b2]:
                            all_triples_to_cover.add(t_idx)
                        
                        # Check if these triples are covered by other selected blocks
                        safe = True
                        for t_idx in all_triples_to_cover:
                            covered = False
                            for b_prime in triple_to_blocks[t_idx]:
                                if b_prime != b1 and b_prime != b2 and b_prime in selected:
                                    covered = True
                                    break
                            if not covered:
                                safe = False
                                break
                        
                        if safe:
                            selected.remove(b1)
                            selected.remove(b2)
                            improved = True
                            break
            
            if not improved and iteration > max_iter // 2:
                # Give up on this trajectory
                break
        
        return list(selected)
    
    # Run multiple local search instances
    start_time = time.time()
    max_time = 110
    
    best_solution = None
    best_size = float('inf')
    
    # First run to get a baseline
    sol_idx = local_search(0, max_iter=2000)
    if sol_idx:
        best_size = len(sol_idx)
        best_solution = sol_idx
    
    # Run more with different seeds
    for i in range(1, 30):
        if time.time() - start_time > max_time:
            break
        
        sol_idx = local_search(i, max_iter=2000)
        if sol_idx and len(sol_idx) < best_size:
            best_size = len(sol_idx)
            best_solution = sol_idx
    
    if best_solution is None:
        # Fallback: simple greedy
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples]
        sol_idx = []
        while uncovered:
            max_gain = -1
            best_block = -1
            for i in range(num_blocks):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    best_block = i
            if max_gain == 0:
                break
            sol_idx.append(best_block)
            for t_idx in block_triples[best_block]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
        best_solution = sol_idx
    
    # Convert indices to blocks
    result = [list(all_blocks[i]) for i in best_solution]
    return result
