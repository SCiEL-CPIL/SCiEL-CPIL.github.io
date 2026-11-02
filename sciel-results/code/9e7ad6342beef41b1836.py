import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a two-phase approach:
    1. Greedy construction using a max-saturation heuristic
    2. Local search with block removal and 1-for-1 swap optimization
    """
    start_time = time.time()
    time_limit = 115.0
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build coverage matrix: block_coverage[b][s] = 1 if block b covers t-subset s
    # For v=19, k=8, t=5: 75582 blocks, 11628 t-subsets
    # Each block covers C(8,5)=56 t-subsets
    block_coverage = []
    for block in all_blocks:
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(t_to_idx[ts])
        block_coverage.append(covered)
    
    # Build inverse index: for each t-subset, list of blocks that cover it
    # This is useful for efficient swap search
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, covered in enumerate(block_coverage):
        for s in covered:
            t_to_blocks[s].append(b_idx)
    
    # Convert to numpy arrays for faster operations
    # Create a binary matrix where coverage_matrix[b, s] = 1 if block b covers t-subset s
    coverage_matrix = np.zeros((num_blocks, num_t), dtype=np.uint8)
    for b_idx, covered in enumerate(block_coverage):
        for s in covered:
            coverage_matrix[b_idx, s] = 1
    
    # Phase 1: Greedy construction
    # Start with empty covering
    selected_blocks = set()
    covered_count = np.zeros(num_t, dtype=np.uint8)
    
    # Precompute block sizes (all same) and use a priority approach
    # At each step, pick the block that covers the most uncovered t-subsets
    # Use a heap-like approach or just scan
    
    # For efficiency, track uncovered t-subsets
    uncovered = set(range(num_t))
    
    max_iter = 10000
    for iteration in range(max_iter):
        if not uncovered:
            break
        if time.time() - start_time > time_limit * 0.3:
            break
        
        best_block = -1
        best_gain = 0
        
        # Sample a subset of blocks for speed, or scan all
        # For v=19, scanning 75582 blocks is fast enough
        for b_idx in range(num_blocks):
            if b_idx in selected_blocks:
                continue
            # Count how many uncovered t-subsets this block covers
            gain = 0
            for s in uncovered:
                if coverage_matrix[b_idx, s]:
                    gain += 1
                    if gain > best_gain:
                        # Early termination: can't do better than all uncovered
                        if gain == len(uncovered):
                            break
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1:
            break
        
        selected_blocks.add(best_block)
        # Update covered count
        for s in uncovered:
            if coverage_matrix[best_block, s]:
                covered_count[s] += 1
                if covered_count[s] == 1:
                    uncovered.discard(s)
        
        # Early stop if we've used too many blocks
        if len(selected_blocks) > 400:
            break
    
    # Ensure full coverage with a simple greedy completion if needed
    while uncovered and time.time() - start_time < time_limit * 0.4:
        # Find any block that covers at least one uncovered t-subset
        found = False
        for s in list(uncovered):
            for b_idx in t_to_blocks[s]:
                if b_idx not in selected_blocks:
                    selected_blocks.add(b_idx)
                    for s2 in uncovered:
                        if coverage_matrix[b_idx, s2]:
                            covered_count[s2] += 1
                            if covered_count[s2] == 1:
                                uncovered.discard(s2)
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    # Phase 2: Local search - removal and swap
    # Maintain a list for easy manipulation
    block_list = list(selected_blocks)
    
    def get_uncovered_for_block(b_idx):
        """Return list of t-subsets that would become uncovered if b_idx is removed"""
        result = []
        for s in range(num_t):
            if coverage_matrix[b_idx, s] and covered_count[s] == 1:
                result.append(s)
        return result
    
    def try_remove(b_idx):
        """Try to remove block b_idx if it's redundant"""
        unique = get_uncovered_for_block(b_idx)
        if not unique:
            # Remove it
            for s in range(num_t):
                if coverage_matrix[b_idx, s]:
                    covered_count[s] -= 1
            block_list.remove(b_idx)
            return True
        return False
    
    # Iterative removal pass
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(block_list)):
            b_idx = block_list[i]
            if try_remove(b_idx):
                changed = True
                break
    
    best_size = len(block_list)
    
    # Local search: 1-for-1 swaps
    # For each block, try to replace it with another block that covers its unique t-subsets
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        improved = False
        random.shuffle(block_list)
        
        for i, b_idx in enumerate(block_list):
            if time.time() - start_time > time_limit:
                break
            
            unique = get_uncovered_for_block(b_idx)
            if not unique:
                if try_remove(b_idx):
                    improved = True
                    if len(block_list) < best_size:
                        best_size = len(block_list)
                    break
                continue
            
            # Find a replacement block that covers all unique t-subsets
            # Use intersection of t_to_blocks for each unique subset
            candidates = None
            for s in unique:
                if candidates is None:
                    candidates = set(t_to_blocks[s])
                else:
                    candidates &= set(t_to_blocks[s])
            
            if candidates is None or not candidates:
                continue
            
            candidates.discard(b_idx)
            
            # Pick the best candidate: one that maximizes additional coverage
            best_cand = None
            best_score = -1
            for cand in candidates:
                if cand in set(block_list):
                    continue
                # Score: how many currently covered (count>1) t-subsets does this add?
                # Plus how many unique ones it covers (all of them)
                score = len(unique)  # It must cover all unique
                for s in range(num_t):
                    if coverage_matrix[cand, s] and covered_count[s] > 1:
                        score += 0.5  # Small bonus for redundancy
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            if best_cand is not None:
                # Perform swap
                for s in range(num_t):
                    if coverage_matrix[b_idx, s]:
                        covered_count[s] -= 1
                for s in range(num_t):
                    if coverage_matrix[best_cand, s]:
                        covered_count[s] += 1
                block_list[i] = best_cand
                improved = True
                
                # Try to remove the new block if it's now redundant
                if try_remove(best_cand):
                    if len(block_list) < best_size:
                        best_size = len(block_list)
        
        if not improved:
            # No improvement in this pass; do a perturbation
            if len(block_list) > 10:
                # Randomly swap a block with a random one that helps
                for _ in range(10):
                    i = random.randint(0, len(block_list) - 1)
                    b_idx = block_list[i]
                    unique = get_uncovered_for_block(b_idx)
                    if not unique:
                        continue
                    candidates = None
                    for s in unique:
                        if candidates is None:
                            candidates = set(t_to_blocks[s])
                        else:
                            candidates &= set(t_to_blocks[s])
                    if candidates is None or not candidates:
                        continue
                    candidates.discard(b_idx)
                    if candidates:
                        cand = random.choice(list(candidates))
                        if cand not in set(block_list):
                            for s in range(num_t):
                                if coverage_matrix[b_idx, s]:
                                    covered_count[s] -= 1
                            for s in range(num_t):
                                if coverage_matrix[cand, s]:
                                    covered_count[s] += 1
                            block_list[i] = cand
                            break
    
    # Final cleanup
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(block_list)):
            b_idx = block_list[i]
            if try_remove(b_idx):
                changed = True
                break
    
    # Return as list of lists
    return [list(all_blocks[b_idx]) for b_idx in block_list]
