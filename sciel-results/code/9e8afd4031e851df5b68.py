import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
    
    # Compute triple coverage frequency: how many blocks cover each triple
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
        
    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    
    def entropy_gain(block_idx, load):
        """Compute Shannon entropy gain from adding a block to the current load distribution.
        
        The entropy of the load distribution H = -sum(load_i * log(load_i)) is maximized
        when load is as uniform as possible. Adding a block increases load on its triples.
        We want to choose the block that maximizes the entropy of the resulting distribution,
        which encourages balanced coverage across all triples.
        
        For efficiency, we compute the change in entropy:
        delta_H = -sum_{j in block} [new_load_j * log(new_load_j) - old_load_j * log(old_load_j)]
        where new_load_j = old_load_j + 1 for j in block, and unchanged otherwise.
        
        Since total load increases by c_k_t, we normalize by total load to get a proper
        probability distribution for entropy calculation.
        """
        old_loads = load[block_triples[block_idx]].astype(np.float64)
        new_loads = old_loads + 1.0
        
        # Use safe log: log(0) = 0 by convention
        old_entropy_terms = np.where(old_loads > 0, old_loads * np.log(old_loads), 0.0)
        new_entropy_terms = np.where(new_loads > 0, new_loads * np.log(new_loads), 0.0)
        
        # Entropy change (negative of sum of x*log(x) terms)
        # We want to maximize entropy, so we minimize the sum of x*log(x)
        return -(np.sum(new_entropy_terms) - np.sum(old_entropy_terms))
    
    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False, temperature=0.0, use_entropy=False):
        """Run greedy set cover, optionally weighted or entropy-based."""
        uncovered = uncovered_init.copy()
        selected = []
        load = np.zeros(n_triples, dtype=np.float64)  # Load distribution for entropy
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if use_entropy:
                # Maximum entropy sampling: choose block that maximizes entropy of load distribution
                # For speed, compute entropy gain for all blocks
                gains = np.zeros(n_blocks, dtype=np.float64)
                for i in range(n_blocks):
                    if np.any(uncovered[block_triples[i]]):  # Only consider blocks that cover uncovered triples
                        gains[i] = entropy_gain(i, load)
                    else:
                        gains[i] = -np.inf
            elif use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
                
            if gains.max() == 0:
                break
                
            if random_tie:
                # Softmax-based random selection for diversity
                if temperature > 0:
                    # Boltzmann selection
                    valid_gains = gains.copy()
                    valid_gains[valid_gains == -np.inf] = -1e10
                    scaled = valid_gains / temperature
                    scaled -= scaled.max()
                    probs = np.exp(scaled)
                    probs /= probs.sum()
                    best_idx = np.random.choice(n_blocks, p=probs)
                else:
                    max_gain = np.max(gains)
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            load[block_triples[best_idx]] += 1.0
            
        return selected
    
    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering using efficient coverage counting."""
        if not selected_indices:
            return selected_indices
            
        remaining = list(selected_indices)
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for block_idx in remaining:
            coverage_count[block_triples[block_idx]] += 1
            
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            for block_idx in remaining:
                if time.time() > deadline:
                    break
                    
                block_triple_indices = block_triples[block_idx]
                
                # Check if all triples in this block have coverage > 1
                if np.all(coverage_count[block_triple_indices] > 1):
                    coverage_count[block_triple_indices] -= 1
                    changed = True
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
        return remaining
    
    def local_search_improve(solution_blocks, max_iterations=100):
        """Try to improve solution by removing and swapping blocks."""
        if not solution_blocks:
            return solution_blocks
            
        current_blocks = list(solution_blocks)
        
        # Build coverage count
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for block_idx in current_blocks:
            coverage_count[block_triples[block_idx]] += 1
            
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Phase 1: Try removing blocks
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    return current_blocks
                    
                block_idx = current_blocks[i]
                block_triple_indices = block_triples[block_idx]
                
                if np.all(coverage_count[block_triple_indices] > 1):
                    coverage_count[block_triple_indices] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
            
            # Phase 2: Try 1-for-1 swaps
            if not improved:
                for i in range(len(current_blocks)):
                    if time.time() > deadline:
                        return current_blocks
                        
                    block_idx = current_blocks[i]
                    block_triple_indices = block_triples[block_idx]
                    
                    # Temporarily remove
                    coverage_count[block_triple_indices] -= 1
                    
                    uncovered_mask = coverage_count == 0
                    if not np.any(uncovered_mask):
                        current_blocks.pop(i)
                        improved = True
                        break
                    
                    uncovered_indices = np.where(uncovered_mask)[0]
                    
                    # Vectorized: find blocks covering all uncovered triples
                    # For each candidate block, count how many uncovered triples it covers
                    covers_uncovered = np.sum(uncovered_mask[block_triples], axis=1)
                    n_uncovered = len(uncovered_indices)
                    
                    # Blocks that cover ALL uncovered triples
                    full_cover_mask = covers_uncovered == n_uncovered
                    full_cover_mask[block_idx] = False  # exclude self
                    
                    # Also exclude blocks already in current solution
                    current_set = set(current_blocks)
                    for cb in current_blocks:
                        full_cover_mask[cb] = False
                    
                    if np.any(full_cover_mask):
                        candidates = np.where(full_cover_mask)[0]
                        # Among full-cover blocks, pick one that adds fewest new covered triples
                        # (i.e., maximizes overlap with already-covered triples)
                        best_new = candidates[0]
                        best_score = -1
                        for cand in candidates:
                            # Score: number of triples in this block that are already covered
                            already = np.sum(coverage_count[block_triples[cand]] > 0)
                            if already > best_score:
                                best_score = already
                                best_new = cand
                        
                        coverage_count[block_triples[best_new]] += 1
                        current_blocks[i] = best_new
                        improved = True
                        break
                    else:
                        # Restore coverage
                        coverage_count[block_triple_indices] += 1
                        continue
                else:
                    pass
        
        # Phase 3: Try 2-for-2 swaps (remove 2 blocks, add 2 different blocks)
        if len(current_blocks) >= 2 and time.time() < deadline:
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(current_blocks)):
                    if time.time() > deadline:
                        break
                        
                    b1 = current_blocks[i]
                    b2 = current_blocks[j]
                    
                    # Temporarily remove both
                    coverage_count[block_triples[b1]] -= 1
                    coverage_count[block_triples[b2]] -= 1
                    
                    uncovered_mask = coverage_count == 0
                    if not np.any(uncovered_mask):
                        current_blocks.pop(j)
                        current_blocks.pop(i)
                        improved = True
                        break
                    
                    uncovered_indices = np.where(uncovered_mask)[0]
                    n_unc = len(uncovered_indices)
                    
                    # Find pairs of blocks that together cover all uncovered
                    covers_unc = np.sum(uncovered_mask[block_triples], axis=1)
                    full_cover_mask = covers_unc == n_unc
                    full_cover_mask[b1] = False
                    full_cover_mask[b2] = False
                    current_set = set(current_blocks)
                    for cb in current_set:
                        full_cover_mask[cb] = False
                    
                    if np.sum(full_cover_mask) >= 2:
                        candidates = np.where(full_cover_mask)[0]
                        # Try all pairs of candidates
                        found = False
                        for a in range(len(candidates)):
                            for b in range(a+1, len(candidates)):
                                ca, cb = candidates[a], candidates[b]
                                # Check together they cover all uncovered
                                temp_cov = coverage_count.copy()
                                temp_cov[block_triples[ca]] += 1
                                temp_cov[block_triples[cb]] += 1
                                if np.all(temp_cov > 0):
                                    coverage_count[block_triples[ca]] += 1
                                    coverage_count[block_triples[cb]] += 1
                                    current_blocks[i] = ca
                                    current_blocks[j] = cb
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            improved = True
                            break
                    else:
                        # Restore
                        coverage_count[block_triples[b1]] += 1
                        coverage_count[block_triples[b2]] += 1
                    if found:
                        break
                if improved:
                    break
        
        return current_blocks
    
    # Run multiple greedy passes with different strategies
    best_solution = None
    best_size = float('inf')
    
    def try_solution(selected):
        nonlocal best_solution, best_size
        if selected:
            pruned = prune_blocks(selected)
            pruned = local_search_improve(pruned)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(blocks[i]) for i in pruned]
    
    # Strategy 1: Standard greedy (unweighted)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False)
    try_solution(selected)
    
    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    try_solution(selected)
    
    # Strategy 3: Maximum entropy greedy (deterministic)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False, use_entropy=True)
    try_solution(selected)
    
    # Strategy 4-20: Random restarts with varying strategies
    for restart in range(18):
        if time.time() > deadline - 15.0:
            break
            
        if restart < 6:
            # Entropy-based with randomness
            use_entropy = True
            use_weighted = True
            temp = 0.005 + (restart % 3) * 0.005
        elif restart < 12:
            # Weighted greedy with randomness
            use_entropy = False
            use_weighted = True
            temp = 0.01 + (restart % 3) * 0.01
        else:
            # Unweighted greedy with randomness
            use_entropy = False
            use_weighted = False
            temp = 0.005 + (restart % 3) * 0.005
            
        selected = greedy_cover(np.ones(n_triples, dtype=bool), 
                                use_weighted=use_weighted, 
                                random_tie=True, 
                                temperature=temp,
                                use_entropy=use_entropy)
        try_solution(selected)
    
    # Final: try to improve best solution further with more local search
    if best_solution is not None and time.time() < deadline - 5.0:
        best_indices = []
        for b in best_solution:
            for i, blk in enumerate(blocks):
                if list(blk) == b:
                    best_indices.append(i)
                    break
        improved = local_search_improve(best_indices, max_iterations=200)
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = [list(blocks[i]) for i in improved]
    
    if best_solution is None:
        return [list(b) for b in blocks[:100]]
        
    return best_solution
