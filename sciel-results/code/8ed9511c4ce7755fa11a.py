import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Number of t-subsets in a k-set
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
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
    
    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False, temperature=0.0,
                     redundancy_penalty=0.0):
        """Run greedy set cover with optional structural redundancy penalty.
        
        redundancy_penalty > 0 penalizes blocks that overlap with already-selected
        blocks on uncovered triples, encouraging diverse selection that helps pruning.
        """
        uncovered = uncovered_init.copy()
        selected = []
        
        # Track which triples are covered by selected blocks (for redundancy calc)
        # selected_triple_count[i] = how many selected blocks cover triple i
        selected_triple_count = np.zeros(n_triples, dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if use_weighted:
                # Weighted gain: sum of weights of uncovered triples in block
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                # Unweighted gain
                gains = np.sum(uncovered[block_triples], axis=1)
                
            # Apply structural redundancy penalty
            if redundancy_penalty > 0 and len(selected) > 0:
                # For each block, count how many of its uncovered triples are also
                # covered by already-selected blocks (redundant coverage)
                # overlap[i] = number of triples in block i that are uncovered AND
                # already covered by selected blocks
                # We compute: for block i, sum over its triples of (uncovered AND selected_count > 0)
                # Use matrix ops: block_triples is (n_blocks, c_k_t), 
                # uncovered is (n_triples,), selected_triple_count is (n_triples,)
                # redundant_mask[j] = 1 if triple j is uncovered and selected_triple_count[j] > 0
                redundant_mask = uncovered & (selected_triple_count > 0)
                # redundancy_per_block[i] = number of triples in block i that are redundant
                redundancy_per_block = np.sum(redundant_mask[block_triples], axis=1)
                # Penalize: reduce gain by penalty * redundancy
                gains = gains - redundancy_penalty * redundancy_per_block
                # Ensure gains don't go negative (a block covering at least one unique
                # uncovered triple should still be selectable)
                gains = np.maximum(gains, 0.0)
                
            # Find the block with maximum gain
            if random_tie and temperature > 0:
                # Soft-max sampling with temperature
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                # Scale gains by temperature for soft-max
                scaled = gains / temperature
                # Numerical stability
                scaled -= np.max(scaled)
                probs = np.exp(scaled)
                probs /= np.sum(probs)
                best_idx = np.random.choice(n_blocks, p=probs)
            elif random_tie:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                    
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            selected_triple_count[block_triples[best_idx]] += 1
            
        return selected
    
    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering using efficient coverage tracking."""
        if not selected_indices:
            return selected_indices
            
        remaining = list(selected_indices)
        
        # Build coverage count for each triple: how many remaining blocks cover it
        # Use a dictionary or array for efficiency
        triple_coverage = np.zeros(n_triples, dtype=np.int32)
        for block_idx in remaining:
            triple_coverage[block_triples[block_idx]] += 1
            
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            # Try removing each block
            for i, block_idx in enumerate(remaining):
                if time.time() > deadline:
                    break
                    
                # Check if this block can be removed
                # It can be removed if every triple it covers has coverage > 1
                block_triples_set = block_triples[block_idx]
                
                can_remove = True
                for triple_idx in block_triples_set:
                    if triple_coverage[triple_idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove this block and update coverage
                    triple_coverage[block_triples_set] -= 1
                    changed = True
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
        return remaining
    
    def local_search(selected_indices, max_iters=500):
        """Local search: try to improve by removing redundant blocks and
        performing targeted swaps to enable further removals."""
        if not selected_indices:
            return selected_indices
            
        remaining = list(selected_indices)
        
        # Build coverage count for each triple
        triple_coverage = np.zeros(n_triples, dtype=np.int32)
        for block_idx in remaining:
            triple_coverage[block_triples[block_idx]] += 1
            
        # Track which blocks cover each triple (for swap operations)
        triple_to_blocks = {}
        for block_idx in remaining:
            for triple_idx in block_triples[block_idx]:
                if triple_idx not in triple_to_blocks:
                    triple_to_blocks[triple_idx] = set()
                triple_to_blocks[triple_idx].add(block_idx)
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove all redundant blocks (fixed point)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_remaining = []
                for i, block_idx in enumerate(remaining):
                    if time.time() > deadline:
                        break
                    block_triples_set = block_triples[block_idx]
                    can_remove = True
                    for triple_idx in block_triples_set:
                        if triple_coverage[triple_idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        triple_coverage[block_triples_set] -= 1
                        for triple_idx in block_triples_set:
                            triple_to_blocks[triple_idx].discard(block_idx)
                        changed = True
                    else:
                        new_remaining.append(block_idx)
                remaining = new_remaining
                if changed:
                    improved = True
                    
            # Phase 2: Try targeted swaps to enable further removals
            if time.time() > deadline:
                break
                
            # Identify blocks with the most "critical" (singly-covered) triples
            # Swapping these might allow removal of other blocks
            for _ in range(min(max_iters, 200)):
                if time.time() > deadline:
                    break
                    
                if len(remaining) < 2:
                    break
                    
                # Pick a block with the most critical triples (hardest to remove)
                best_swap_idx = -1
                best_critical_count = 0
                for idx, block_idx in enumerate(remaining):
                    critical_count = 0
                    for triple_idx in block_triples[block_idx]:
                        if triple_coverage[triple_idx] == 1:
                            critical_count += 1
                    if critical_count > best_critical_count:
                        best_critical_count = critical_count
                        best_swap_idx = idx
                        
                if best_swap_idx < 0 or best_critical_count == 0:
                    break
                    
                old_block = remaining[best_swap_idx]
                
                # Get triples that would become uncovered if we remove old_block
                critical_triples = []
                for triple_idx in block_triples[old_block]:
                    if triple_coverage[triple_idx] == 1:
                        critical_triples.append(triple_idx)
                        
                # Find candidate blocks that cover all critical_triples
                # Score by: number of additional over-covered triples (enables removals)
                candidates = []
                for candidate_block in range(n_blocks):
                    if candidate_block in remaining:
                        continue
                    # Quick check: does candidate cover all critical triples?
                    candidate_triple_set = set(block_triples[candidate_block].tolist())
                    if not all(t in candidate_triple_set for t in critical_triples):
                        continue
                    # Calculate benefit: how many triples it covers that are
                    # currently over-covered (coverage >= 2), which could enable
                    # removal of other blocks
                    additional = 0
                    for triple_idx in block_triples[candidate_block]:
                        if triple_coverage[triple_idx] >= 2:
                            additional += 1
                    candidates.append((additional, candidate_block))
                        
                if not candidates:
                    break
                    
                # Pick the best candidate
                candidates.sort(reverse=True)
                _, new_block = candidates[0]
                
                # Perform the swap
                triple_coverage[block_triples[old_block]] -= 1
                for triple_idx in block_triples[old_block]:
                    triple_to_blocks[triple_idx].discard(old_block)
                    
                triple_coverage[block_triples[new_block]] += 1
                for triple_idx in block_triples[new_block]:
                    if triple_idx not in triple_to_blocks:
                        triple_to_blocks[triple_idx] = set()
                    triple_to_blocks[triple_idx].add(new_block)
                    
                remaining[best_swap_idx] = new_block
                improved = True
                
        return remaining
    
    # Run multiple greedy passes with different strategies
    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Standard greedy (unweighted)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(all_blocks[i]) for i in pruned]
    
    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(all_blocks[i]) for i in pruned]
    
    # Strategy 3: Weighted greedy with redundancy penalty (encourages diversity)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False,
                            redundancy_penalty=0.5)
    if selected:
        pruned = prune_blocks(selected)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(all_blocks[i]) for i in pruned]
    
    # Strategy 4: Weighted greedy with higher redundancy penalty + local search
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False,
                            redundancy_penalty=1.0)
    if selected:
        pruned = prune_blocks(selected)
        local_search_result = local_search(pruned)
        if len(local_search_result) < best_size:
            best_size = len(local_search_result)
            best_solution = [list(all_blocks[i]) for i in local_search_result]
    
    # Strategy 5-16: Random tie-breaking restarts with various parameters
    for restart in range(12):
        if time.time() > deadline - 25.0:
            break
            
        use_weighted = (restart % 2 == 0)
        temperature = 0.5 + 0.5 * restart
        # Vary redundancy penalty: 0, 0.3, 0.7, 1.2 cycling
        red_pen = [0.0, 0.3, 0.7, 1.2][restart % 4]
        
        selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=use_weighted, 
                               random_tie=True, temperature=temperature,
                               redundancy_penalty=red_pen)
        if selected:
            pruned = prune_blocks(selected)
            # Apply local search to some restarts
            if restart % 3 == 0 and time.time() < deadline - 15.0:
                local_search_result = local_search(pruned)
                if len(local_search_result) < best_size:
                    best_size = len(local_search_result)
                    best_solution = [list(all_blocks[i]) for i in local_search_result]
            else:
                if len(pruned) < best_size:
                    best_size = len(pruned)
                    best_solution = [list(all_blocks[i]) for i in pruned]
    
    # Final polish: try local search on the best solution found
    if best_solution is not None and time.time() < deadline - 5.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_solution]
        polished = local_search(best_indices, max_iters=300)
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = [list(all_blocks[i]) for i in polished]
    
    # Final validation and return
    if best_solution is None:
        # Fallback: return a valid covering (shouldn't happen)
        return [list(b) for b in all_blocks[:100]]
        
    return best_solution
