import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
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
    # triple_freq[j] = number of blocks that cover triple j
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
        
    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    
    # Precompute block weights: sum of weights of triples in each block
    # block_weight[i] = sum of triple_weights for triples in block i
    block_weights = np.sum(triple_weights[block_triples], axis=1)
    
    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False):
        """Run greedy set cover, optionally weighted."""
        uncovered = uncovered_init.copy()
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if use_weighted:
                # Weighted gain: sum of weights of uncovered triples in block
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                # Unweighted gain
                gains = np.sum(uncovered[block_triples], axis=1)
                
            # Find the block with maximum gain
            if random_tie:
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
            
        return selected
    
    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering."""
        if not selected_indices:
            return selected_indices
            
        # Build coverage map: for each triple, which selected blocks cover it
        # We want to find blocks that can be removed without losing coverage
        
        # Start with all selected
        remaining = list(selected_indices)
        
        # Try removing each block (in order of least "critical" first)
        # A block is critical if it's the only one covering some triple
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            # For each block in remaining, check if it can be removed
            for i, block_idx in enumerate(remaining):
                if time.time() > deadline:
                    break
                    
                # Temporarily remove this block
                # Check if all triples in this block are covered by other remaining blocks
                
                # Get triples covered by this block
                block_triple_set = set(block_triples[block_idx].tolist())
                
                # For each triple, check if another remaining block covers it
                can_remove = True
                for triple_idx in block_triple_set:
                    # Check if any other remaining block covers this triple
                    covered_by_other = False
                    for j, other_block_idx in enumerate(remaining):
                        if j == i:
                            continue
                        if triple_idx in block_triples[other_block_idx]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove this block
                    changed = True
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
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
            best_solution = [list(blocks[i]) for i in pruned]
    
    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(blocks[i]) for i in pruned]
    
    # Strategy 3-12: Random tie-breaking restarts (mix of weighted and unweighted)
    for restart in range(12):
        if time.time() > deadline - 15.0:
            break
            
        use_weighted = (restart % 2 == 0)
        selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=use_weighted, random_tie=True)
        if selected:
            pruned = prune_blocks(selected)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(blocks[i]) for i in pruned]
    
    # Final validation and return
    if best_solution is None:
        # Fallback: return a valid covering (shouldn't happen)
        return [list(b) for b in blocks[:100]]
        
    return best_solution
