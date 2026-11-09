"""Improved probabilistic construction: Johnson-scheme-inspired Greedy + Optimized Local Search."""
import itertools
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Strategy:
    1. Precompute all k-subsets and their covered t-subsets as bitsets.
    2. Greedy construction using bitset popcount for fast gain calculation.
    3. Local search: Remove redundant blocks or replace necessary blocks with lower-overlap alternatives.
    """
    # Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute bitsets for each block
    # bitset[i] is an integer where bit j is 1 if tsub j is covered by block i
    block_bitsets = [0] * n_blocks
    # Precompute list of tsub indices for each block (for local search cost calculation)
    block_tsubs = []
    
    # Map tsub index to list of block indices that cover it
    triple_to_blocks = [[] for _ in range(n_tsubs)]
    
    for i, block in enumerate(all_blocks):
        bitset = 0
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            bitset |= (1 << idx)
            tsubs_in_block.append(idx)
            triple_to_blocks[idx].append(i)
        block_bitsets[i] = bitset
        block_tsubs.append(tsubs_in_block)
    
    # Phase 1: Greedy Construction
    random.seed(42)
    uncovered_mask = (1 << n_tsubs) - 1
    solution = []
    
    while uncovered_mask:
        best_block_idx = -1
        best_gain = -1
        best_candidates = []
        
        for i in range(n_blocks):
            gain = (block_bitsets[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_candidates = [i]
            elif gain == best_gain and gain > 0:
                best_candidates.append(i)
        
        if not best_candidates:
            break # Should not happen
            
        # Pick randomly from best candidates to add diversity (Johnson-like randomness)
        chosen_idx = random.choice(best_candidates)
        solution.append(chosen_idx)
        uncovered_mask &= ~block_bitsets[chosen_idx]
        
    # Phase 2: Local Search Minimization
    # Maintain coverage counts
    coverage_counts = np.zeros(n_tsubs, dtype=np.int32)
    for b_idx in solution:
        for t_idx in block_tsubs[b_idx]:
            coverage_counts[t_idx] += 1
    
    max_iterations = 50
    for _ in range(max_iterations):
        improved = False
        # Shuffle solution order to avoid bias
        random.shuffle(solution)
        
        for i in range(len(solution) - 1, -1, -1):
            b_idx = solution[i]
            
            # Temporarily remove block b_idx
            for t_idx in block_tsubs[b_idx]:
                coverage_counts[t_idx] -= 1
            
            # Find triples that are now uncovered (count == 0)
            newly_uncovered = []
            for t_idx in block_tsubs[b_idx]:
                if coverage_counts[t_idx] == 0:
                    newly_uncovered.append(t_idx)
            
            if not newly_uncovered:
                # Block was redundant, remove it permanently
                solution.pop(i)
                improved = True
                continue
            else:
                # Find replacement block
                # Must cover all triples in newly_uncovered
                # Candidate blocks are in the intersection of triple_to_blocks[t] for t in newly_uncovered
                
                # Start with the list of blocks for the first uncovered triple
                # Use the one with the smallest list to speed up intersection?
                # Or just intersect all.
                candidates = set(triple_to_blocks[newly_uncovered[0]])
                for t_idx in newly_uncovered[1:]:
                    candidates.intersection_update(triple_to_blocks[t_idx])
                
                # Exclude the block we just removed
                candidates.discard(b_idx)
                
                if not candidates:
                    # No single block can cover all newly uncovered triples
                    # Restore original
                    for t_idx in block_tsubs[b_idx]:
                        coverage_counts[t_idx] += 1
                    continue
                
                # Find candidate with minimum overlap with current coverage
                # Cost = sum of coverage_counts[t] for t in block
                # We want to minimize this to reduce redundancy
                best_candidate = -1
                best_cost = float('inf')
                
                for c_idx in candidates:
                    cost = 0
                    for t_idx in block_tsubs[c_idx]:
                        cost += coverage_counts[t_idx]
                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = c_idx
                
                if best_candidate != -1:
                    # Replace b_idx with best_candidate
                    solution[i] = best_candidate
                    for t_idx in block_tsubs[best_candidate]:
                        coverage_counts[t_idx] += 1
                    improved = True
                else:
                    # Restore original
                    for t_idx in block_tsubs[b_idx]:
                        coverage_counts[t_idx] += 1
        
        if not improved:
            break
    
    # Convert block indices back to lists of points
    return [list(all_blocks[i]) for i in solution]
