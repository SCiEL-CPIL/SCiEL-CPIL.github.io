import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute t-indices for each block (used in local search)
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(idxs)
    
    # Precompute coverage masks as numpy uint64 array
    n_uint64s = (n_t + 63) // 64
    coverage_masks = np.zeros((n_blocks, n_uint64s), dtype=np.uint64)
    
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            word_idx = idx // 64
            bit_idx = idx % 64
            coverage_masks[i, word_idx] |= (1 << bit_idx)
    
    if time.time() > deadline - 30.0:
        # Fallback: standard greedy
        return _standard_greedy(v, k, t, all_blocks, coverage_masks, block_t_indices, n_t, n_uint64s, deadline)

    # 2. Entropy-Weighted Diverse Greedy
    # Weight uncovered triples by scarcity: triples with rare points get higher weight
    # This is inspired by the information-theoretic approach
    
    best_solution_blocks = None
    best_solution_size = float('inf')
    
    # Try multiple lambda values for overlap penalty
    lambdas = [0, 1, 2, 5, 10, 20]
    
    # Also try different random seeds for tie-breaking diversity
    random.seed(42)
    
    for lam in lambdas:
        if time.time() > deadline - 15.0:
            break
            
        # Run entropy-weighted greedy
        solution = _entropy_greedy(v, k, t, all_blocks, coverage_masks, block_t_indices, 
                                   n_t, n_uint64s, lam, deadline)
        
        if solution is not None:
            current_size = len(solution)
            if current_size < best_solution_size:
                best_solution_size = current_size
                best_solution_blocks = solution.copy()
    
    # Fallback if no valid solution found
    if best_solution_blocks is None:
        best_solution_blocks = _standard_greedy(v, k, t, all_blocks, coverage_masks, 
                                                 block_t_indices, n_t, n_uint64s, deadline)
        best_solution_size = len(best_solution_blocks)

    # 3. Local Search: Block Removal
    best_solution_blocks = _remove_redundant_blocks(best_solution_blocks, block_t_indices, n_t, deadline)
    
    # 4. Local Search: Swap/Replacement Moves
    if time.time() < deadline - 10.0:
        best_solution_blocks = _swap_local_search(v, k, t, best_solution_blocks, 
                                                   block_t_indices, n_t, deadline)
    
    # 5. Final removal pass after swaps
    best_solution_blocks = _remove_redundant_blocks(best_solution_blocks, block_t_indices, n_t, deadline)
    
    # Final Verification
    final_blocks = [list(all_blocks[i]) for i in best_solution_blocks]
    
    # Check validity
    test_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in best_solution_blocks:
        test_counts[block_t_indices[b_idx]] += 1
        
    if np.any(test_counts == 0):
        # Invalid. Fallback to standard greedy
        return _standard_greedy(v, k, t, all_blocks, coverage_masks, block_t_indices, 
                                 n_t, n_uint64s, deadline)

    return final_blocks


def _entropy_greedy(v, k, t, all_blocks, coverage_masks, block_t_indices, 
                     n_t, n_uint64s, lam, deadline):
    """
    Greedy covering with entropy-weighted gains.
    Uncovered triples are weighted by the rarity of their points.
    """
    covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
    selected_blocks = []
    
    # Track point frequencies in uncovered triples
    # Initially all points appear in C(v-1, t-1) triples
    point_freq = np.full(v, np.comb(v - 1, t - 1), dtype=np.float64)
    
    max_iterations = 200
    for _ in range(max_iterations):
        if time.time() > deadline - 15.0:
            break
            
        # Check if all covered
        if np.bitwise_count(covered_mask).sum() == n_t:
            break
            
        # Compute weighted gains for all blocks
        not_covered = ~covered_mask
        new_covered = coverage_masks & not_covered
        
        # For entropy weighting, we need to weight each newly covered triple
        # by the rarity of its points. We approximate this by computing
        # the sum of 1/(freq+1) for each point in each newly covered triple.
        
        # Vectorized approach: For each block, compute weighted gain
        # This is complex to fully vectorize, so we use a hybrid approach
        
        # First, compute unweighted new counts
        new_counts = np.bitwise_count(new_covered).sum(axis=1)
        
        # Compute overlap with existing
        union_mask = np.zeros(n_uint64s, dtype=np.uint64)
        for b_idx in selected_blocks:
            union_mask |= coverage_masks[b_idx]
        
        overlaps = coverage_masks & union_mask
        overlap_counts = np.bitwise_count(overlaps).sum(axis=1)
        
        # Base score
        scores = new_counts.astype(np.float64) - lam * overlap_counts.astype(np.float64)
        
        # Add entropy bonus: blocks that cover triples with rare points get bonus
        # Approximate: for each block, sum over its newly covered triples of
        # (1/freq[p1] + 1/freq[p2] + 1/freq[p3])
        
        # To keep this fast, we only compute the entropy bonus for the top candidates
        top_k = 100
        if new_counts.max() > 0:
            # Get top candidates by unweighted score
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            
            # Compute entropy bonus for top candidates
            for idx in top_indices:
                if new_counts[idx] == 0:
                    continue
                    
                # Get the t-indices for this block that are newly covered
                t_idxs = block_t_indices[idx]
                # Filter to only newly covered
                newly_covered_idxs = []
                for ti in t_idxs:
                    word_idx = ti // 64
                    bit_idx = ti % 64
                    if not_covered[word_idx] & (1 << bit_idx):
                        newly_covered_idxs.append(ti)
                
                if not newly_covered_idxs:
                    continue
                
                # Compute entropy weight for these triples
                entropy_bonus = 0.0
                for ti in newly_covered_idxs:
                    triple = all_t_subsets[ti]  # Need access to this
                    w = 0.0
                    for p in triple:
                        w += 1.0 / (point_freq[p] + 1)
                    entropy_bonus += w
                
                scores[idx] += 0.5 * entropy_bonus  # Scale factor
        
        # Pick the block with max score
        best_block_idx = np.argmax(scores)
        best_score = scores[best_block_idx]
        
        if new_counts[best_block_idx] == 0:
            break
        
        selected_blocks.append(best_block_idx)
        covered_mask |= coverage_masks[best_block_idx]
        
        # Update point frequencies
        for ti in block_t_indices[best_block_idx]:
            word_idx = ti // 64
            bit_idx = ti % 64
            if covered_mask[word_idx] & (1 << bit_idx):
                # This triple was already covered, don't update
                continue
            triple = all_t_subsets[ti]
            for p in triple:
                point_freq[p] -= 1
    
    # Check if valid
    if np.bitwise_count(covered_mask).sum() == n_t:
        return selected_blocks
    return None


def _standard_greedy(v, k, t, all_blocks, coverage_masks, block_t_indices, 
                      n_t, n_uint64s, deadline):
    """Standard greedy set cover."""
    covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
    selected_blocks = []
    
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
            
        not_covered = ~covered_mask
        intersections = coverage_masks & not_covered
        gains = np.bitwise_count(intersections).sum(axis=1)
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
        
        selected_blocks.append(best_idx)
        covered_mask |= coverage_masks[best_idx]
    
    return selected_blocks


def _remove_redundant_blocks(blocks, block_t_indices, n_t, deadline):
    """Remove redundant blocks from the solution."""
    blocks = list(blocks)
    
    if not blocks:
        return blocks
    
    # Initialize cover counts
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in blocks:
        cover_counts[block_t_indices[b_idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try to remove blocks in reverse order
        for i in range(len(blocks) - 1, -1, -1):
            b_idx = blocks[i]
            t_idxs = block_t_indices[b_idx]
            
            # Check if any t-subset has count 1
            if np.min(cover_counts[t_idxs]) > 1:
                # Safe to remove
                cover_counts[t_idxs] -= 1
                blocks.pop(i)
                improved = True
                break
    
    return blocks


def _swap_local_search(v, k, t, blocks, block_t_indices, n_t, deadline):
    """
    Try to swap blocks to reduce the total count.
    For each block, check if it can be replaced by another block that covers
    its essential triples and allows further removals.
    """
    blocks = list(blocks)
    
    if len(blocks) < 2:
        return blocks
    
    # Initialize cover counts
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in blocks:
        cover_counts[block_t_indices[b_idx]] += 1
    
    # Precompute all block indices for efficiency
    all_block_indices = set(range(len(block_t_indices)))
    
    improved = True
    max_swaps = 50
    swap_count = 0
    
    while improved and swap_count < max_swaps and time.time() < deadline - 5.0:
        improved = False
        
        for i in range(len(blocks)):
            if time.time() > deadline - 5.0:
                break
                
            b_idx = blocks[i]
            t_idxs = block_t_indices[b_idx]
            
            # Find triples covered only by this block
            essential = [ti for ti in t_idxs if cover_counts[ti] == 1]
            
            if not essential:
                # Can remove this block
                cover_counts[t_idxs] -= 1
                blocks.pop(i)
                improved = True
                swap_count += 1
                break
            
            # Try to find a replacement block that covers all essential triples
            # and has good coverage of other uncovered triples
            
            # Build a set of essential triple indices
            essential_set = set(essential)
            
            # Find candidate blocks that cover all essential triples
            candidates = []
            for cand_idx in all_block_indices:
                if cand_idx == b_idx:
                    continue
                
                cand_t_idxs = block_t_indices[cand_idx]
                cand_set = set(cand_t_idxs)
                
                # Check if candidate covers all essential triples
                if essential_set.issubset(cand_set):
                    # Compute how many new triples this candidate would cover
                    # (triples not currently covered by other blocks)
                    new_coverage = 0
                    for ti in cand_t_idxs:
                        if cover_counts[ti] == 0:
                            new_coverage += 1
                    
                    candidates.append((new_coverage, cand_idx))
            
            if candidates:
                # Pick the candidate with maximum new coverage
                candidates.sort(reverse=True)
                best_new_coverage, best_cand_idx = candidates[0]
                
                # Only swap if it improves coverage
                if best_new_coverage > 0:
                    # Perform the swap
                    cover_counts[t_idxs] -= 1  # Remove old block's coverage
                    cover_counts[block_t_indices[best_cand_idx]] += 1  # Add new block's coverage
                    blocks[i] = best_cand_idx
                    improved = True
                    swap_count += 1
                    break
    
    return blocks
