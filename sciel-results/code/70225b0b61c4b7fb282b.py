import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Precompute block_triples: list of lists of triple indices for each block
    block_triples = []
    # Precompute triple_to_blocks: list of lists of block indices for each triple
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        b_triples = []
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            b_triples.append(t_idx)
            triple_to_blocks[t_idx].append(i)
        block_triples.append(b_triples)
        
    # Precompute block overlap matrix (how many triples two blocks share)
    # This is expensive for large n_blocks, so we compute it lazily or use a faster method
    # For C(24,7,3), n_blocks = C(24,7) = 346104 which is too large for full overlap matrix
    # Instead, we'll compute overlap on-the-fly using the triple_to_blocks structure
    
    def greedy_cover_annealed(random_seed, deadline):
        rng = np.random.RandomState(random_seed)
        
        # Initial gain for each block is k_t (all its triples are uncovered)
        gains = np.full(n_blocks, k_t, dtype=np.int32)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_mask = np.zeros(n_blocks, dtype=bool)
        
        selected_blocks = []
        remaining = n_triples
        
        # DTMC-inspired: maintain a "penalty" for blocks that overlap heavily with recent selections
        # This encourages diversity in the solution path
        recent_selections = []  # Keep track of last few selected blocks for overlap penalty
        max_recent = 50  # How many recent selections to consider for overlap penalty
        
        # Temperature schedule for annealing
        initial_temp = 1.0
        final_temp = 0.01
        n_total_steps_estimate = n_triples // k_t + 50  # Rough estimate
        
        while remaining > 0:
            if time.time() > deadline:
                break
                
            # Find max gain among unselected blocks
            gains[selected_mask] = -1
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 1:
                best_idx = candidates[0]
            else:
                # DTMC-inspired: bias selection toward blocks with low overlap with recent selections
                # Compute a score for each candidate: gain - lambda * overlap_with_recent
                # This creates a smooth surrogate that balances coverage gain against redundancy
                
                # Estimate overlap with recent selections
                # For each candidate, count how many of its triples were in recently selected blocks
                overlap_scores = np.zeros(len(candidates), dtype=np.float32)
                
                for i, cand_idx in enumerate(candidates):
                    # Count overlap with recent selections
                    overlap = 0
                    for recent_block in recent_selections[-max_recent:]:
                        # Count common triples between cand_idx and recent_block
                        # Use the triple lists to compute intersection size
                        cand_triples = set(block_triples[cand_idx])
                        recent_triples = set(block_triples[recent_block])
                        overlap += len(cand_triples & recent_triples)
                    overlap_scores[i] = overlap
                
                # Temperature-based selection
                # Higher temperature -> more exploration; lower -> more exploitation
                step = len(selected_blocks)
                temp = initial_temp * (final_temp / initial_temp) ** (step / max(n_total_steps_estimate, 1))
                
                # Score = gain - lambda * overlap, where lambda increases as temperature decreases
                lambda_overlap = (1.0 - temp) * 2.0  # Scale overlap penalty
                scores = max_gain - lambda_overlap * overlap_scores
                
                # Soft-max selection (smooth surrogate for argmax)
                # This creates a probability distribution biased toward high-score candidates
                scores_normalized = scores - np.max(scores)
                probs = np.exp(scores_normalized / max(temp, 1e-6))
                probs /= probs.sum()
                
                best_idx = candidates[rng.choice(len(candidates), p=probs)]
            
            # Select block
            selected_mask[best_idx] = True
            selected_blocks.append(list(all_blocks[best_idx]))
            recent_selections.append(best_idx)
            if len(recent_selections) > max_recent:
                recent_selections.pop(0)
            
            # Update gains for affected blocks
            newly_covered = []
            for t_idx in block_triples[best_idx]:
                if uncovered_mask[t_idx]:
                    uncovered_mask[t_idx] = False
                    newly_covered.append(t_idx)
                    remaining -= 1
            
            if newly_covered:
                for t_idx in newly_covered:
                    affected = triple_to_blocks[t_idx]
                    if affected:
                        gains[affected] -= 1
                        
        return selected_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Deadline guard: use 140s out of 150s to be safe
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Adaptive number of restarts based on time budget
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        blocks = greedy_cover_annealed(i, deadline)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # Final polish: try to remove redundant blocks from the best solution
    if best_blocks is not None and time.time() < deadline - 5.0:
        # Try to remove blocks that don't contribute unique coverage
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(best_blocks)):
                if time.time() > deadline - 5.0:
                    break
                # Temporarily remove block i and check if coverage is still complete
                test_blocks = best_blocks[:i] + best_blocks[i+1:]
                
                # Check if all triples are still covered
                covered_triples = set()
                for block in test_blocks:
                    for sub in itertools.combinations(block, t):
                        covered_triples.add(sub)
                
                if len(covered_triples) == n_triples:
                    # Block i is redundant
                    best_blocks = test_blocks
                    improved = True
                    break
    
    return best_blocks
