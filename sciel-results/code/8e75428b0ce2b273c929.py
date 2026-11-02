"""
Covering design C(v,k,t) using Simulated Annealing with bitmask representation.
Strategy:
1. Represent the set of possible blocks as a bitmask.
2. Use vectorized operations (NumPy) to compute coverage counts for all t-subsets.
3. Start with a valid covering (obtained via a fast greedy method).
4. Run Simulated Annealing to minimize the number of blocks (cost) while maintaining coverage (constraint).
   - Moves: Flip a bit (add or remove a block).
   - Cost function: Number of blocks + Penalty * (Uncovered t-subsets).
   - The penalty is high enough to prioritize validity, but allows temporary invalidity to escape local optima.
"""

import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # 1. Precomputation
    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # All possible k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Mapping from block index to list of t-subset indices it covers
    # block_to_tsubs[bi] = array of t-subset indices
    block_to_tsubs = []
    for block in all_blocks:
        indices = np.array([tsub_index[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)
        block_to_tsubs.append(indices)
    
    # Reverse mapping: t-subset index to list of block indices covering it
    # This is useful for greedy initialization and potentially for smarter moves
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, indices in enumerate(block_to_tsubs):
        for idx in indices:
            tsub_to_blocks[idx].append(bi)
    
    # Convert block_to_tsubs to a list of arrays for easier handling
    # For efficient coverage updates, we need to know which t-subsets a block covers.
    
    def compute_coverage(block_mask):
        """
        Compute coverage counts for all t-subsets given a mask of selected blocks.
        block_mask: boolean array of shape (n_blocks,)
        Returns: array of shape (n_tsubs,) with counts
        """
        # This is the bottleneck. 
        # Efficient way: For each t-subset, sum the bits of blocks that cover it.
        # Since n_tsubs (3876) and n_blocks (78035) are manageable, we can do this efficiently.
        # However, doing it from scratch every step is O(n_blocks * C(k,t)).
        # C(7,4) = 35. 78035 * 35 ~ 2.7M operations. In Python loop this is slow.
        # With NumPy, we can do it in one shot if we have a matrix, but the matrix is sparse and huge (3876 x 78035).
        
        # Alternative: Maintain coverage incrementally.
        # But for the sake of the SA step, we need to evaluate the delta.
        # Delta for removing block b: -1 for all tsubs in b.
        # Delta for adding block b: +1 for all tsubs in b.
        
        # We will maintain the coverage array incrementally in the main loop.
        # This function is for initialization or validation.
        
        cov = np.zeros(n_tsubs, dtype=np.int32)
        selected_blocks = np.where(block_mask)[0]
        for bi in selected_blocks:
            indices = block_to_tsubs[bi]
            cov[indices] += 1
        return cov

    def greedy_init():
        """
        Construct a valid covering using a randomized greedy algorithm.
        """
        rng = random.Random(42)
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        selected_mask = np.zeros(n_blocks, dtype=bool)
        
        while uncovered.any():
            # Find the t-subset with the fewest available covering blocks
            # To make it fast, we sample a few uncovered t-subsets
            uncovered_indices = np.where(uncovered == 1)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered t-subset
            # Or pick the one with minimum degree? 
            # For speed, let's just pick random from uncovered, but prioritize those with few options if we can check quickly.
            # Let's stick to random for now, it's faster.
            ts_idx = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
            
            covering_blocks = tsub_to_blocks[ts_idx]
            if not covering_blocks:
                # Should not happen if covering exists
                break
            
            # Pick the block that covers the most other uncovered t-subsets
            best_bi = -1
            best_gain = -1
            # Sample covering blocks if list is long
            if len(covering_blocks) > 20:
                candidates = rng.sample(covering_blocks, 20)
            else:
                candidates = covering_blocks
            
            for bi in candidates:
                if selected_mask[bi]:
                    continue
                # Calculate gain
                # How many currently uncovered t-subsets does this block cover?
                block_indices = block_to_tsubs[bi]
                gain = np.sum(uncovered[block_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            
            if best_bi == -1:
                # Fallback: just pick the first available
                for bi in covering_blocks:
                    if not selected_mask[bi]:
                        best_bi = bi
                        break
            
            if best_bi != -1:
                selected_mask[best_bi] = True
                # Update uncovered
                block_indices = block_to_tsubs[best_bi]
                uncovered[block_indices] = 0
        
        return selected_mask

    # 2. Initialization
    # Get a valid starting point
    mask = greedy_init()
    coverage = compute_coverage(mask)
    current_cost = np.sum(mask)
    
    # Ensure validity
    if not np.all(coverage >= 1):
        # If greedy failed (unlikely), force add blocks
        uncovered = np.where(coverage == 0)[0]
        for ts_idx in uncovered:
            for bi in tsub_to_blocks[ts_idx]:
                if not mask[bi]:
                    mask[bi] = True
                    coverage[block_to_tsubs[bi]] += 1
                    current_cost += 1
                    break
        # Recalculate just in case
        coverage = compute_coverage(mask)
        current_cost = np.sum(mask)

    best_mask = mask.copy()
    best_cost = current_cost
    best_coverage = coverage.copy()
    
    # 3. Simulated Annealing
    # Objective: Minimize (num_blocks + penalty * uncovered_count)
    # We want to reduce num_blocks.
    # Penalty for uncovered t-subsets must be high to keep solution valid mostly.
    # Let's use a dynamic penalty or a fixed high penalty.
    # If penalty is too high, we never leave valid states.
    # If penalty is too low, we end up with invalid solutions.
    # Strategy: 
    #   Cost = num_blocks
    #   Constraint: uncovered == 0
    #   We can use a method where we only accept moves that keep validity OR improve cost significantly?
    #   Standard SA on cost = num_blocks + P * uncovered.
    #   Start with high P, decrease? Or keep P high enough that invalid states are rare but possible.
    
    P = 100.0  # Penalty for each uncovered t-subset
    
    def get_cost(mask, coverage):
        return np.sum(mask) + P * np.sum(coverage == 0)
    
    current_energy = get_cost(mask, coverage)
    
    T = 10.0  # Initial temperature
    alpha = 0.99  # Cooling rate
    
    # Precompute list of all block indices for random selection
    all_block_indices = np.arange(n_blocks)
    
    steps = 0
    max_steps = 500000  # Hard cap on steps if time runs out
    
    while time.time() < deadline - 5 and steps < max_steps:
        # Select a random block to flip
        bi = rng.choice(all_block_indices)
        
        delta_blocks = 1 if not mask[bi] else -1
        block_indices = block_to_tsubs[bi]
        
        # Calculate change in uncovered count
        # If we add a block (delta_blocks = 1):
        #   Coverage increases by 1 for block_indices.
        #   Uncovered decreases where coverage was 0.
        # If we remove a block (delta_blocks = -1):
        #   Coverage decreases by 1 for block_indices.
        #   Uncovered increases where coverage was 1.
        
        if not mask[bi]:
            # Adding
            # New coverage: coverage + 1 on block_indices
            # Uncovered before: coverage == 0
            # Uncovered after: (coverage + 1) == 0 => coverage == -1 (impossible) 
            # Wait, if coverage was 0, it becomes 1. So uncovered decreases.
            # Number of newly covered t-subsets = sum(coverage[block_indices] == 0)
            newly_covered = np.sum(coverage[block_indices] == 0)
            delta_uncovered = -newly_covered
        else:
            # Removing
            # New coverage: coverage - 1 on block_indices
            # Uncovered before: coverage == 0
            # Uncovered after: (coverage - 1) == 0 => coverage == 1
            # Number of newly uncovered t-subsets = sum(coverage[block_indices] == 1)
            newly_uncovered = np.sum(coverage[block_indices] == 1)
            delta_uncovered = newly_uncovered
        
        delta_energy = delta_blocks + P * delta_uncovered
        
        # Acceptance probability
        if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
            # Accept move
            mask[bi] = not mask[bi]
            if not mask[bi]:
                # Added
                coverage[block_indices] += 1
            else:
                # Removed
                coverage[block_indices] -= 1
            
            current_cost = np.sum(mask)
            current_energy = get_cost(mask, coverage)
            
            if np.all(coverage >= 1):
                # Valid solution
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_mask = mask.copy()
                    best_coverage = coverage.copy()
        
        # Cooling
        T *= alpha
        steps += 1
        
        # Periodic check for time
        if steps % 1000 == 0 and time.time() > deadline - 10:
            break

    # 4. Final cleanup
    # If the best solution found is not valid (should be rare if P is high), try to repair it
    if not np.all(best_coverage >= 1):
        # Repair: Add blocks to cover uncovered t-subsets
        mask = best_mask.copy()
        coverage = best_coverage.copy()
        uncovered = np.where(coverage == 0)[0]
        for ts_idx in uncovered:
            for bi in tsub_to_blocks[ts_idx]:
                if not mask[bi]:
                    mask[bi] = True
                    coverage[block_to_tsubs[bi]] += 1
                    break
        # Recalculate
        coverage = compute_coverage(mask)
        best_cost = np.sum(mask)
        best_mask = mask
        best_coverage = coverage

    # Convert mask to list of blocks
    selected_indices = np.where(best_mask)[0]
    return [list(all_blocks[i]) for i in selected_indices]

