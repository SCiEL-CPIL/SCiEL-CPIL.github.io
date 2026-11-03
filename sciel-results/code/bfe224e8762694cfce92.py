"""Bitmask-based greedy set cover with randomized restarts for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a greedy set cover algorithm
    with bitmask acceleration and multiple randomized restarts.
    
    Returns a list of blocks, where each block is a list of k distinct integers in 0..v-1.
    """
    
    # Precompute all t-subsets and assign them indices 0 to n_t-1
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_sub_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute a bitmask for each block indicating which t-subsets it covers
    # Since n_t can be large, we use Python's arbitrary precision integers as bitmasks.
    block_masks = [0] * n_k
    for i, blk in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(blk, t):
            idx = t_sub_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # The initial state of uncovered t-subsets is all 1s (all bits set)
    initial_uncovered = (1 << n_t) - 1
    
    def run_greedy(seed):
        """
        Run a single greedy pass with a specific random seed for tie-breaking.
        Returns the list of blocks selected.
        """
        rng = random.Random(seed)
        uncovered = initial_uncovered
        used_blocks = set()
        selected_blocks = []
        
        # For efficiency, we can pre-sort blocks or just iterate.
        # Since n_k is ~50k, iterating is fine. 
        # We track which blocks are used to skip them.
        
        while uncovered:
            best_score = -1
            best_candidates = []
            
            # Iterate through all potential blocks
            for i in range(n_k):
                if i in used_blocks:
                    continue
                
                # Calculate how many new t-subsets this block covers
                # Intersection of uncovered set and block's coverage
                common = uncovered & block_masks[i]
                score = common.bit_count()
                
                if score > best_score:
                    best_score = score
                    best_candidates = [i]
                elif score == best_score and score > 0:
                    best_candidates.append(i)
                    
            if not best_candidates:
                break # No block can cover any new t-subsets (should not happen if valid)
            
            # Randomly select from the best candidates to diversify
            chosen_idx = rng.choice(best_candidates)
            used_blocks.add(chosen_idx)
            selected_blocks.append(list(k_subsets[chosen_idx]))
            
            # Update uncovered set
            uncovered &= ~block_masks[chosen_idx]
            
        return selected_blocks

    # Run multiple restarts with different seeds to find a better covering
    # Time limit is 120s. We can run many restarts.
    # Let's run for a fixed number of restarts or until time limit is approached.
    
    best_solution = None
    best_len = float('inf')
    
    # Estimate time per restart. 
    # Each restart takes roughly O(n_k * iterations * bit_ops).
    # Let's try 50 restarts first, or time-based.
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    seed = 42
    while time.time() - start_time < timeout:
        sol = run_greedy(seed)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
        # If we found a very good solution, we might break early? 
        # No, keep searching for better.
        seed += 1
        
    return best_solution

# EVOLVE-BLOCK-END
