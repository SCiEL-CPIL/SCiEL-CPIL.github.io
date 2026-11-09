import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # 2. Precompute all k-subsets (blocks) and their coverage masks
    k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(k_subsets)
    
    block_masks = [0] * n_k_subsets
    
    # Inverted index: t_idx -> list of block indices covering it
    # This allows us to quickly find candidates for a specific uncovered t-subset
    candidates_for_t = [[] for _ in range(n_t_subsets)]
    
    for b_idx, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            candidates_for_t[idx].append(b_idx)
        block_masks[b_idx] = mask
        
    # 3. Randomized Greedy Set Cover with Restarts
    best_blocks = None
    best_len = float('inf')
    
    # Initial deterministic run to get a baseline
    # Actually, let's just run randomized restarts. The first one might be good.
    # We can also try a pure greedy first to establish a strong upper bound quickly.
    
    def run_greedy_randomized():
        uncovered = (1 << n_t_subsets) - 1
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline:
                return None # Timeout
            
            # Pick a random uncovered t-subset
            # Get a random set bit
            # We can pick a random integer in [0, n_t_subsets) and check if it's set?
            # No, better to pick from the list of set bits or use a random number generator
            # to pick a position.
            # Since n_t_subsets is ~4000, picking a random index and checking if set is fast.
            # But we need to ensure we pick an UNCOVERED one.
            # Let's pick a random index from 0 to n_t_subsets-1.
            # If it's not uncovered, try again?
            # Or, extract a random set bit efficiently.
            
            # Efficient random set bit extraction:
            # Convert to list of set bits? Too slow if done every step.
            # Just pick a random index.
            rand_idx = random.randint(0, n_t_subsets - 1)
            if not (uncovered >> rand_idx) & 1:
                # If not uncovered, we need to find one.
                # To avoid complex bit manipulation, we can just pick the first set bit?
                # No, that defeats the purpose of randomization.
                # Let's use a simpler heuristic: pick a random candidate from the 
                # candidates of a random t-subset that IS uncovered.
                
                # Let's just iterate a few times to find an uncovered one
                found = False
                for _ in range(10):
                    rand_idx = random.randint(0, n_t_subsets - 1)
                    if (uncovered >> rand_idx) & 1:
                        found = True
                        break
                if not found:
                    # Fallback: find first set bit
                    # This is rare if many are uncovered
                    for i in range(n_t_subsets):
                        if (uncovered >> i) & 1:
                            rand_idx = i
                            break
            
            # Get candidate blocks for this t-subset
            candidates = candidates_for_t[rand_idx]
            
            if not candidates:
                # Should not happen if a covering exists
                return None
                
            best_gain = -1
            best_block_idx = -1
            
            # Find the block among candidates that covers the most uncovered t-subsets
            for b_idx in candidates:
                mask = block_masks[b_idx]
                gain = (uncovered & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1 or best_gain == 0:
                # No progress possible?
                return None
                
            selected_indices.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        return selected_indices

    # Run a few restarts
    # We can also add a "local search" or "swap" phase, but greedy with restarts is a good start.
    
    # First, do a deterministic greedy to get a quick baseline
    # Deterministic: always pick the block with max gain.
    # We can simulate this by not randomizing the target, but picking the target that
    # leads to the best block? No, standard greedy picks max gain block globally.
    # Our randomized approach picks a random target, then best block for it.
    # This is slightly different from standard greedy.
    # Standard greedy:
    #   for each block, calc gain. pick max.
    # Randomized target greedy:
    #   pick random uncovered t. pick best block covering it.
    # The latter is a valid heuristic that diversifies.
    
    # Let's run the randomized version multiple times.
    
    # To ensure we have a valid result if time runs out, we should probably
    # finish the current run if it's close, or return the best completed run.
    # The function returns a list of blocks. If we return an incomplete run, it's invalid.
    # So we must only update `best_blocks` if the run completes successfully.
    
    num_restarts = 0
    while time.time() < deadline - 1.0: # Leave 1s for final conversion
        res = run_greedy_randomized()
        num_restarts += 1
        if res is not None:
            if len(res) < best_len:
                best_len = len(res)
                best_blocks = res
                # If we found a very good one, we might want to break or continue?
                # Continue to try to find better.
        
        # Optional: If we have spent enough time, break?
        # The loop condition handles it.
        
    if best_blocks is None:
        # Fallback: should not happen, but if it does, return empty or raise
        # The problem states a covering exists.
        return []
        
    return [list(k_subsets[i]) for i in best_blocks]

