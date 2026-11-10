import itertools
import time
import random

def construct(v, k, t):
    # Parameters for this task are fixed, but we keep the signature general.
    # We assume v=19, k=8, t=5 based on the task description.
    # If parameters differ, we might need a fallback, but the task implies specific params.
    
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for potential bitset operations, but sets of tuples are fine here.
    
    best_blocks = None
    best_len = float('inf')
    
    # Points list
    points = list(range(v))
    
    # Precompute all possible blocks? No, we generate them on the fly based on the chosen t-subset.
    # But we need to be able to quickly find blocks containing a specific t-subset.
    
    def greedy_once(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a t-subset from uncovered. 
            # Heuristic: Pick one that is "rare" (covered by fewer potential blocks) or just random.
            # For simplicity and speed, let's pick a random one from uncovered.
            # To improve, we could calculate the number of blocks covering each uncovered subset, 
            # but that is expensive. Let's stick to random or first for now, 
            # and rely on multiple restarts.
            
            # Optimization: If we pick the first one, it's deterministic. 
            # Let's pick randomly to allow restarts to explore different paths.
            s = random.choice(list(uncovered))
            
            # Find all blocks of size k that contain s
            # s has t elements. We need to add k-t elements from the remaining v-t points.
            s_set = set(s)
            remaining_points = [p for p in points if p not in s_set]
            
            best_block = None
            best_gain = -1
            best_gain_tie_breaker = -1
            
            # Generate all combinations of k-t points from remaining_points
            # For v=19, k=8, t=5: need 3 points from 14. C(14,3) = 364.
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(s + combo))
                
                # Calculate gain: how many uncovered t-subsets does this block cover?
                # The block covers all t-subsets of itself.
                # We only care about those in `uncovered`.
                gain = 0
                # There are C(k, t) t-subsets in a block.
                # For k=8, t=5, C(8,5)=56.
                for sub in itertools.combinations(block, t):
                    if sub in uncovered:
                        gain += 1
                
                # Tie-breaking: if gain is same, pick one that covers more "rare" subsets?
                # Or just random tie-break.
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_gain_tie_breaker = random.random() # Not used yet, but structure is there
                elif gain == best_gain and gain > 0:
                    # Random tie break to diversify
                    if random.random() < 0.5:
                        best_block = block
            
            if best_block is None:
                # Should not happen if uncovered is not empty
                break
                
            blocks.append(best_block)
            
            # Remove covered t-subsets from uncovered
            # Get all t-subsets of best_block
            covered_subs = set(itertools.combinations(best_block, t))
            uncovered -= covered_subs
            
            # Time check
            if time.time() - start_time > time_limit - 5.0:
                return None # Abort this run if time is running out
        
        return blocks

    # Run multiple restarts
    # How many restarts? Depends on time.
    # Each restart takes roughly 335 steps * 364 blocks * 56 checks ~ 6.8M ops.
    # In Python, 6.8M ops might take ~1-2 seconds? 
    # Let's estimate. 1M simple ops ~ 0.1s. So 6.8M ~ 0.7s.
    # We have 110s. So we can do ~100 restarts.
    
    num_restarts = 0
    while time.time() - start_time < time_limit - 2.0:
        seed = num_restarts
        res = greedy_once(seed)
        num_restarts += 1
        
        if res is not None:
            # Validate? The greedy construction guarantees validity if it completes.
            # But let's double check length.
            if len(res) < best_len:
                best_len = len(res)
                best_blocks = res
        
        # If we find a very good solution, we might stop early?
        # The lower bound is 264. Best known is 335.
        # If we hit 264, we stop.
        if best_len <= 264:
            break

    if best_blocks is None:
        # Fallback: return a valid (though maybe large) covering
        # This should not happen if the loop runs at least once
        # But just in case, run a deterministic greedy once without time pressure if we have time
        if time.time() - start_time < time_limit - 5.0:
             best_blocks = greedy_once(0)
    
    return best_blocks
