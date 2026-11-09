"""Randomized greedy covering for C(19,7,4) with weighted stochastic selection."""
import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Randomized greedy set cover for covering design C(v,k,t).
    Uses weighted random selection for block construction to explore the solution space.
    """
    start_time = time.time()
    time_limit = 110  # seconds
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    def _get_gains(current_block, uncovered_set, v, k, t):
        """
        Calculate the gain for each candidate point not in current_block.
        Gain = number of uncovered t-subsets covered by adding this point.
        """
        block_set = set(current_block)
        gains = {}
        # Precompute combinations of current_block of size t-1 to speed up check
        # If len(current_block) < t-1, gain is 0 for all? 
        # No, if we add x, we form subsets of size t from (current_block + x).
        # A subset {x} U R where R is a (t-1)-subset of current_block.
        # So we only care if there are uncovered sets of that form.
        
        if len(current_block) < t - 1:
            # Can't form any t-subset containing x and t-1 others from current_block
            # Wait, if current_block has size m, and we add x, 
            # the new t-subsets are {x} U R where R is a (t-1)-subset of current_block.
            # If m < t-1, there are no such R, so gain is 0.
            for x in range(v):
                if x not in block_set:
                    gains[x] = 0
            return gains

        # Optimization: Group uncovered sets by their (t-1) subsets? 
        # That's complex. Let's just iterate.
        # For each x, count how many R in C(current_block, t-1) such that {x} U R is in uncovered_set.
        
        # To make this faster, let's iterate through uncovered_set? No, that's 3876 items.
        # Iterating through candidates (19) and combinations of block (max C(6,3)=20) is 19*20 = 380 checks.
        # This is fast enough.
        
        block_combos = list(itertools.combinations(current_block, t - 1))
        
        for x in range(v):
            if x in block_set:
                continue
            gain = 0
            for R in block_combos:
                # Create the tuple for the t-subset
                # R is a tuple of t-1 elements. Add x.
                # Need to sort to match the format in uncovered_set (which is sorted tuples)
                # It's faster to construct the sorted tuple directly if possible, 
                # but R is sorted. x might be anywhere.
                # Since v is small, sorting is fine.
                subset = tuple(sorted(R + (x,)))
                if subset in uncovered_set:
                    gain += 1
            gains[x] = gain
        return gains

    def _build_block(seed_subset, uncovered_set, v, k, t, rng, mode='weighted'):
        """
        Build a block starting with seed_subset.
        mode: 'greedy' (max gain), 'random' (uniform random from positive gain), 'weighted' (prob prop to gain)
        """
        block = list(seed_subset)
        block_set = set(block)
        
        while len(block) < k:
            gains = _get_gains(block, uncovered_set, v, k, t)
            
            # Filter candidates with positive gain
            candidates = [x for x, g in gains.items() if g > 0]
            
            if not candidates:
                # No gain, pick random remaining
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    block.append(rng.choice(remaining))
                    block_set.add(block[-1])
                else:
                    break
                continue
            
            if mode == 'greedy':
                max_gain = max(gains[x] for x in candidates)
                best_candidates = [x for x in candidates if gains[x] == max_gain]
                chosen = rng.choice(best_candidates)
            elif mode == 'random':
                chosen = rng.choice(candidates)
            elif mode == 'weighted':
                # Probability proportional to gain
                # Using sqrt(gain) to soften the preference slightly
                weights = [math.sqrt(gains[x]) for x in candidates]
                chosen = rng.choices(candidates, weights=weights, k=1)[0]
            else:
                chosen = candidates[0]
                
            block.append(chosen)
            block_set.add(chosen)
            
        return sorted(block)

    def _run_trial(trial_idx):
        """
        Run a single trial of the algorithm.
        Returns (blocks, success)
        """
        uncovered_local = set(all_t_subsets)
        blocks = []
        
        # Determine mode based on trial index or random
        # Mix of greedy and weighted random
        if trial_idx % 3 == 0:
            mode = 'greedy'
        elif trial_idx % 3 == 1:
            mode = 'weighted'
        else:
            mode = 'random'
            
        rng = random.Random(trial_idx * 1000 + 7)
        
        while uncovered_local:
            if time.time() - start_time > time_limit - 2:
                return None, False
                
            # Pick a random uncovered t-subset to anchor the next block
            # This ensures we target the hardest parts of the cover
            seed_subset = rng.choice(list(uncovered_local))
            
            block = _build_block(seed_subset, uncovered_local, v, k, t, rng, mode)
            
            # Remove covered subsets
            for combo in itertools.combinations(block, t):
                uncovered_local.discard(combo)
                
            blocks.append(block)
            
        return blocks, True

    # Run multiple trials
    for trial in range(100):
        if time.time() - start_time > time_limit - 5:
            break
            
        blocks, success = _run_trial(trial)
        
        if success and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # Fallback if no trial completed in time (should not happen with v=19)
    if best_blocks is None:
        # Simple greedy fallback
        uncovered_local = set(all_t_subsets)
        blocks = []
        while uncovered_local:
            seed_subset = min(uncovered_local)
            block = _build_block(seed_subset, uncovered_local, v, k, t, random.Random(0), 'greedy')
            for combo in itertools.combinations(block, t):
                uncovered_local.discard(combo)
            blocks.append(block)
        best_blocks = blocks
        
    return best_blocks
