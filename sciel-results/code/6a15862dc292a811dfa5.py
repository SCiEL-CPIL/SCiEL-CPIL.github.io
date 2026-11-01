"""Covering design C(19,8,5) - optimized greedy with bitmask and multi-restart."""
import itertools
import time
import random
import sys

# Precompute all t-subsets as bitmasks for fast coverage checks
# For v=19, k=8, t=5: there are C(19,5) = 11628 t-subsets


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline (20s safety margin before 160s kill)
    
    # Generate all t-subsets and assign indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset (as a frozenset/tuple) to its index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each point x, which t-subsets contain it
    # point_to_tsubsets[x] = list of t-subset indices containing x
    point_to_tsubsets = [[] for _ in range(v)]
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            point_to_tsubsets[x].append(i)
    
    # Precompute for each (t-1)-subset, which points can complete it to a t-subset
    # For a given (t-1)-subset S and point x not in S, the t-subset is S ∪ {x}
    # We need to quickly compute how many uncovered t-subsets a new point x covers
    # given the current block B (which has t-1 elements already in it)
    
    # For each (t-1)-subset, precompute the mapping to t-subset indices
    # (t-1)-subset is a tuple of t-1 sorted elements
    from_t1_to_tsubsets = {}
    for i, ts in enumerate(all_t_subsets):
        for j in range(t):
            # Remove element j to get a (t-1)-subset
            t1 = tuple(ts[:j] + ts[j+1:])
            if t1 not in from_t1_to_tsubsets:
                from_t1_to_tsubsets[t1] = []
            from_t1_to_tsubsets[t1].append(i)
    
    def greedy_cover(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # uncovered: set of t-subset indices not yet covered
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            # Pick the first uncovered t-subset to start a block
            # For better performance, pick one that has the most overlap potential
            # Simple: pick the minimum index for determinism, or random
            first = min(uncovered)
            initial_ts = all_t_subsets[first]
            block = list(initial_ts)
            
            # Greedily add points to the block
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                
                # For each candidate point not in block, compute gain
                # Gain = number of uncovered t-subsets that would be covered
                # A t-subset S is covered by block if S ⊆ block
                # When we add point x to block, the new t-subsets covered are those
                # t-subsets that contain x and (t-1) elements from the current block
                # and are currently uncovered
                
                # Precompute: for each (t-1)-subset of block, which t-subsets extend it
                # For each (t-1)-subset S of block, and point x not in S∪block... 
                # Actually, for each (t-1)-subset S of the current block, 
                # the t-subsets that are S ∪ {x} for x not in S
                # We need those that are in uncovered
                
                # Efficient approach: for each (t-1)-subset of block, get the list of
                # t-subset indices from from_t1_to_tsubsets, then count how many are in uncovered
                # and whose completing point is x
                
                # Actually, let's think differently. For a candidate point x:
                # The t-subsets newly covered are {S ∪ {x} : S is a (t-1)-subset of block, x ∉ S, S ∪ {x} is a valid t-subset}
                # Since S ⊆ block and x ∉ block, S ∪ {x} is a t-subset if |S| = t-1
                # So gain(x) = |{S ∈ C(block, t-1) : S ∪ {x} is a t-subset and idx(S∪{x}) ∈ uncovered}|
                
                # Precompute for each (t-1)-subset of block the list of (point, t-subset-idx) pairs
                block_t1_subsets = list(itertools.combinations(block, t-1))
                
                # For efficiency, for each candidate x, we can iterate over block_t1_subsets
                # and check if S ∪ {x} is a valid t-subset (it always is if x ∉ S)
                # and if its index is in uncovered
                
                # Build a lookup: for each (t-1)-subset S, map point x -> t-subset index
                # from_t1_to_tsubsets already gives us lists, but we need the mapping to specific points
                # Let's precompute: for each (t-1)-subset S, a dict from completing point to t-subset idx
                
                # Actually, from_t1_to_tsubsets[S] gives all t-subset indices that contain S as a (t-1)-subset
                # But we need to know which point completes each one. Since S has t-1 elements,
                # each t-subset containing S has exactly one additional element.
                
                # Let me precompute this more carefully
                # For each (t-1)-subset S (as a tuple), create a dict: point -> t_subset_idx
                # This is expensive to do on the fly, so let's precompute for all (t-1)-subsets
                
                # Hmm, this is getting complex. Let me use a different approach.
                
                # For each candidate x, compute gain by checking all (t-1)-subsets of block
                # and seeing if the resulting t-subset is uncovered
                # This is O(C(k-1, t-1) * (v-k)) per step, which for k=8, t=5 is C(7,4)*11 = 35*11 = 385
                # That's manageable.
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for S in block_t1_subsets:
                        # Check if x is in S (it shouldn't be since x not in block and S ⊆ block)
                        # Form the t-subset S ∪ {x}
                        # S is a tuple of t-1 sorted elements, x is not in S
                        # Create the sorted t-tuple
                        ts_with_x = tuple(sorted(S + (x,)))
                        # Look up its index
                        idx = t_subset_to_idx.get(ts_with_x, -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered: remove all t-subsets contained in this block
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx.get(tuple(ts), -1)
                if idx != -1:
                    uncovered.discard(idx)
            
            # Check deadline
            if time.time() > deadline:
                break
        
        return blocks
    
    # Try multiple seeds and keep the best result
    best_blocks = None
    best_len = float('inf')
    
    # First try deterministic greedy (seed=0)
    blocks = greedy_cover(seed=0)
    best_blocks = blocks
    best_len = len(blocks)
    
    # Try more seeds if time permits
    for seed in range(1, 100):
        if time.time() > deadline - 5:
            break
        blocks = greedy_cover(seed=seed)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
    
    # Convert tuples back to lists
    return [list(b) for b in best_blocks]
