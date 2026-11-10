"""Optimized covering design C(19,8,5) using bitset-accelerated greedy with randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    total_start = time.time()
    total_deadline = total_start + 115.0  # Leave 5s safety margin

    # Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_total = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute for each point x, the bitset of t-subsets that contain x
    # A t-subset contains x if x is one of its elements
    point_bitsets = []
    for x in range(v):
        bits = np.zeros(num_total, dtype=np.uint8)
        for ts in all_tsubsets:
            if x in ts:
                bits[tsubset_to_idx[ts]] = 1
        point_bitsets.append(bits)

    # For each point x, precompute the list of t-subsets containing x (for gain computation)
    # Actually, we need: for a block B, gain of adding x = |{ts in uncovered : ts subset of B ∪ {x}}|
    # ts subset of B ∪ {x} and x in ts means ts = {x} ∪ rest where rest is a (t-1)-subset of B
    # So gain(x, B) = |{rest in C(B, t-1) : rest ∪ {x} is a t-subset and uncovered}|
    # We can precompute for each (x, frozenset_rest) whether rest ∪ {x} is a t-subset
    # But that's expensive. Instead, use the bitset approach:
    # gain(x, B) = popcount(uncovered_bitset & point_bitset[x] & mask_B)
    # where mask_B has 1s at positions of t-subsets that are subsets of B ∪ {x}
    # Actually, a t-subset ts is subset of B ∪ {x} iff ts ⊆ B ∪ {x}
    # This is equivalent to: ts contains x AND (ts \ {x}) ⊆ B, OR ts ⊆ B (but ts has size t and B has size < k)
    # More precisely: ts ⊆ B ∪ {x} means all elements of ts are in B or equal to x.
    # If x ∉ ts, then ts ⊆ B. If x ∈ ts, then ts \ {x} ⊆ B.
    # So the set of t-subsets covered by B ∪ {x} = {ts : ts ⊆ B ∪ {x}}
    # We can compute this as: for each x, the t-subsets containing x that have all other elements in B.
    # Let's precompute for each x, a mapping from (t-1)-subset of other points to the t-subset index.
    
    # Alternative efficient approach: for each x, precompute all t-subsets containing x,
    # and for each such t-subset, store the (t-1)-subset of other elements.
    # Then gain(x, B) = number of t-subsets ts containing x such that (ts \ {x}) ⊆ B and ts is uncovered.
    
    # Precompute: for each x, list of (t_subset_index, frozenset_of_other_elements)
    x_tsubsets = [[] for _ in range(v)]
    for ts in all_tsubsets:
        idx = tsubset_to_idx[ts]
        for x in ts:
            others = tuple(sorted(y for y in ts if y != x))
            x_tsubsets[x].append((idx, others))

    def _greedy_build(v, k, t, all_tsubsets, tsubset_to_idx, x_tsubsets, 
                      randomize=False, seed=42, max_time=10.0):
        """
        Greedy: repeatedly pick an uncovered t-subset, build a block around it.
        """
        rng = random.Random(seed)
        start_time = time.time()
        deadline = start_time + max_time
        
        covered = np.zeros(num_total, dtype=bool)
        num_uncovered = num_total
        blocks = []
        
        # Maintain list of uncovered indices for fast random selection
        uncovered_list = list(range(num_total))
        
        while num_uncovered > 0:
            if time.time() > deadline:
                break
            
            # Pick an uncovered t-subset to focus on
            if randomize:
                # Pick a random uncovered t-subset
                pos = rng.randrange(len(uncovered_list))
                idx = uncovered_list[pos]
            else:
                idx = uncovered_list[0]
            
            focus_tsubset = all_tsubsets[idx]
            block_set = set(focus_tsubset)
            
            # Greedily add points to maximize coverage
            while len(block_set) < k:
                best_point = None
                best_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count uncovered t-subsets that would be covered by adding x
                    # A t-subset ts is covered by B ∪ {x} if ts ⊆ B ∪ {x}
                    # This means: ts contains x and (ts \ {x}) ⊆ B, OR ts ⊆ B
                    # Since |ts| = t and |B| grows, let's just check directly using precomputed data
                    gain = 0
                    for (ts_idx, others) in x_tsubsets[x]:
                        if covered[ts_idx]:
                            continue
                        # Check if others ⊆ block_set
                        all_in = True
                        for o in others:
                            if o not in block_set:
                                all_in = False
                                break
                        if all_in:
                            gain += 1
                    # Also check t-subsets fully in B (not containing x)
                    # These are already covered if they were in B before, but we're adding x
                    # Actually, t-subsets fully in B were already counted when B was formed
                    # We only need to count NEW t-subsets covered by adding x
                    # A t-subset is newly covered if it contains x and (ts\{x}) ⊆ B
                    # So the above loop is correct.
                    
                    if gain > best_gain or (gain == best_gain and randomize and rng.random() < 0.15):
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                        elif gain == best_gain and best_point is not None and randomize and rng.random() < 0.3:
                            best_point = x
                
                if best_point is None:
                    break
                block_set.add(best_point)
            
            block = sorted(block_set)
            blocks.append(block)
            
            # Mark all t-subsets of this block as covered
            for ts in itertools.combinations(block, t):
                i = tsubset_to_idx.get(ts, -1)
                if i != -1 and not covered[i]:
                    covered[i] = True
                    num_uncovered -= 1
            
            # Remove newly covered from uncovered_list (amortized)
            # For efficiency, periodically rebuild
            if len(uncovered_list) > 0 and (len(blocks) % 50 == 0):
                uncovered_list = [i for i in uncovered_list if not covered[i]]
        
        return blocks

    def _greedy_best_block(v, k, t, all_tsubsets, tsubset_to_idx, x_tsubsets,
                           randomize=False, seed=42, max_time=10.0):
        """
        Greedy: at each step, find the k-subset that covers the most uncovered t-subsets.
        Use local search: start from a random k-subset or build around uncovered t-subsets.
        """
        rng = random.Random(seed)
        start_time = time.time()
        deadline = start_time + max_time
        
        covered = np.zeros(num_total, dtype=bool)
        num_uncovered = num_total
        blocks = []
        
        while num_uncovered > 0:
            if time.time() > deadline:
                break
            
            # Try multiple candidate blocks and pick the best
            best_block = None
            best_gain = -1
            
            # Candidate 1: build around a random uncovered t-subset
            # Find some uncovered t-subsets
            uncovered_sample = []
            for _ in range(200):
                i = rng.randrange(num_total)
                if not covered[i]:
                    uncovered_sample.append(i)
                    if len(uncovered_sample) >= 8:
                        break
            
            if not uncovered_sample:
                break
            
            for idx in uncovered_sample:
                focus = all_tsubsets[idx]
                block_set = set(focus)
                
                while len(block_set) < k:
                    best_point = None
                    best_local_gain = -1
                    
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for (ts_idx, others) in x_tsubsets[x]:
                            if covered[ts_idx]:
                                continue
                            all_in = True
                            for o in others:
                                if o not in block_set:
                                    all_in = False
                                    break
                            if all_in:
                                gain += 1
                        if gain > best_local_gain:
                            best_local_gain = gain
                            best_point = x
                        elif gain == best_local_gain and best_point is not None and randomize and rng.random() < 0.2:
                            best_point = x
                    
                    if best_point is None:
                        break
                    block_set.add(best_point)
                
                block = sorted(block_set)
                
                # Calculate total gain of this block
                gain = 0
                for ts in itertools.combinations(block, t):
                    i = tsubset_to_idx.get(ts, -1)
                    if i != -1 and not covered[i]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
            
            blocks.append(best_block)
            
            for ts in itertools.combinations(best_block, t):
                i = tsubset_to_idx.get(ts, -1)
                if i != -1 and not covered[i]:
                    covered[i] = True
                    num_uncovered -= 1
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Strategy 1: Deterministic greedy build
    blocks1 = _greedy_build(v, k, t, all_tsubsets, tsubset_to_idx, x_tsubsets, 
                            randomize=False, seed=1, max_time=15.0)
    if len(blocks1) < best_len:
        best_len = len(blocks1)
        best_blocks = blocks1
    
    if time.time() > total_deadline:
        return best_blocks if best_blocks else []
    
    # Strategy 2: Multiple randomized greedy builds
    seeds = [42, 123, 777, 2024, 999, 31415, 27182, 16180, 55555, 13579]
    for seed in seeds:
        if time.time() > total_deadline:
            break
        remaining_time = max(3.0, total_deadline - time.time())
        # Use shorter time per run since we have more runs
        blocks = _greedy_build(v, k, t, all_tsubsets, tsubset_to_idx, x_tsubsets,
                               randomize=True, seed=seed, max_time=min(8.0, remaining_time * 0.35))
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if time.time() > total_deadline:
        return best_blocks if best_blocks else []
    
    # Strategy 3: Greedy best-block approach
    blocks3 = _greedy_best_block(v, k, t, all_tsubsets, tsubset_to_idx, x_tsubsets,
                                 randomize=False, seed=1, max_time=12.0)
    if len(blocks3) < best_len:
        best_len = len(blocks3)
        best_blocks = blocks3
    
    if time.time() > total_deadline:
        return best_blocks if best_blocks else []
    
    # Strategy 4: Randomized best-block
    for seed in [42, 123, 777]:
        if time.time() > total_deadline:
            break
        remaining_time = max(3.0, total_deadline - time.time())
        blocks = _greedy_best_block(v, k, t, all_tsubsets, tsubset_to_idx, x_tsubsets,
                                    randomize=True, seed=seed, max_time=min(8.0, remaining_time * 0.4))
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    return best_blocks if best_blocks else []
# EVOLVE-BLOCK-END
