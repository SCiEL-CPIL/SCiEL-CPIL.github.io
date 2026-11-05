import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute for each point x, the list of t-subsets containing x
    # and the "rest" (t-1 subset) for each such t-subset
    t_subsets_by_point = [[] for _ in range(v)]
    for s in all_t_subsets:
        for x in s:
            rest = tuple(sorted(s[:s.index(x)] + s[s.index(x)+1:]))
            t_subsets_by_point[x].append((s, rest))
            
    # Precompute all k-subsets? No, we construct them on the fly.
    
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Run multiple greedy instances
    for trial in range(50):
        if time.time() > deadline:
            break
            
        uncovered = [True] * num_t_subsets
        uncovered_count = num_t_subsets
        blocks = []
        
        # Shuffle points to vary the greedy choice
        points = list(range(v))
        random.shuffle(points)
        
        while uncovered_count > 0:
            # Find an uncovered t-subset
            # Instead of scanning all, we can just pick the first one or a random one
            # Picking the first one is deterministic. Let's pick a random uncovered one for diversity
            # But scanning for the first uncovered is O(N).
            # We can maintain a list of uncovered indices.
            
            # For efficiency, let's just pick the first uncovered one.
            # This makes the greedy deterministic for a given trial setup.
            # To add randomness, we can shuffle the order of t-subsets initially?
            # No, the greedy logic depends on the set of uncovered subsets.
            
            # Let's pick a random uncovered t-subset to start the block
            # We need a way to pick a random uncovered index.
            # We can maintain a set of uncovered indices.
            
            if not hasattr(construct, '_uncovered_set'):
                # This is just for logic, we'll use a list and scan
                pass
            
            # Find first uncovered
            start_idx = -1
            for i in range(num_t_subsets):
                if uncovered[i]:
                    start_idx = i
                    break
            if start_idx == -1:
                break
                
            start_subset = all_t_subsets[start_idx]
            block = list(start_subset)
            
            # Grow the block
            while len(block) < k:
                best_gain = -1
                best_point = None
                
                # For each candidate point not in block
                for x in points:
                    if x in block:
                        continue
                    
                    gain = 0
                    # Check t-subsets containing x
                    for s, rest in t_subsets_by_point[x]:
                        idx = t_subset_to_idx[s]
                        if not uncovered[idx]:
                            continue
                        # Check if rest is subset of block
                        # rest is a tuple of t-1 elements
                        # block is a list
                        is_subset = True
                        for y in rest:
                            if y not in block:
                                is_subset = False
                                break
                        if is_subset:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        # Optional: early break if gain is max possible?
                        # Max possible is number of uncovered t-subsets containing x and subset of B U {x}
                        # Not easy to compute max.
                    
                if best_point is None:
                    # Should not happen if uncovered_count > 0
                    break
                    
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            # For all t-subsets in this block, mark as covered
            for s in itertools.combinations(block, t):
                idx = t_subset_to_idx[s]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
    return best_blocks
