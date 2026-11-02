import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe
    
    # Precompute all t-subsets and their indices
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    best_blocks = None
    best_count = float('inf')
    
    attempt = 0
    while time.time() < deadline:
        rng = random.Random(attempt)
        
        # Shuffle the order of subsets to prioritize different "hard" subsets
        shuffled_order = list(range(n_subs))
        rng.shuffle(shuffled_order)
        
        # Bitset for uncovered subsets
        uncovered = [True] * n_subs
        uncovered_count = n_subs
        
        blocks = []
        
        while uncovered_count > 0:
            # Find the first uncovered subset in the shuffled order
            idx_to_add = -1
            for idx in shuffled_order:
                if uncovered[idx]:
                    idx_to_add = idx
                    break
            
            if idx_to_add == -1:
                break
            
            current_block = list(all_subs[idx_to_add])
            
            # Greedily add points to current_block
            while len(current_block) < k:
                candidates = [x for x in range(v) if x not in current_block]
                
                best_gain = -1
                best_point = -1
                
                # Precompute (t-1) subsets of current_block
                partials = list(itertools.combinations(current_block, t-1))
                
                for x in candidates:
                    gain = 0
                    for p in partials:
                        new_sub = tuple(sorted(p + (x,)))
                        # new_sub is always a valid t-subset of range(v)
                        idx = sub_to_idx[new_sub]
                        if uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point != -1:
                    current_block.append(best_point)
                else:
                    # If no gain, pick the first candidate to ensure progress
                    current_block.append(candidates[0])
            
            current_block.sort()
            blocks.append(current_block)
            
            # Update uncovered
            for sub in itertools.combinations(current_block, t):
                idx = sub_to_idx[sub]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        
        # Local Search: Remove redundant blocks
        # Convert to list of tuples for hashing? No, list of lists is fine.
        # We need to check if a block is essential.
        
        # Optimization: Maintain a count of how many blocks cover each subset.
        # This makes removal check O(1) per subset.
        
        # Re-build coverage counts
        cover_count = [0] * n_subs
        for b in blocks:
            for sub in itertools.combinations(b, t):
                idx = sub_to_idx[sub]
                cover_count[idx] += 1
        
        # Remove blocks with no unique subsets
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(blocks):
                b = blocks[i]
                is_essential = False
                for sub in itertools.combinations(b, t):
                    idx = sub_to_idx[sub]
                    if cover_count[idx] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    # Remove block i
                    for sub in itertools.combinations(b, t):
                        idx = sub_to_idx[sub]
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    # Do not increment i
                else:
                    i += 1
        
        current_count = len(blocks)
        if current_count < best_count:
            best_count = current_count
            best_blocks = [list(b) for b in blocks]
        
        attempt += 1
        
        # Early exit if we hit a known good bound?
        # Best known is 76. If we get 76, it's likely optimal or near.
        # But we can keep trying to see if we get 75?
        # Given the time, just let it run.
        
    if best_blocks is None:
        return []
        
    return best_blocks
