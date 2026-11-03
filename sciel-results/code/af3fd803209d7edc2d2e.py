import itertools
import random
import time

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin for 120s limit
    all_tsets = list(itertools.combinations(range(v), t))
    all_tset_set = set(all_tsets)
    
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        uncovered = set(all_tset_set)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-set to start the block
            # Converting to list is O(N), but N is small enough
            start_tset = random.choice(list(uncovered))
            block = list(start_tset)
            
            while len(block) < k:
                # Find point x that maximizes new t-sets covered
                best_gain = -1
                best_x = -1
                candidates = [x for x in range(v) if x not in block]
                
                # Shuffle candidates to break ties randomly
                random.shuffle(candidates)
                
                for x in candidates:
                    # Calculate gain: number of new t-sets formed by adding x
                    gain = 0
                    for s in itertools.combinations(block, t - 1):
                        # s is a tuple of t-1 elements
                        # new_tset is sorted tuple of s + (x,)
                        # We can optimize sorting if we maintain block sorted?
                        # But block is built incrementally.
                        # s + (x,) is not sorted.
                        # tuple(sorted(...)) is needed for set lookup.
                        if tuple(sorted(s + (x,))) in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        # Early exit if we covered all uncovered?
                        # Unlikely to cover all with one point.
                
                if best_x == -1:
                    # Should not happen
                    break
                block.append(best_x)
            
            block.sort()
            block_tuple = tuple(block)
            blocks.append(block_tuple)
            
            # Update uncovered
            for ts in itertools.combinations(block_tuple, t):
                if ts in uncovered:
                    uncovered.remove(ts)
        
        # Local Improvement: Remove redundant blocks
        # Build count of how many blocks cover each t-set
        tset_count = {}
        for b in blocks:
            for ts in itertools.combinations(b, t):
                tset_count[ts] = tset_count.get(ts, 0) + 1
        
        i = 0
        while i < len(blocks):
            b = blocks[i]
            redundant = True
            for ts in itertools.combinations(b, t):
                if tset_count[ts] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for ts in itertools.combinations(b, t):
                    tset_count[ts] -= 1
                blocks.pop(i)
                # Do not increment i
            else:
                i += 1
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Convert tuples to lists for return
    if best_blocks is None:
        return []
    return [list(b) for b in best_blocks]
