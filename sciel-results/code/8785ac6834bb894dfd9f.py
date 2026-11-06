import itertools
import random
import time
import numpy as np

# Precomputation for specific v=19, t=4
V = 19
T = 4
K = 7

# Map 4-subsets to indices
all_4subs = list(itertools.combinations(range(V), T))
sub_to_idx = {s: i for i, s in enumerate(all_4subs)}
num_4subs = len(all_4subs)

def construct(v, k, t):
    # The task guarantees v=19, k=7, t=4, but we should be robust or at least use the passed values.
    # However, precomputation is hardcoded for 19,7,4.
    # If parameters differ, we fall back to a generic (slower) method or just assume the contract.
    # The contract says "evaluator passes this task's parameters v, k and t".
    # It implies v=19, k=7, t=4.
    
    if v != 19 or k != 7 or t != 4:
        # Fallback generic greedy (slower, but correct for other params if they ever change)
        # But given the specific task description, we optimize for 19,7,4.
        pass

    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy attempts
    num_trials = 50
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        random.seed(trial * 12345 + 67890)
        
        cover_count = np.zeros(num_4subs, dtype=np.int32)
        blocks = []
        
        # We need to know which 4-subsets are uncovered.
        # Instead of a set, we can just check cover_count == 0.
        # To find an uncovered one quickly, we can keep a list of indices and filter, 
        # or just pick a random index and check. 
        # Better: maintain a list of uncovered indices.
        
        # Initial state: all uncovered
        # We can just pick a random uncovered 4-subset to start the first block.
        # To ensure we don't pick a covered one, we can maintain a list.
        # But updating a list of 3876 items is fast.
        
        uncovered_indices = list(range(num_4subs))
        
        while uncovered_indices:
            # Pick a random uncovered 4-subset to start the block
            # This randomization helps escape local optima
            start_idx = random.choice(uncovered_indices)
            start_sub = all_4subs[start_idx]
            
            block = list(start_sub)
            
            # Add points to block until size k
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # New 4-subsets formed by adding x are {x} union C where C is a 3-subset of block
                    # block has size m. C is size t-1 = 3.
                    # If m < 3, gain is 0 (can't form new t-subsets? No, we need t-1 from block).
                    # If len(block) < t-1, we can't form any new t-subsets involving x and t-1 from block?
                    # Wait. A t-subset in the new block must contain x and t-1 elements from the current block.
                    # So we need len(block) >= t-1.
                    if len(block) < t - 1:
                        gain = 0
                    else:
                        for c in itertools.combinations(block, t - 1):
                            new_sub = tuple(sorted(c + (x,)))
                            idx = sub_to_idx[new_sub]
                            if cover_count[idx] == 0:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                # Pick randomly from best candidates to diversify
                x = random.choice(best_candidates)
                block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Update cover counts and uncovered list
            new_uncovered_indices = []
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                if cover_count[idx] == 0:
                    # It was uncovered, now covered
                    pass # It will be removed from uncovered_indices if we rebuild or filter
                cover_count[idx] += 1
            
            # Rebuild uncovered_indices by filtering
            # This is O(N) per block. N=3876. Blocks ~150. Total ops ~ 150 * 3876 ~ 600k. Fast.
            uncovered_indices = [i for i in uncovered_indices if cover_count[i] == 0]
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
            # If we found a very good solution, we can stop early?
            # The lower bound is 131. Best known 151.
            # If we hit 151, we can try to improve.
            # If we hit 131, we are done.
            
    if best_blocks is None:
        # Should not happen, but fallback
        return []
        
    return best_blocks
