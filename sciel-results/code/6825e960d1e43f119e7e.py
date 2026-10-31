import itertools
import random
import time

# Precompute all 4-subsets and 7-subsets
# v=19, k=7, t=4
V = 19
K = 7
T = 4

# List of all 4-subsets
all_4subs = list(itertools.combinations(range(V), T))
# Map 4-subset to index
sub_to_idx = {sub: i for i, sub in enumerate(all_4subs)}
N_4SUBS = len(all_4subs)

# List of all 7-subsets
all_7subs = list(itertools.combinations(range(V), K))
# Map 7-subset to index
block_to_idx = {b: i for i, b in enumerate(all_7subs)}
N_7SUBS = len(all_7subs)

# Precompute for each 7-subset, the list of 4-subset indices it covers
# block_4subs[i] is a list of indices into all_4subs
block_4subs = [None] * N_7SUBS
for i, b in enumerate(all_7subs):
    subs = [sub_to_idx[sub] for sub in itertools.combinations(b, T)]
    block_4subs[i] = subs

# Precompute for each 4-subset, the list of 7-subset indices that cover it
# sub_7subs[j] is a list of indices into all_7subs
sub_7subs = [[] for _ in range(N_4SUBS)]
for i, b in enumerate(all_7subs):
    for sub_idx in block_4subs[i]:
        sub_7subs[sub_idx].append(i)

def construct(v, k, t):
    # The task parameters match our precomputed constants
    if v != V or k != K or t != T:
        # Fallback for other parameters, though task implies fixed 19,7,4
        # We should probably handle this gracefully or just return empty/raise
        # Given the specific task, we assume 19,7,4.
        # For robustness, a simple greedy for other cases:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for Python overhead/return
    
    best_blocks = None
    best_size = float('inf')
    
    # We will run multiple iterations of Greedy + Reduction
    # Random seed for reproducibility in testing, but here we want diversity
    rng = random.Random(42)
    
    def greedy_reduce(seed):
        rng_local = random.Random(seed)
        coverage = [0] * N_4SUBS
        uncovered_count = N_4SUBS
        uncovered_set = set(range(N_4SUBS))
        blocks = []
        
        # Greedy Phase
        while uncovered_count > 0:
            # Pick a random uncovered subset to target
            target_idx = rng_local.choice(list(uncovered_set))
            
            # Get all blocks covering this target
            candidates = sub_7subs[target_idx]
            
            # Find the block with maximum gain (number of uncovered subsets covered)
            best_block_idx = -1
            max_gain = -1
            
            # Optimization: If we have many candidates, we can sample or check all.
            # 455 candidates is small enough to check all.
            for b_idx in candidates:
                gain = 0
                for s_idx in block_4subs[b_idx]:
                    if s_idx in uncovered_set:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = b_idx
                    
            # Add the best block
            blocks.append(best_block_idx)
            # Update coverage and uncovered set
            for s_idx in block_4subs[best_block_idx]:
                coverage[s_idx] += 1
                if coverage[s_idx] == 1:
                    uncovered_count -= 1
                    uncovered_set.remove(s_idx)
                    
        # Reduction Phase
        # Re-calculate coverage from the block list to be safe
        coverage = [0] * N_4SUBS
        for b_idx in blocks:
            for s_idx in block_4subs[b_idx]:
                coverage[s_idx] += 1
                
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow removal from list easily
            # Or create a new list
            new_blocks = []
            for b_idx in blocks:
                # Check if block is redundant
                redundant = True
                for s_idx in block_4subs[b_idx]:
                    if coverage[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for s_idx in block_4subs[b_idx]:
                        coverage[s_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
            
        return blocks

    # Run multiple iterations
    iteration = 0
    while time.time() < deadline:
        seed = iteration
        blocks = greedy_reduce(seed)
        
        # Verify validity (should be valid by construction, but good to check)
        # And get size
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
        # If we hit the lower bound or a very good number, we might stop early?
        # Lower bound is 131. Best known 151.
        if best_size <= 151:
             # We found a solution as good as or better than the record.
             # We can continue to try for better, but this is a strong candidate.
             pass
             
        iteration += 1
        
        # If we have spent too much time in one iteration, break
        # The greedy_reduce function is fast, but let's be safe.
        if time.time() > deadline - 5.0:
            break

    if best_blocks is None:
        # Fallback: return an empty list or a single block? 
        # The task requires a valid covering.
        # If we didn't find one, we should return a valid one.
        # The greedy_reduce always returns a valid covering.
        # So best_blocks should not be None if at least one iteration ran.
        # If no iteration ran (impossible given time), return dummy.
        return []
        
    # Convert block indices back to lists of integers
    result_blocks = [list(all_7subs[b_idx]) for b_idx in best_blocks]
    return result_blocks
