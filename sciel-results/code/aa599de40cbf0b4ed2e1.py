import numpy as np
import itertools
import time

# Precomputation for v=19, t=5
V = 19
T = 5
K = 8

# 1. Map all t-subsets to indices
all_t_subsets = list(itertools.combinations(range(V), T))
N_T = len(all_t_subsets)
t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

# 2. Precompute indices for 4-subsets extended by x
# We need to map a 4-subset (tuple of 4 ints) to an index in a structure 
# that allows fast lookup of idx(4_subset + x)
# Let's create a dictionary: key = 4-subset, value = array of size V 
# where value[x] = t_subset_to_idx[sorted(4_subset + (x,))] if x not in 4_subset, else -1

four_subsets = list(itertools.combinations(range(V), 4))
four_subset_to_idx = {s: i for i, s in enumerate(four_subsets)}
N_4 = len(four_subsets)

# ext_idx[Q_idx, x] = t_subset_to_idx[Q + x]
ext_idx = np.full((N_4, V), -1, dtype=np.int32)
for i, Q in enumerate(four_subsets):
    Q_set = set(Q)
    for x in range(V):
        if x not in Q_set:
            S = tuple(sorted(Q + (x,)))
            ext_idx[i, x] = t_subset_to_idx[S]

# 3. Precompute 4-subsets for each 5-subset? 
# No, we build blocks incrementally.
# We need to know the 4-subsets of the current block B.
# When B has size j, its 4-subsets are combinations of its elements.
# We can compute these on the fly using itertools.combinations(B, 4) and map to indices.
# To speed up, we can precompute for every possible 5-subset (which is our starting point), 
# the list of its 4-subset indices.
# And when we add a point, we update the list of 4-subsets.

# Let's precompute the 4-subset indices for all 5-subsets
t_to_four_indices = np.zeros((N_T, 5), dtype=np.int32)
for i, S in enumerate(all_t_subsets):
    S_set = set(S)
    count = 0
    for Q in itertools.combinations(S, 4):
        t_to_four_indices[i, count] = four_subset_to_idx[Q]
        count += 1

# Function to get 4-subset indices for a block B (list of ints)
def get_four_indices(B):
    B_set = set(B)
    indices = []
    for Q in itertools.combinations(B, 4):
        indices.append(four_subset_to_idx[Q])
    return indices

def construct(v, k, t):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initialize
    uncovered = np.ones(N_T, dtype=bool)
    cover_counts = np.zeros(N_T, dtype=np.int32)
    blocks = []
    
    # Candidate points
    points = np.arange(v)
    
    while True:
        # Check time
        if time.time() > deadline:
            break
        
        if not np.any(uncovered):
            break
            
        # Find an uncovered t-subset
        # Pick the first one, or one with some heuristic?
        # For speed, just pick the first True index.
        # To improve quality, we might pick one that is "central" or has many potential extensions, 
        # but greedy on any uncovered is standard.
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_subset = all_t_subsets[start_idx]
        
        # Initialize block with start_subset
        block = list(start_subset)
        block_set = set(block)
        
        # Initialize 4-subset indices for the current block
        # Since block size is 5, we have 5 4-subsets
        current_four_indices = t_to_four_indices[start_idx].copy()
        
        # Extend block to size k
        while len(block) < k:
            if time.time() > deadline:
                break
                
            candidates = [x for x in range(v) if x not in block_set]
            if not candidates:
                break
                
            gains = np.zeros(v, dtype=np.int32)
            
            # For each 4-subset Q in current block, calculate gain for each candidate x
            for Q_idx in current_four_indices:
                # ext_idx[Q_idx] is an array of size v
                # We want to add 1 to gains[x] if uncovered[ext_idx[Q_idx, x]] is True
                # and x is a candidate.
                
                # Get indices of t-subsets formed by Q + x
                idxs = ext_idx[Q_idx]
                
                # Mask for candidates
                # We only care about x in candidates
                # Let's just iterate candidates for clarity, or vectorize if fast enough
                # v=19, so loop is fine.
                for x in candidates:
                    idx = idxs[x]
                    if idx != -1 and uncovered[idx]:
                        gains[x] += 1
            
            # Pick best candidate
            # If all gains are 0, pick any candidate (random or first)
            best_gain = np.max(gains[candidates])
            if best_gain == 0:
                # Pick a candidate that maximizes future potential? 
                # Or just random. Let's pick the one with highest "degree" in uncovered?
                # Simple: pick first candidate.
                best_x = candidates[0]
            else:
                # Pick candidate with max gain
                max_gain_candidates = [x for x in candidates if gains[x] == best_gain]
                # Tie-breaking: pick the one that covers the most *other* uncovered?
                # For now, random pick among ties to add diversity
                best_x = max_gain_candidates[0]
                
            # Add best_x to block
            block.append(best_x)
            block_set.add(best_x)
            
            # Update current_four_indices
            # New 4-subsets are formed by taking 3-subsets of the old block and adding best_x
            # Old block size was j. New size j+1.
            # New 4-subsets = { Q' where Q' is a 4-subset of new_block and best_x in Q' }
            # These correspond to 3-subsets Q_old of old_block such that Q_old U {best_x} is a 4-subset.
            
            # Get 3-subsets of the block BEFORE adding best_x
            # We can compute combinations of the previous block
            prev_block = block[:-1]
            new_four_indices = list(current_four_indices) # Keep old ones
            
            for Q3 in itertools.combinations(prev_block, 3):
                Q4 = tuple(sorted(Q3 + (best_x,)))
                new_four_indices.append(four_subset_to_idx[Q4])
                
            current_four_indices = np.array(new_four_indices, dtype=np.int32)
            
        # Block is complete (size k)
        if len(block) == k:
            block.sort()
            blocks.append(block)
            
            # Update uncovered and cover_counts
            # Get all 5-subsets of this block
            for S in itertools.combinations(block, t):
                idx = t_subset_to_idx[S]
                uncovered[idx] = False
                cover_counts[idx] += 1
    
    # Phase 2: Deletion Heuristic
    # Iterate through blocks and remove redundant ones
    # A block is redundant if all its t-subsets are covered by at least one other block.
    # i.e., cover_counts[S] > 1 for all S in block.
    
    # To make deletion efficient, we can do multiple passes.
    # We need to be careful with time.
    
    max_deletion_passes = 10
    for _ in range(max_deletion_passes):
        if time.time() > deadline:
            break
            
        removed_any = False
        new_blocks = []
        
        # Iterate in reverse? Or forward?
        # Forward is fine.
        for b in blocks:
            # Check if b is redundant
            redundant = True
            for S in itertools.combinations(b, t):
                idx = t_subset_to_idx[S]
                if cover_counts[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove b
                removed_any = True
                for S in itertools.combinations(b, t):
                    idx = t_subset_to_idx[S]
                    cover_counts[idx] -= 1
            else:
                new_blocks.append(b)
                
        blocks = new_blocks
        
        if not removed_any:
            break
            
    return blocks
