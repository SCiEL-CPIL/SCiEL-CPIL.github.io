import itertools
import random
import time
import numpy as np

# Precomputation
V = 19
K = 8
T = 5

# Map 5-subsets to indices
all_5_subs = list(itertools.combinations(range(V), T))
sub_index = {s: i for i, s in enumerate(all_5_subs)}
N_SUBS = len(all_5_subs) # 11628

# Precompute coverage for all 8-blocks
# block_id -> list of sub_indices
all_8_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_8_blocks) # 75582
block_coverage = []
for b in all_8_blocks:
    cov = [sub_index[s] for s in itertools.combinations(b, T)]
    block_coverage.append(cov)

# Map from 5-subset to list of block indices containing it
# sub_idx -> list of block_idxs
sub_to_blocks = [[] for _ in range(N_SUBS)]
for b_idx, b in enumerate(all_8_blocks):
    for s_idx in block_coverage[b_idx]:
        sub_to_blocks[s_idx].append(b_idx)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    def rodl_nibble():
        """
        Rödl nibble: randomized greedy that selects blocks with probability
        proportional to their coverage of currently uncovered t-subsets.
        This differs from deterministic greedy by using random selection,
        which can escape local optima and produce diverse initial solutions.
        """
        uncovered = [True] * N_SUBS
        blocks = []
        
        while True:
            if time.time() > deadline - 20:
                break
            
            # Find an uncovered subset
            target_idx = -1
            for i in range(N_SUBS):
                if uncovered[i]:
                    target_idx = i
                    break
            if target_idx == -1:
                break
            
            # Get all blocks covering this subset
            candidates = sub_to_blocks[target_idx]
            
            # Compute gain for each candidate (number of newly covered subsets)
            gains = []
            for b_idx in candidates:
                gain = 0
                for s_idx in block_coverage[b_idx]:
                    if uncovered[s_idx]:
                        gain += 1
                gains.append(gain)
            
            # Rödl nibble: select with probability proportional to gain
            # This creates a randomized selection that favors high-gain blocks
            # but allows lower-gain blocks to be selected occasionally
            total_gain = sum(gains)
            if total_gain == 0:
                # No progress possible, pick random
                b_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                # Weighted random selection
                r = random.random() * total_gain
                cumulative = 0
                b_idx = candidates[-1]
                for i, g in enumerate(gains):
                    cumulative += g
                    if r <= cumulative:
                        b_idx = candidates[i]
                        break
            
            # Add the block
            blocks.append(b_idx)
            for s_idx in block_coverage[b_idx]:
                uncovered[s_idx] = False
        
        return blocks

    def greedy_construct():
        uncovered = [True] * N_SUBS
        blocks = []
        while True:
            # Find an uncovered subset
            target_idx = -1
            for i in range(N_SUBS):
                if uncovered[i]:
                    target_idx = i
                    break
            if target_idx == -1:
                break
            
            # Get all blocks covering this subset
            candidates = sub_to_blocks[target_idx]
            
            # Pick the block that covers the most uncovered subsets
            best_block = -1
            max_gain = -1
            for b_idx in candidates:
                gain = 0
                for s_idx in block_coverage[b_idx]:
                    if uncovered[s_idx]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            # Add the block
            blocks.append(best_block)
            for s_idx in block_coverage[best_block]:
                uncovered[s_idx] = False
        
        return blocks

    def count_coverage(blocks):
        counts = [0] * N_SUBS
        for b_idx in blocks:
            for s_idx in block_coverage[b_idx]:
                counts[s_idx] += 1
        return counts

    def is_redundant(b_idx, counts):
        for s_idx in block_coverage[b_idx]:
            if counts[s_idx] <= 1:
                return False
        return True

    def local_search(blocks):
        counts = count_coverage(blocks)
        block_set = set(blocks)
        
        improved = True
        while improved:
            improved = False
            # Pass 1: Remove redundant blocks
            random.shuffle(blocks)
            for b_idx in blocks[:]: 
                if is_redundant(b_idx, counts):
                    for s_idx in block_coverage[b_idx]:
                        counts[s_idx] -= 1
                    blocks.remove(b_idx)
                    block_set.remove(b_idx)
                    improved = True
            
            # Pass 2: Try to replace a block with a better one
            if time.time() > deadline - 10:
                break

            for b_idx in blocks[:]:
                unique_subs = []
                for s_idx in block_coverage[b_idx]:
                    if counts[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    continue
                
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(sub_to_blocks[s_idx])
                    else:
                        candidates &= set(sub_to_blocks[s_idx])
                
                if not candidates:
                    continue
                
                candidates.discard(b_idx)
                
                if not candidates:
                    continue
                
                new_b_idx = next(iter(candidates))
                
                for s_idx in block_coverage[b_idx]:
                    counts[s_idx] -= 1
                for s_idx in block_coverage[new_b_idx]:
                    counts[s_idx] += 1
                
                blocks.remove(b_idx)
                blocks.append(new_b_idx)
                block_set.discard(b_idx)
                block_set.add(new_b_idx)
                improved = True
                break
            
            if time.time() > deadline - 5:
                break

        return blocks

    def lagrangian_polish(blocks):
        """
        Uses a heuristic inspired by Lagrangian relaxation / subgradient method.
        We maintain weights for uncovered subsets (or all subsets) to guide a greedy re-construction
        or a local search that prioritizes high-weight subsets.
        Here we use it to drive a "weighted greedy" reconstruction followed by local search.
        """
        best_blocks = blocks
        best_len = len(blocks)
        
        counts = count_coverage(blocks)
        
        # Start with weights 0
        weights = np.zeros(N_SUBS)
        
        for _ in range(10): # Limit iterations
            if time.time() > deadline - 15:
                break
            
            # Randomized version of the replacement pass
            new_blocks = list(blocks)
            new_counts = list(counts)
            
            improved = True
            while improved:
                improved = False
                # Remove redundants
                random.shuffle(new_blocks)
                for b_idx in new_blocks[:]:
                    if is_redundant(b_idx, new_counts):
                        for s_idx in block_coverage[b_idx]:
                            new_counts[s_idx] -= 1
                        new_blocks.remove(b_idx)
                        improved = True
                
                if time.time() > deadline - 5:
                    break

                # Randomized replacement
                for b_idx in new_blocks[:]:
                    unique_subs = []
                    for s_idx in block_coverage[b_idx]:
                        if new_counts[s_idx] == 1:
                            unique_subs.append(s_idx)
                    
                    if not unique_subs:
                        continue
                    
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = set(sub_to_blocks[s_idx])
                        else:
                            candidates &= set(sub_to_blocks[s_idx])
                    
                    if not candidates:
                        continue
                    
                    candidates.discard(b_idx)
                    
                    if not candidates:
                        continue
                    
                    # Pick a random candidate
                    new_b_idx = random.choice(list(candidates))
                    
                    for s_idx in block_coverage[b_idx]:
                        new_counts[s_idx] -= 1
                    for s_idx in block_coverage[new_b_idx]:
                        new_counts[s_idx] += 1
                    
                    new_blocks.remove(b_idx)
                    new_blocks.append(new_b_idx)
                    improved = True
                    break # Restart
                
                if len(new_blocks) < best_len:
                    best_blocks = new_blocks
                    best_len = len(new_blocks)
                    counts = new_counts
                    blocks = new_blocks

        return best_blocks

    # Generate multiple diverse initial solutions using Rödl nibble
    best_solution = None
    best_len = float('inf')
    
    # Try several randomized constructions
    n_trials = 5
    for trial in range(n_trials):
        if time.time() > deadline - 30:
            break
        
        # Rödl nibble construction
        candidate_blocks = rodl_nibble()
        
        # Local search to improve
        candidate_blocks = local_search(candidate_blocks)
        
        # Lagrangian-inspired polish
        candidate_blocks = lagrangian_polish(candidate_blocks)
        
        # Final redundant removal
        counts = count_coverage(candidate_blocks)
        candidate_blocks = [b for b in candidate_blocks if not is_redundant(b, counts)]
        
        if len(candidate_blocks) < best_len:
            best_len = len(candidate_blocks)
            best_solution = candidate_blocks
    
    # If Rödl nibble didn't produce a valid solution, fall back to greedy
    if best_solution is None:
        current_blocks = greedy_construct()
        current_blocks = local_search(current_blocks)
        current_blocks = lagrangian_polish(current_blocks)
        counts = count_coverage(current_blocks)
        current_blocks = [b for b in current_blocks if not is_redundant(b, counts)]
        best_solution = current_blocks
    
    # Convert indices back to blocks
    result = [list(all_8_blocks[b_idx]) for b_idx in best_solution]
    
    return result
