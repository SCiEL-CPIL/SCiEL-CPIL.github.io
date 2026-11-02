import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # For t=3, precompute: for each pair (a,b), the list of 3-subset indices {a,b,z} for z != a,b
    # This makes gain calculation O(1) per pair lookup
    if t == 3:
        # pair_to_subsets[pair] = list of subset indices containing both elements of pair
        pair_to_subsets = defaultdict(list)
        for idx, sub in enumerate(all_subsets):
            for i in range(3):
                for j in range(i+1, 3):
                    pair = (sub[i], sub[j]) if sub[i] < sub[j] else (sub[j], sub[i])
                    pair_to_subsets[pair].append(idx)
    else:
        pair_to_subsets = None
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(seed_rng):
        """Greedy construction with fast gain calculation."""
        blocks = []
        # Use a list of booleans for uncovered tracking (faster than bit operations for large n)
        uncovered = [True] * n_subsets
        num_uncovered = n_subsets
        
        def get_gain(block, candidate_x):
            """Calculate how many new subsets are covered by adding x to block."""
            if len(block) < 2:
                return 0
            gain = 0
            if t == 3:
                # For each pair (a,b) in block, check subsets {x,a,b}
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        a, b = block[i], block[j]
                        pair = (a, b) if a < b else (b, a)
                        if pair in pair_to_subsets:
                            for idx in pair_to_subsets[pair]:
                                # Check if x is in this subset and it's uncovered
                                if x in all_subsets[idx] and uncovered[idx]:
                                    gain += 1
            else:
                # General fallback
                for combo in itertools.combinations(block, t-1):
                    for sub in all_subsets:
                        if x in sub and all(c in sub for c in combo):
                            idx = subset_to_idx[sub]
                            if uncovered[idx]:
                                gain += 1
                            break
            return gain
        
        while num_uncovered > 0:
            if time.time() > deadline:
                return None
            
            # Find a random uncovered subset
            seed_idx = -1
            for _ in range(50):
                candidate = seed_rng.randint(0, n_subsets - 1)
                if uncovered[candidate]:
                    seed_idx = candidate
                    break
            if seed_idx == -1:
                for i in range(n_subsets):
                    if uncovered[i]:
                        seed_idx = i
                        break
            if seed_idx == -1:
                break
            
            seed_sub = all_subsets[seed_idx]
            block = list(seed_sub)
            
            # Greedily add points to maximize gain
            while len(block) < k:
                block_set = set(block)
                best_point = None
                best_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = get_gain(block, x)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None or best_gain <= 0:
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = seed_rng.choice(available)
                    else:
                        break
                
                block.append(best_point)
            
            # Fill to size k if needed
            if len(block) < k:
                block_set = set(block)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if uncovered[idx]:
                    uncovered[idx] = False
                    num_uncovered -= 1
                
        return blocks
    
    def local_search(blocks):
        """Try to improve by removing redundant blocks and swapping."""
        blocks = list(blocks)
        
        # Build coverage count for each subset
        def compute_coverage(blocks_list):
            cov_count = [0] * n_subsets
            for block in blocks_list:
                for sub in itertools.combinations(block, t):
                    idx = subset_to_idx[sub]
                    cov_count[idx] += 1
            return cov_count
        
        # Remove redundant blocks iteratively
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            cov_count = compute_coverage(blocks)
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                block_i = blocks[i]
                redundant = True
                for sub in itertools.combinations(block_i, t):
                    idx = subset_to_idx[sub]
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        
        # Try block replacement
        max_replacement_attempts = 200
        for _ in range(max_replacement_attempts):
            if time.time() > deadline - 10:
                break
            if len(blocks) == 0:
                break
            
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            
            # Find subsets uniquely covered by old_block
            cov_count = compute_coverage(blocks)
            unique_subs = []
            for sub in itertools.combinations(old_block, t):
                idx = subset_to_idx[sub]
                if cov_count[idx] == 1:
                    unique_subs.append(sub)
            
            if not unique_subs:
                blocks.pop(i)
                continue
            
            # Build new block from union of unique subsets
            candidates = set()
            for sub in unique_subs:
                candidates.update(sub)
            
            if len(candidates) > k:
                continue
            
            # Try multiple random completions
            best_new_block = None
            best_new_unique = 0
            for _ in range(10):
                cand = set(candidates)
                while len(cand) < k:
                    x = random.randint(0, v - 1)
                    if x not in cand:
                        cand.add(x)
                
                new_block = tuple(sorted(cand))
                
                # Verify it covers all unique_subs
                valid = True
                for sub in unique_subs:
                    if not all(x in new_block for x in sub):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Count unique subsets of new_block (excluding block i)
                cov_without_i = compute_coverage([b for j, b in enumerate(blocks) if j != i])
                new_unique = 0
                for sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[sub]
                    if cov_without_i[idx] == 0:
                        new_unique += 1
                
                if new_unique > best_new_unique:
                    best_new_unique = new_unique
                    best_new_block = new_block
            
            if best_new_block is not None and best_new_unique > len(unique_subs):
                blocks[i] = list(best_new_block)
                blocks[i].sort()
        
        return blocks
    
    # Run multiple greedy restarts
    num_restarts = 0
    while time.time() < deadline - 30:
        seed_rng = random.Random(num_restarts)
        blocks = greedy_construct(seed_rng)
        if blocks is None:
            continue
        
        if len(blocks) < best_count:
            improved = local_search(blocks)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
        
        num_restarts += 1
    
    if best_blocks is None:
        if time.time() < deadline - 5:
            seed_rng = random.Random(999)
            blocks = greedy_construct(seed_rng)
            if blocks:
                return blocks
        return []
    
    return best_blocks
