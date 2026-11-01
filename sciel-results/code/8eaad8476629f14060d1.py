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
    
    # Precompute for each point x: list of subset indices that contain x
    # This helps in gain calculation
    point_subsets = defaultdict(list)
    for idx, sub in enumerate(all_subsets):
        for x in sub:
            point_subsets[x].append(idx)
    
    # For each pair (a,b), precompute the list of subset indices containing both a and b
    # This is useful for t=3: subsets containing {x,y} are those with the third element z
    pair_subsets = {}
    for idx, sub in enumerate(all_subsets):
        for i in range(t):
            for j in range(i+1, t):
                pair = tuple(sorted((sub[i], sub[j])))
                if pair not in pair_subsets:
                    pair_subsets[pair] = []
                pair_subsets[pair].append(idx)
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(seed_rng):
        """Greedy construction with bitset tracking."""
        blocks = []
        local_uncovered = (1 << n_subsets) - 1
        
        def get_gain(block, candidate_x):
            """Calculate how many new subsets are covered by adding x to block."""
            if len(block) == 0:
                return 0
            gain = 0
            # For t=3: subsets containing x and two elements from block
            # We need subsets {x, a, b} where a,b are in block
            if t == 3:
                for a, b in itertools.combinations(block, 2):
                    pair = tuple(sorted((a, b)))
                    if pair in pair_subsets:
                        for idx in pair_subsets[pair]:
                            sub = all_subsets[idx]
                            if x in sub:
                                if (local_uncovered >> idx) & 1:
                                    gain += 1
            else:
                # General case: subsets containing x and t-1 elements from block
                for combo in itertools.combinations(block, t-1):
                    for sub in all_subsets:
                        if x in sub and all(c in sub for c in combo):
                            idx = subset_to_idx[sub]
                            if (local_uncovered >> idx) & 1:
                                gain += 1
                                break
            return gain
        
        while local_uncovered != 0:
            if time.time() > deadline:
                return None
            
            # Find a random uncovered subset
            seed_idx = -1
            for _ in range(100):
                candidate = seed_rng.randint(0, n_subsets - 1)
                if (local_uncovered >> candidate) & 1:
                    seed_idx = candidate
                    break
            if seed_idx == -1:
                # Fallback: scan
                for i in range(n_subsets):
                    if (local_uncovered >> i) & 1:
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
                local_uncovered &= ~(1 << idx)
                
        return blocks
    
    def local_search(blocks):
        """Try to improve by removing redundant blocks and swapping."""
        blocks = list(blocks)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            
            # Remove redundant blocks
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                block_i = blocks[i]
                redundant = True
                for sub in itertools.combinations(block_i, t):
                    covered_by_other = False
                    for j in range(len(blocks)):
                        if i == j:
                            continue
                        if all(x in blocks[j] for x in sub):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        
        # Try block replacement: replace a block with one that covers more unique subsets
        for _ in range(50):  # Limited attempts
            if time.time() > deadline - 5:
                break
            if len(blocks) == 0:
                break
            
            # Pick a random block to potentially replace
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            
            # Find subsets uniquely covered by old_block
            unique_subs = []
            for sub in itertools.combinations(old_block, t):
                idx = subset_to_idx[sub]
                count = 0
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if all(x in blocks[j] for x in sub):
                        count += 1
                if count == 0:
                    unique_subs.append(sub)
            
            if not unique_subs:
                # Block is redundant, remove it
                blocks.pop(i)
                continue
            
            # Try to find a new block that covers all unique_subs and maybe more
            # Build a new block starting from union of unique_subs
            candidates = set()
            for sub in unique_subs:
                candidates.update(sub)
            
            if len(candidates) > k:
                # Too many candidates, skip
                continue
            
            # Add more points to reach size k
            while len(candidates) < k:
                x = random.randint(0, v - 1)
                if x not in candidates:
                    candidates.add(x)
            
            new_block = tuple(sorted(candidates))
            
            # Check if new_block covers all unique_subs
            valid = True
            for sub in unique_subs:
                if not all(x in new_block for x in sub):
                    valid = False
                    break
            
            if valid:
                # Check if new_block is better (covers more unique subsets than old_block)
                # Count unique subsets covered by new_block (excluding block i)
                new_unique = 0
                for sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[sub]
                    count = 0
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if all(x in blocks[j] for x in sub):
                            count += 1
                    if count == 0:
                        new_unique += 1
                
                old_unique = len(unique_subs)
                if new_unique > old_unique:
                    blocks[i] = list(new_block)
                    blocks[i].sort()
        
        return blocks
    
    # Run multiple greedy restarts
    num_restarts = 0
    while time.time() < deadline - 25:
        seed_rng = random.Random(num_restarts)
        blocks = greedy_construct(seed_rng)
        if blocks is None:
            continue
        
        if len(blocks) < best_count:
            # Apply local search
            improved = local_search(blocks)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
        
        num_restarts += 1
    
    if best_blocks is None:
        # Fallback
        if time.time() < deadline - 5:
            seed_rng = random.Random(999)
            blocks = greedy_construct(seed_rng)
            if blocks:
                return blocks
        return []
    
    return best_blocks
