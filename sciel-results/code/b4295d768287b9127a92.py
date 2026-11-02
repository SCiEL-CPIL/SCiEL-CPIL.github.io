import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin for safety

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)

    # Precompute bitset masks for each possible block
    # For each k-subset, store a bitmask of which t-subsets it covers
    # Since n_t_subs = 11628, we need multiple 64-bit words or use numpy
    n_words = (n_t_subs + 63) // 64  # 183 words

    # Precompute masks for all C(19,8) blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build mask for each block: list of (word_idx, bit) pairs
    block_masks = []
    for block in all_blocks:
        mask = [0] * n_words
        for sub in itertools.combinations(block, t):
            idx = t_sub_index[sub]
            mask[idx // 64] |= (1 << (idx % 64))
        block_masks.append(mask)
    
    # Also store the t-subset indices for each block (for count updates)
    block_t_indices = []
    for block in all_blocks:
        indices = [t_sub_index[sub] for sub in itertools.combinations(block, t)]
        block_t_indices.append(indices)

    # Block index lookup
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}

    # Helper to get indices of t-subsets of a block
    def get_block_t_sub_indices(block):
        return [t_sub_index[sub] for sub in itertools.combinations(block, t)]

    # 1. Greedy construction with bitset optimization
    def greedy_construct(seed, time_limit):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered = set(range(n_t_subs))
        
        while uncovered and time.time() < time_limit:
            # Pick a random uncovered subset
            idx = random.choice(list(uncovered))
            target_sub = all_t_subs[idx]
            
            block = list(target_sub)
            
            while len(block) < k:
                best_p = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block]
                
                for p in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        sub = tuple(sorted(rest + (p,)))
                        sub_idx = t_sub_index.get(sub, -1)
                        if sub_idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                    elif gain == best_gain and best_p != -1:
                        if random.random() < 0.5:
                            best_p = p
                
                if best_p == -1:
                    break
                block.append(best_p)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx2 = t_sub_index[sub]
                counts[idx2] += 1
                if counts[idx2] == 1:
                    uncovered.discard(idx2)
                    
        return blocks

    # 2. Local Search with bitset optimization
    def local_search(initial_blocks, time_limit):
        blocks = [list(b) for b in initial_blocks]
        counts = [0] * n_t_subs
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[t_sub_index[sub]] += 1
                
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        
        # Precompute for each block in current set: which t-subsets are uniquely covered
        # For faster redundancy check
        
        while time.time() < time_limit:
            # Try to remove redundant blocks
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    b = blocks[i]
                    # Check if all t-subsets of b have count > 1
                    removable = True
                    for sub in itertools.combinations(b, t):
                        if counts[t_sub_index[sub]] <= 1:
                            removable = False
                            break
                    if removable:
                        # Remove block i
                        for sub in itertools.combinations(b, t):
                            counts[t_sub_index[sub]] -= 1
                        blocks.pop(i)
                        changed = True
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best_blocks = [list(x) for x in blocks]
                        break  # Restart loop as indices changed
            
            if time.time() >= time_limit:
                break
            
            # Perform random swaps with faster evaluation
            n_swaps = 200
            for _ in range(n_swaps):
                if not blocks:
                    break
                i = random.randint(0, len(blocks)-1)
                b = blocks[i]
                x = random.choice(b)
                y = random.choice([p for p in range(v) if p not in b])
                
                # Check validity: all t-subsets of b containing x must have count > 1
                valid = True
                for sub in itertools.combinations(b, t):
                    if x in sub and counts[t_sub_index[sub]] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    # Decrement counts for subsets of b containing x
                    for sub in itertools.combinations(b, t):
                        if x in sub:
                            counts[t_sub_index[sub]] -= 1
                    
                    # Update block
                    b.remove(x)
                    b.append(y)
                    b.sort()
                    blocks[i] = b
                    
                    # Increment counts for subsets of new b containing y
                    for sub in itertools.combinations(b, t):
                        if y in sub:
                            counts[t_sub_index[sub]] += 1
        
        return best_blocks

    # 3. Greedy with lookahead: choose blocks that cover the most uncovered t-subsets
    def greedy_construct_lookahead(seed, time_limit):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered = set(range(n_t_subs))
        
        while uncovered and time.time() < time_limit:
            # Find the block that covers the most uncovered t-subsets
            # Sample a subset of blocks for efficiency
            sample_size = min(500, n_blocks)
            sampled_indices = random.sample(range(n_blocks), sample_size)
            
            best_block = None
            best_gain = -1
            
            for bi in sampled_indices:
                block = all_blocks[bi]
                gain = 0
                for idx in block_t_indices[bi]:
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(block)
            
            if best_block is None or best_gain <= 0:
                # Fallback: pick random uncovered
                idx = random.choice(list(uncovered))
                target_sub = all_t_subs[idx]
                block = list(target_sub)
                while len(block) < k:
                    best_p = -1
                    best_p_gain = -1
                    candidates = [p for p in range(v) if p not in block]
                    for p in candidates:
                        gain = 0
                        for rest in itertools.combinations(block, t-1):
                            sub = tuple(sorted(rest + (p,)))
                            sub_idx = t_sub_index.get(sub, -1)
                            if sub_idx in uncovered:
                                gain += 1
                        if gain > best_p_gain:
                            best_p_gain = gain
                            best_p = p
                    if best_p == -1:
                        break
                    block.append(best_p)
                block.sort()
                best_block = block
            
            blocks.append(best_block)
            
            # Update uncovered
            for sub in itertools.combinations(best_block, t):
                idx2 = t_sub_index[sub]
                counts[idx2] += 1
                if counts[idx2] == 1:
                    uncovered.discard(idx2)
                    
        return blocks

    # Strategy: Run multiple greedy variants and local search on the best
    best_initial = None
    best_initial_len = float('inf')
    
    # Time budget allocation
    greedy_time = min(25, (deadline - start_time) * 0.25)
    search_time = max(50, deadline - start_time - greedy_time - 5)
    
    # Try different greedy strategies
    strategies = [
        ("greedy", greedy_construct),
        ("lookahead", greedy_construct_lookahead),
    ]
    
    for name, func in strategies:
        for seed in range(3):
            if time.time() > deadline - search_time - 10:
                break
            t_limit = time.time() + greedy_time / 3
            blocks = func(seed, t_limit)
            if len(blocks) < best_initial_len:
                best_initial_len = len(blocks)
                best_initial = blocks
    
    if best_initial is None:
        best_initial = greedy_construct(0, time.time() + greedy_time)
    
    # Local search on the best initial
    final_blocks = local_search(best_initial, deadline)
    
    return final_blocks
