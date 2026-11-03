"""Covering design C(19,8,5) - greedy with randomized restarts and local search."""
import itertools
import random
import time

# EVOLVE-BLOCK-START

def _count_uncovered_in_block(block, uncovered_set):
    """Count how many t-subsets of block are still uncovered."""
    cnt = 0
    for sub in itertools.combinations(block, 5):
        if sub in uncovered_set:
            cnt += 1
    return cnt


def _greedy_build(v, k, t, uncovered_set, rng, max_time=None, start_time=None):
    """
    Greedily build a covering. Start with a random-ish uncovered t-subset,
    then grow each block by picking the point that covers the most uncovered t-subsets.
    """
    if max_time is None:
        max_time = 110
    if start_time is None:
        start_time = time.time()
    
    blocks = []
    uncovered = set(uncovered_set)
    
    while uncovered and (time.time() - start_time) < max_time:
        # Pick a random uncovered t-subset to start the block
        start_sub = tuple(rng.choice(list(uncovered)))
        block = list(start_sub)
        block_set = set(block)
        
        while len(block) < k and (time.time() - start_time) < max_time:
            best_x = None
            best_gain = -1
            # For each candidate point not in block, compute gain
            for x in range(v):
                if x in block_set:
                    continue
                # Gain: number of (t-1)-subsets of block that with x form an uncovered t-subset
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        block_tuple = tuple(block)
        
        # Remove covered t-subsets
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
        
        blocks.append(block)
    
    return blocks, uncovered


def _local_search(blocks, v, k, t, max_time=None, start_time=None, rng=None):
    """
    Try to remove redundant blocks and replace blocks with better ones.
    """
    if max_time is None:
        max_time = 110
    if start_time is None:
        start_time = time.time()
    if rng is None:
        rng = random.Random()
    
    # Build coverage count for each t-subset
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Count how many blocks cover each t-subset
    cover_count = [0] * n_t
    block_subs = []
    for block in blocks:
        subs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_subs.append(subs)
        for idx in subs:
            cover_count[idx] += 1
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        
        # Try removing blocks that don't uniquely cover any t-subset
        for i in range(len(blocks) - 1, -1, -1):
            if (time.time() - start_time) > max_time:
                break
            unique = False
            for idx in block_subs[i]:
                if cover_count[idx] == 1:
                    unique = True
                    break
            if not unique:
                # Remove block i
                for idx in block_subs[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_subs.pop(i)
                improved = True
        
        # Try replacing a block with a better one
        if (time.time() - start_time) > max_time:
            break
        
        for i in range(len(blocks)):
            if (time.time() - start_time) > max_time:
                break
            
            # Find t-subsets uniquely covered by block i
            unique_subs = [idx for idx in block_subs[i] if cover_count[idx] == 1]
            if not unique_subs:
                continue
            
            # Try to find a replacement block that covers all unique_subs
            # and potentially more
            # Use greedy: start from one of the unique subsets, grow
            best_new_block = None
            best_new_gain = 0
            
            # Sample a few starting points from unique_subs
            sample_size = min(5, len(unique_subs))
            start_indices = rng.sample(unique_subs, sample_size) if len(unique_subs) > sample_size else unique_subs
            
            for start_idx in start_indices:
                start_sub = t_subsets[start_idx]
                block = list(start_sub)
                block_set = set(block)
                
                while len(block) < k:
                    best_x = None
                    best_gain = -1
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        # Count how many unique_subs this x would help cover
                        for rest in itertools.combinations(block, t - 1):
                            sub = tuple(sorted(rest + (x,)))
                            if sub in t_sub_to_idx:
                                idx = t_sub_to_idx[sub]
                                if idx in unique_subs or cover_count[idx] >= 1:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                
                block.sort()
                # Check if this block covers all unique_subs
                block_sub_set = set(itertools.combinations(block, t))
                all_covered = all(tuple(t_subsets[si]) in block_sub_set for si in unique_subs)
                if all_covered:
                    # Count additional coverage
                    new_block_subs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
                    new_cover_gain = sum(1 for idx in new_block_subs if cover_count[idx] == 0)
                    if new_cover_gain > best_new_gain:
                        best_new_gain = new_cover_gain
                        best_new_block = block
                        best_new_block_subs = new_block_subs
            
            if best_new_block is not None and best_new_gain > 0:
                # Replace block i
                for idx in block_subs[i]:
                    cover_count[idx] -= 1
                for idx in best_new_block_subs:
                    cover_count[idx] += 1
                blocks[i] = best_new_block
                block_subs[i] = best_new_block_subs
                improved = True
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # All t-subsets
    all_t_subs = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized greedy restarts
    n_restarts = 0
    while time.time() < deadline - 20:  # Leave 20s for local search
        rng = random.Random(n_restarts * 12345 + 678)
        blocks, remaining = _greedy_build(v, k, t, all_t_subs, rng, 
                                           max_time=min(110, deadline - time.time()),
                                           start_time=start_time)
        n_restarts += 1
        
        if remaining:
            # Incomplete - skip
            continue
        
        # Apply local search
        if time.time() < deadline - 5:
            blocks = _local_search(blocks, v, k, t, 
                                   max_time=min(105, deadline - time.time() - 5),
                                   start_time=start_time,
                                   rng=random.Random(n_restarts * 999))
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
    
    if best_blocks is None:
        # Fallback: simple greedy without randomization
        rng = random.Random(42)
        blocks, _ = _greedy_build(v, k, t, all_t_subs, rng,
                                   max_time=110, start_time=start_time)
        best_blocks = blocks
    
    return best_blocks

# EVOLVE-BLOCK-END
