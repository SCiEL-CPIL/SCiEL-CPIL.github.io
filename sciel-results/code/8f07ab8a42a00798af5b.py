"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets contain it
    point_t_subsets = defaultdict(set)
    for i, s in enumerate(all_t_subsets):
        for p in s:
            point_t_subsets[p].add(i)
    
    # Greedy initialization
    uncovered = set(range(n_t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline - 5.0:
            break
        # Pick the first uncovered t-subset
        first = min(uncovered)
        block = list(all_t_subsets[first])
        
        # Greedily add points that cover the most uncovered t-subsets
        while len(block) < k:
            best_gain = -1
            best_point = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    s = tuple(sorted(rest + (x,)))
                    idx = t_subset_index.get(s)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        # Update uncovered
        for s in itertools.combinations(block, t):
            idx = t_subset_index[s]
            uncovered.discard(idx)
    
    # If we ran out of time during greedy, try to finish quickly with random blocks
    while uncovered and time.time() < deadline - 2.0:
        block = sorted(random.sample(range(v), k))
        blocks.append(block)
        for s in itertools.combinations(block, t):
            idx = t_subset_index[s]
            uncovered.discard(idx)
    
    # Local search: try to improve by swapping points in blocks
    # Represent coverage count for each t-subset
    cover_count = [0] * n_t
    for block in blocks:
        for s in itertools.combinations(block, t):
            idx = t_subset_index[s]
            cover_count[idx] += 1
    
    # Helper to compute how many newly covered t-subsets a block would add
    def block_new_coverage(block):
        count = 0
        for s in itertools.combinations(block, t):
            idx = t_subset_index[s]
            if cover_count[idx] == 0:
                count += 1
        return count
    
    # Local search phase
    rng = random.Random(42)
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        # Try removing a block if it doesn't uniquely cover any t-subset
        for i in range(len(blocks)):
            if time.time() > deadline - 3.0:
                break
            block = blocks[i]
            # Check if this block is the only cover for any t-subset
            essential = False
            for s in itertools.combinations(block, t):
                idx = t_subset_index[s]
                if cover_count[idx] == 1:
                    essential = True
                    break
            if not essential:
                # Remove this block
                for s in itertools.combinations(block, t):
                    idx = t_subset_index[s]
                    cover_count[idx] -= 1
                blocks.pop(i)
                improved = True
                break
        
        if time.time() > deadline - 3.0:
            break
        
        # Try adding a new block that covers the most uncovered t-subsets
        if uncovered:
            # Find a good new block
            best_block = None
            best_gain = 0
            # Try blocks centered around uncovered t-subsets
            sample_uncovered = list(uncovered)
            rng.shuffle(sample_uncovered)
            for idx in sample_uncovered[:50]:
                if time.time() > deadline - 3.0:
                    break
                base = all_t_subsets[idx]
                # Build a block around this t-subset
                block = list(base)
                while len(block) < k:
                    best_gain = -1
                    best_point = -1
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            s = tuple(sorted(rest + (x,)))
                            si = t_subset_index.get(s)
                            if si is not None and si in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    block.append(best_point)
                block.sort()
                # Compute actual gain
                gain = 0
                for s in itertools.combinations(block, t):
                    si = t_subset_index[s]
                    if cover_count[si] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block and best_gain > 0:
                blocks.append(best_block)
                for s in itertools.combinations(best_block, t):
                    idx = t_subset_index[s]
                    cover_count[idx] += 1
                    if cover_count[idx] == 1:
                        uncovered.discard(idx)
                improved = True
        
        if time.time() > deadline - 3.0:
            break
    
    # Final cleanup: remove any redundant blocks
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                break
            block = blocks[i]
            essential = False
            for s in itertools.combinations(block, t):
                idx = t_subset_index[s]
                if cover_count[idx] == 1:
                    essential = True
                    break
            if not essential:
                for s in itertools.combinations(block, t):
                    idx = t_subset_index[s]
                    cover_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break
    
    # Deduplicate blocks
    seen = set()
    unique_blocks = []
    for block in blocks:
        key = tuple(block)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(block)
    
    return unique_blocks
# EVOLVE-BLOCK-END
