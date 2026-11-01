import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    if v == 24 and k == 7 and t == 3:
        return construct_24_7_3(deadline)
    else:
        return construct_generic(v, k, t, deadline)

def construct_24_7_3(deadline):
    v, k, t = 24, 7, 3
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute blocks and their masks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Masks for triples covered by each block
    # Using Python ints for bitmasks
    block_masks = [0] * num_blocks
    block_triple_lists = [[] for _ in range(num_blocks)]
    
    # Precomputation
    for i, block in enumerate(all_blocks):
        mask = 0
        t_list = []
        # Optimization: precompute triples for this block?
        # itertools.combinations(block, t)
        for triple in itertools.combinations(block, t):
            idx = tsub_to_idx[triple]
            mask |= (1 << idx)
            t_list.append(idx)
        block_masks[i] = mask
        block_triple_lists[i] = t_list
        
    if time.time() > deadline - 30:
        # If precomp took too long (unlikely), fallback to simple greedy
        pass

    def greedy_construct(randomize=False):
        uncovered_mask = (1 << num_tsubs) - 1
        blocks = []
        
        while uncovered_mask:
            best_idx = -1
            best_cov = -1
            
            # Scan all blocks
            for i in range(num_blocks):
                m = block_masks[i]
                # Count overlapping bits
                # In Python 3.10+, int.bit_count() is available
                cov = (m & uncovered_mask).bit_count()
                
                # Randomize tie-breaking if needed
                if randomize:
                    cov += random.random() * 0.1 # Small noise
                
                if cov > best_cov:
                    best_cov = cov
                    best_idx = i
            
            if best_idx == -1:
                break
                
            blocks.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
            if time.time() > deadline - 10:
                break
                
        return blocks

    def refine(blocks):
        # blocks is list of indices
        if not blocks:
            return []
            
        # Count coverage
        counts = [0] * num_tsubs
        for b_idx in blocks:
            for t_idx in block_triple_lists[b_idx]:
                counts[t_idx] += 1
                
        changed = True
        while changed:
            changed = False
            # Iterate backwards to remove
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                redundant = True
                for t_idx in block_triple_lists[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for t_idx in block_triple_lists[b_idx]:
                        counts[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    
                    if time.time() > deadline - 5:
                        return blocks
        return blocks

    # Run 1: Deterministic
    blocks_idx = greedy_construct(randomize=False)
    blocks_idx = refine(blocks_idx)
    
    best_blocks = [list(all_blocks[i]) for i in blocks_idx]
    
    # Run 2: Randomized if time permits
    if time.time() < deadline - 15:
        for _ in range(3): # A few restarts
            if time.time() > deadline - 5:
                break
            blocks_idx_r = greedy_construct(randomize=True)
            blocks_idx_r = refine(blocks_idx_r)
            if len(blocks_idx_r) < len(blocks_idx):
                blocks_idx = blocks_idx_r
                best_blocks = [list(all_blocks[i]) for i in blocks_idx]
                
    return best_blocks

def construct_generic(v, k, t, deadline):
    # Fallback for other parameters
    # Fixed version of the original logic
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    def greedy_construct():
        uncovered = set(range(num_tsubs))
        blocks = []
        
        while uncovered:
            best_block = None
            best_count = -1
            candidates = set()
            for idx in list(uncovered)[:50]:
                triple = all_tsubs[idx]
                remaining = [x for x in range(v) if x not in triple]
                for combo in itertools.combinations(remaining, k - t):
                    block = tuple(sorted(triple + combo))
                    candidates.add(block)
            
            if not candidates:
                break
                
            for block in candidates:
                count = 0
                for triple in itertools.combinations(block, t):
                    idx = tsub_to_idx[triple]
                    if idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            for triple in itertools.combinations(best_block, t):
                idx = tsub_to_idx[triple]
                uncovered.discard(idx)
            
            if time.time() > deadline - 5:
                break
        
        return blocks

    blocks = greedy_construct()
    
    # Simple refinement: remove redundant blocks
    # Using sets for speed in generic case (might be slow if v is large, but fallback)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            block_set = set(blocks[i])
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                triple_set = set(triple)
                covered = False
                for j, other_block in enumerate(blocks):
                    if j == i:
                        continue
                    if triple_set.issubset(set(other_block)):
                        covered = True
                        break
                if not covered:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                
    return blocks
