"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _greedy_cover(v, k, t, rng, deadline):
    """Greedy covering with randomized block selection."""
    all_tsubs = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None  # timeout
        
        # Pick a random uncovered t-subset to start a new block
        start_tsub = rng.choice(list(uncovered))
        block = list(start_tsub)
        
        while len(block) < k:
            # Find point that covers most uncovered t-subsets
            best_gain = -1
            best_point = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    if tuple(sorted(rest + (x,))) in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # No point improves; pick random
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
            
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    
    return blocks


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    from collections import defaultdict
    cover_count = defaultdict(int)
    block_tsubs = []
    
    for block in blocks:
        tsubs = list(itertools.combinations(block, t))
        block_tsubs.append(tsubs)
        for ts in tsubs:
            cover_count[ts] += 1
    
    # Remove blocks where all their t-subsets are covered by other blocks
    kept_blocks = []
    kept_indices = []
    for i, block in enumerate(blocks):
        all_redundant = True
        for ts in block_tsubs[i]:
            if cover_count[ts] == 1:
                all_redundant = False
                break
        if not all_redundant:
            kept_blocks.append(block)
            kept_indices.append(i)
    
    # Update cover counts for remaining blocks
    if len(kept_blocks) < len(blocks):
        cover_count2 = defaultdict(int)
        for block in kept_blocks:
            for ts in itertools.combinations(block, t):
                cover_count2[ts] += 1
        
        # Second pass: remove blocks that are now redundant
        final_blocks = []
        for i, block in enumerate(kept_blocks):
            all_redundant = True
            for ts in itertools.combinations(block, t):
                if cover_count2[ts] == 1:
                    all_redundant = False
                    break
            if not all_redundant:
                final_blocks.append(block)
        
        return final_blocks
    
    return blocks


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    all_tsubs = set(itertools.combinations(range(v), t))
    return all_tsubs.issubset(covered)


def construct(v, k, t):
    deadline = time.time() + 140  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        blocks = _greedy_cover(v, k, t, rng, deadline)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, t)
        
        # Verify validity
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        # Early exit if we hit a good solution
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy (original approach)
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
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
