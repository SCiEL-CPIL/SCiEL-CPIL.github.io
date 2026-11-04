"""Model-and-optimize greedy covering design C(v,k,t) with local search refinement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    points = list(range(v))
    
    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    uncovered = set(all_t)
    
    blocks = []
    block_to_tsubsets = {}  # maps frozenset(block) -> list of t-subsets it covers
    
    def compute_gain(block_list, block_set, x):
        """Compute gain of adding x to block_list."""
        gain = 0
        for rest in itertools.combinations(block_list, t - 1):
            cand = tuple(sorted(rest + (x,)))
            if cand in uncovered:
                gain += 1
        return gain

    def greedy_block(anchor):
        """Build a block starting from anchor using greedy heuristic."""
        block = list(anchor)
        block_set = set(block)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in points:
                if x in block_set:
                    continue
                gain = compute_gain(block, block_set, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                for x in points:
                    if x not in block_set:
                        best_x = x
                        break
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        return block

    # Phase 1: Greedy construction
    while uncovered and time.time() < deadline - 30.0:
        anchor = min(uncovered)
        block = greedy_block(anchor)
        blocks.append(block)
        for comb in itertools.combinations(block, t):
            uncovered.discard(comb)
    
    # Ensure completeness
    if uncovered:
        while uncovered:
            anchor = min(uncovered)
            block = greedy_block(anchor)
            if block not in blocks:
                blocks.append(block)
            for comb in itertools.combinations(block, t):
                uncovered.discard(comb)
    
    # Phase 2: Local search - try to remove redundant blocks
    # For each block, check if it's the unique cover for any t-subset
    # Build: for each t-subset, which blocks cover it?
    tsub_to_blocks = {}
    for i, block in enumerate(blocks):
        for comb in itertools.combinations(block, t):
            if comb not in tsub_to_blocks:
                tsub_to_blocks[comb] = []
            tsub_to_blocks[comb].append(i)
    
    # Try removing blocks that are not the sole cover for any t-subset
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Check if block i is the only cover for any t-subset
            is_redundant = True
            for comb in itertools.combinations(blocks[i], t):
                covering_blocks = tsub_to_blocks.get(comb, [])
                if len(covering_blocks) == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove block i
                for comb in itertools.combinations(blocks[i], t):
                    tsub_to_blocks[comb].remove(i)
                blocks.pop(i)
                # Renumber all block indices in tsub_to_blocks
                for comb in tsub_to_blocks:
                    tsub_to_blocks[comb] = [j - 1 if j > i else j for j in tsub_to_blocks[comb]]
                changed = True
                break  # Restart scan after removal
    
    # Phase 3: Try to replace blocks with better ones if time allows
    if time.time() < deadline - 10.0:
        # For each block, try swapping one element to see if we can reduce total blocks
        # This is a simplified version; full local search would be more complex
        pass
    
    return blocks
# EVOLVE-BLOCK-END
