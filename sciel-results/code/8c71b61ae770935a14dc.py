"""Improved covering design C(24,7,3) via greedy + local search (block swaps)."""
import itertools
import time
import random

def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s (evaluator kills at 150s)
    
    # Precompute all t-subsets for quick membership checks
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    # --- Phase 1: Greedy construction ---
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered and time.time() < deadline - 20:
        # Pick the first uncovered triple to anchor a new block
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        # Greedily add points that cover the most uncovered triples
        while len(block) < k and time.time() < deadline - 20:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered triples would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    triple = tuple(sorted(rest + (x,)))
                    if triple in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove covered triples
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
    
    if not uncovered:
        return blocks
    
    # --- Phase 2: Local search with block swaps ---
    # Build a coverage count: for each triple, how many blocks cover it
    triple_count = {}
    for b in blocks:
        for triple in itertools.combinations(b, t):
            triple_count[triple] = triple_count.get(triple, 0) + 1
    
    # Map from point to list of block indices containing it
    point_to_blocks = {p: [] for p in range(v)}
    for i, b in enumerate(blocks):
        for p in b:
            point_to_blocks[p].append(i)
    
    # Try to remove redundant blocks
    def try_remove_redundant():
        removed = []
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                redundant = True
                for triple in itertools.combinations(blocks[i], t):
                    if triple_count.get(triple, 0) <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for triple in itertools.combinations(blocks[i], t):
                        triple_count[triple] -= 1
                    for p in blocks[i]:
                        point_to_blocks[p].remove(i)
                    # Renumber
                    for j in range(len(blocks)):
                        for p in blocks[j]:
                            if j > i and i in point_to_blocks[p]:
                                point_to_blocks[p][point_to_blocks[p].index(i)] = j - 1
                    removed.append(blocks.pop(i))
                    changed = True
                    break
        return removed
    
    # Try block swaps: replace one block with a better one
    def try_swaps():
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                # Find the triples that ONLY block i covers
                only_triples = []
                for triple in itertools.combinations(blocks[i], t):
                    if triple_count.get(triple, 0) == 1:
                        only_triples.append(triple)
                
                if not only_triples:
                    # Block is redundant, try to remove it
                    for triple in itertools.combinations(blocks[i], t):
                        triple_count[triple] -= 1
                    for p in blocks[i]:
                        point_to_blocks[p].remove(i)
                    for j in range(len(blocks)):
                        for p in blocks[j]:
                            if j > i and i in point_to_blocks[p]:
                                point_to_blocks[p][point_to_blocks[p].index(i)] = j - 1
                    blocks.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement block that covers all only_triples
                # and has minimal overlap with other blocks' unique triples
                best_block = None
                best_score = float('inf')
                
                # Candidate: build a block around the intersection of only_triples
                # The new block must contain all points that appear in only_triples
                required_points = set()
                for triple in only_triples:
                    required_points.update(triple)
                
                if len(required_points) > k:
                    continue
                
                # Try all ways to fill the remaining k - len(required_points) spots
                remaining_spots = k - len(required_points)
                available_points = [p for p in range(v) if p not in required_points]
                
                if remaining_spots == 0:
                    candidate = tuple(sorted(required_points))
                elif remaining_spots == 1:
                    for p in available_points:
                        candidate = tuple(sorted(required_points | {p}))
                        # Score: how many NEW unique triples does this create?
                        new_unique = 0
                        for triple in itertools.combinations(candidate, t):
                            if triple_count.get(triple, 0) == 0:
                                new_unique += 1
                        score = new_unique
                        if score < best_score:
                            best_score = score
                            best_block = candidate
                elif remaining_spots <= 4:
                    # Limited search for small remaining spots
                    for combo in itertools.combinations(available_points, remaining_spots):
                        candidate = tuple(sorted(required_points | set(combo)))
                        new_unique = 0
                        for triple in itertools.combinations(candidate, t):
                            if triple_count.get(triple, 0) == 0:
                                new_unique += 1
                        score = new_unique
                        if score < best_score:
                            best_score = score
                            best_block = candidate
                        if time.time() > deadline - 5:
                            break
                else:
                    continue
                
                if best_block is not None and best_block != tuple(blocks[i]):
                    # Check if this swap is beneficial
                    # Count how many triples the old block uniquely covers vs new
                    old_unique = 0
                    for triple in itertools.combinations(blocks[i], t):
                        if triple_count.get(triple, 0) == 1:
                            old_unique += 1
                    
                    new_unique = 0
                    for triple in itertools.combinations(best_block, t):
                        if triple_count.get(triple, 0) == 0:
                            new_unique += 1
                    
                    # Swap if new block doesn't create more unique triples
                    if new_unique <= old_unique:
                        # Perform the swap
                        old_block = blocks[i]
                        for triple in itertools.combinations(old_block, t):
                            triple_count[triple] -= 1
                        for p in old_block:
                            point_to_blocks[p].remove(i)
                        
                        blocks[i] = list(best_block)
                        for triple in itertools.combinations(best_block, t):
                            triple_count[triple] = triple_count.get(triple, 0) + 1
                        for p in best_block:
                            point_to_blocks[p].append(i)
                        
                        improved = True
                        break
        
        return improved
    
    # Run local search
    try_remove_redundant()
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        if not try_swaps():
            break
        try_remove_redundant()
    
    # Final cleanup
    try_remove_redundant()
    
    return blocks
