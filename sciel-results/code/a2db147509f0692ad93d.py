"""Covering design C(24,7,3) via randomized greedy with redundancy removal and local search."""
import itertools
import random
import time

def _all_triples(v, t):
    return list(itertools.combinations(range(v), t))

def _greedy_cover(v, k, t, triples, rng, deadline=None):
    """Randomized greedy: pick uncovered triple, build block by adding point with max gain."""
    uncovered = set(triples)
    blocks = []
    while uncovered:
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random uncovered triple to seed the block
        seed_triple = rng.choice(list(uncovered))
        block = list(seed_triple)
        while len(block) < k:
            best_gain = -1
            best_x = None
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered triples would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(rest + (x,)))
                    if new_triple in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        if len(block) < k:
            break
        block.sort()
        blocks.append(block)
        # Remove covered triples
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
    return blocks

def _remove_redundant_blocks(blocks, v, k, t, triples, deadline=None):
    """Remove blocks that are not needed for coverage."""
    # Count how many blocks cover each triple
    triple_to_blocks = {triple: [] for triple in triples}
    for i, block in enumerate(blocks):
        for triple in itertools.combinations(block, t):
            triple_to_blocks[triple].append(i)
    
    # Iteratively remove redundant blocks
    changed = True
    while changed:
        if deadline is not None and time.time() > deadline:
            break
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            # Check if block i is redundant: every triple in it is covered by another block
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                if len(triple_to_blocks[triple]) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for triple in itertools.combinations(blocks[i], t):
                    triple_to_blocks[triple].remove(i)
                blocks.pop(i)
                # Fix indices in triple_to_blocks
                for triple in triple_to_blocks:
                    triple_to_blocks[triple] = [j if j < i else j - 1 for j in triple_to_blocks[triple]]
                changed = True
    return blocks

def _local_search(blocks, v, k, t, triples, rng, deadline=None):
    """Try to replace blocks with better ones to reduce total count or improve coverage."""
    current_blocks = list(blocks)
    current_len = len(current_blocks)
    best_blocks = current_blocks
    best_len = current_len
    
    # Try swapping individual blocks with better alternatives
    for i in range(len(current_blocks)):
        if deadline is not None and time.time() > deadline:
            break
        # Try removing block i and see if we can cover the lost triples with fewer additions
        block_i = current_blocks[i]
        lost_triples = set(itertools.combinations(block_i, t))
        
        # Check which lost triples are still covered by other blocks
        other_blocks = current_blocks[:i] + current_blocks[i+1:]
        still_covered = set()
        for block in other_blocks:
            for triple in itertools.combinations(block, t):
                still_covered.add(triple)
        
        truly_lost = lost_triples - still_covered
        if not truly_lost:
            # Block i is redundant
            new_blocks = current_blocks[:i] + current_blocks[i+1:]
            if len(new_blocks) < best_len:
                best_len = len(new_blocks)
                best_blocks = new_blocks
            current_blocks = new_blocks
            continue
        
        # Try to find a single block that covers all truly_lost triples
        # This is expensive, so we'll skip for now and just try random swaps
        pass
    
    return best_blocks

def construct(v, k, t):
    """Construct a covering design C(v, k, t)."""
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin before 150s kill
    
    triples = _all_triples(v, t)
    num_triples = len(triples)
    
    best_blocks = None
    best_len = float('inf')
    
    # Multiple randomized greedy restarts
    num_restarts = 50  # Will be limited by time
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Leave 10s for final processing
            break
        
        # Randomized greedy
        blocks = _greedy_cover(v, k, t, triples, rng, deadline=deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        # Remove redundant blocks
        if time.time() > deadline - 5:
            break
        blocks = _remove_redundant_blocks(blocks, v, k, t, triples, deadline=deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        # Local search
        if time.time() > deadline - 2:
            break
        blocks = _local_search(blocks, v, k, t, triples, rng, deadline=deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Ensure we have a valid covering
    if best_blocks is None:
        # Fallback: deterministic greedy
        best_blocks = _greedy_cover(v, k, t, triples, rng, deadline=deadline)
        best_blocks = _remove_redundant_blocks(best_blocks, v, k, t, triples, deadline=deadline)
    
    # Verify coverage
    covered = set()
    for block in best_blocks:
        for triple in itertools.combinations(block, t):
            covered.add(triple)
    
    # If not fully covered, add more blocks
    if len(covered) < num_triples:
        uncovered = set(triples) - covered
        while uncovered and time.time() < deadline:
            seed_triple = list(uncovered)[0]
            block = list(seed_triple)
            while len(block) < k:
                best_gain = -1
                best_x = None
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t-1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            if len(block) < k:
                break
            block.sort()
            best_blocks.append(block)
            for triple in itertools.combinations(block, t):
                uncovered.discard(triple)
    
    return best_blocks
