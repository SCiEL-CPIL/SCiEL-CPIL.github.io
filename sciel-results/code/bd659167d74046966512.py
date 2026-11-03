import itertools
import time
import random
from collections import defaultdict
import sys

def _greedy_cover(v, k, t, rng):
    """
    Constructs a covering using a weighted greedy heuristic.
    """
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    
    # Pre-compute triples for each potential block? Too memory heavy for v=24, k=7.
    # C(24,7) = 346,104. C(7,3) = 35. Total triples 2024.
    # We can precompute the set of triples for each block if we generate blocks on the fly?
    # No, we generate blocks one by one.
    
    # To speed up, we can precompute for each triple, the list of points that can form a block with it?
    # Or just use the standard greedy: pick uncovered triple, find best block containing it.
    
    while uncovered:
        # Pick a random uncovered triple to start the block
        start_triple = rng.choice(list(uncovered))
        block_set = set(start_triple)
        block = list(start_triple)
        
        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            best_gain = -1
            best_points = []
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Calculate gain: how many new uncovered triples does adding x cover?
                # New triples are those formed by x and two existing points in block
                gain = 0
                # Optimized gain calculation
                # We can precompute the triples for the current partial block?
                # For k=7, this is small.
                
                # To be faster, we can just count
                for a, b in itertools.combinations(block, 2):
                    triple = tuple(sorted((a, b, x)))
                    if triple in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain:
                    best_points.append(x)
            
            if not best_points:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    x = rng.choice(candidates)
                else:
                    break
            else:
                x = rng.choice(best_points)
            
            block.append(x)
            block_set.add(x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return [list(b) for b in blocks]

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
    """
    if not blocks:
        return blocks
    
    # Map each triple to the count of blocks covering it
    triple_cover_count = defaultdict(int)
    block_triples = []
    
    for block in blocks:
        triples = list(itertools.combinations(block, t))
        block_triples.append(triples)
        for triple in triples:
            triple_cover_count[tuple(triple)] += 1
    
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_triples = []
        
        for i, block in enumerate(blocks):
            # Check if this block is essential
            is_essential = False
            for triple in block_triples[i]:
                if triple_cover_count[tuple(triple)] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
                new_block_triples.append(block_triples[i])
            else:
                # Remove this block
                changed = True
                for triple in block_triples[i]:
                    triple_cover_count[tuple(triple)] -= 1
        
        blocks = new_blocks
        block_triples = new_block_triples
    
    return blocks

def _local_search(blocks, v, k, t, rng, deadline):
    """
    Local search to reduce block count.
    Tries:
    1. Remove block if redundant.
    2. Swap block with a better one.
    3. Replace 2 blocks with 1 block if possible.
    """
    blocks = [tuple(b) for b in blocks]
    # Precompute triples for each block
    block_triples = [list(itertools.combinations(b, t)) for b in blocks]
    
    # Cover counts
    triple_cover_count = defaultdict(int)
    for triples in block_triples:
        for triple in triples:
            triple_cover_count[tuple(triple)] += 1
            
    # Map triple to list of block indices covering it (for quick lookup)
    # This is expensive to maintain, so we'll just use counts and re-check when needed.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try removing single blocks
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            
            # Check if block i is redundant
            redundant = True
            for triple in block_triples[i]:
                if triple_cover_count[tuple(triple)] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for triple in block_triples[i]:
                    triple_cover_count[tuple(triple)] -= 1
                blocks.pop(i)
                block_triples.pop(i)
                improved = True
                break # Restart loop as indices changed
        
        if improved:
            continue
            
        # 2. Try swapping blocks (1-for-1) to improve structure
        # This is harder to implement efficiently without a strong heuristic for "better".
        # We'll skip this for now and focus on 2-for-1 which directly reduces count.
        
        # 3. Try replacing 2 blocks with 1 block (2-for-1)
        # Pick two random blocks
        if len(blocks) >= 2:
            i, j = rng.sample(range(len(blocks)), 2)
            
            # Temporarily remove them
            b_i = blocks[i]
            b_j = blocks[j]
            tr_i = block_triples[i]
            tr_j = block_triples[j]
            
            # Calculate uncovered triples if we remove i and j
            # A triple is uncovered if its count drops to 0
            # We need to find all triples that are covered ONLY by i or j (or both, but count becomes 0)
            
            # Let's collect the set of triples that become uncovered
            uncovered_by_removal = set()
            
            # We need to know which triples are covered by i or j and not by others
            # This requires knowing the count.
            
            # Efficient way:
            # For each triple in tr_i, if count == 1, it's uncovered. If count == 2, check if j also covers it.
            # Actually, simpler:
            # Create a temporary count map? No, too slow.
            
            # Let's just iterate through tr_i and tr_j.
            # For each triple in tr_i:
            #   if triple_cover_count[triple] == 1:
            #       it's uncovered.
            #   elif triple_cover_count[triple] == 2:
            #       check if j covers it. If yes, it's uncovered. If no, it's covered by another.
            
            # To check if j covers a triple, we can use a set for tr_j.
            set_tr_j = set(tr_j)
            set_tr_i = set(tr_i)
            
            potential_uncovered = set()
            
            for triple in tr_i:
                count = triple_cover_count[tuple(triple)]
                if count == 1:
                    potential_uncovered.add(tuple(triple))
                elif count == 2:
                    if tuple(triple) in set_tr_j:
                        potential_uncovered.add(tuple(triple))
                        
            for triple in tr_j:
                if tuple(triple) in set_tr_i:
                    continue # Already processed
                count = triple_cover_count[tuple(triple)]
                if count == 1:
                    potential_uncovered.add(tuple(triple))
                # If count == 2, it's covered by i (already checked) or another. 
                # Wait, if count is 2 and it's not in tr_i, then it's covered by j and someone else.
                # So removing i and j leaves it covered by someone else. So it's NOT uncovered.
                
            if not potential_uncovered:
                # We can remove both
                # Remove j first (higher index) to keep indices valid
                for triple in tr_j:
                    triple_cover_count[tuple(triple)] -= 1
                for triple in tr_i:
                    triple_cover_count[tuple(triple)] -= 1
                
                blocks.pop(j)
                block_triples.pop(j)
                blocks.pop(i)
                block_triples.pop(i)
                improved = True
                break
            
            else:
                # Can we find a single block that covers all triples in potential_uncovered?
                # If yes, replace i and j with this new block.
                
                # This is a subproblem: Find a k-subset that contains all triples in potential_uncovered.
                # potential_uncovered is a set of triples.
                # The union of all points in these triples must be <= k.
                # If union > k, impossible.
                
                points_in_uncovered = set()
                for triple in potential_uncovered:
                    for p in triple:
                        points_in_uncovered.add(p)
                        
                if len(points_in_uncovered) <= k:
                    # We can form a block from points_in_uncovered
                    # We need to fill it up to size k with arbitrary points
                    # The new block must cover all triples in potential_uncovered.
                    # Since potential_uncovered triples are subsets of points_in_uncovered,
                    # any superset of points_in_uncovered will cover them.
                    
                    new_block_points = list(points_in_uncovered)
                    # Add random points to reach size k
                    available_points = [p for p in range(v) if p not in points_in_uncovered]
                    rng.shuffle(available_points)
                    while len(new_block_points) < k:
                        if available_points:
                            new_block_points.append(available_points.pop())
                        else:
                            break
                            
                    new_block = tuple(sorted(new_block_points))
                    
                    # Check if this new block is already in the covering?
                    # If it is, we don't add it, we just remove i and j.
                    # But wait, if the new block is already there, does it cover potential_uncovered?
                    # Yes, by construction.
                    
                    # However, we must ensure that removing i and j doesn't uncover anything else.
                    # We already checked potential_uncovered.
                    # But what if the new block is one of the other blocks?
                    # It doesn't matter, the coverage is maintained.
                    
                    # Update counts
                    # Remove i and j
                    for triple in tr_j:
                        triple_cover_count[tuple(triple)] -= 1
                    for triple in tr_i:
                        triple_cover_count[tuple(triple)] -= 1
                    
                    # Add new block
                    new_triples = list(itertools.combinations(new_block, t))
                    for triple in new_triples:
                        triple_cover_count[tuple(triple)] += 1
                        
                    # Replace i and j with new_block
                    # Remove j
                    blocks.pop(j)
                    block_triples.pop(j)
                    # Remove i
                    blocks.pop(i)
                    block_triples.pop(i)
                    # Add new
                    blocks.append(new_block)
                    block_triples.append(new_triples)
                    
                    improved = True
                    break

    return [list(b) for b in blocks]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_blocks = None
    best_size = float('inf')
    
    # Initial deterministic run
    rng = random.Random(42)
    blocks = _greedy_cover(v, k, t, rng)
    blocks = _reduce_covering(blocks, v, k, t)
    blocks = _local_search(blocks, v, k, t, rng, deadline)
    
    best_blocks = blocks
    best_size = len(blocks)
    
    # Try multiple randomized runs
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _greedy_cover(v, k, t, rng)
        
        if time.time() > deadline:
            continue
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        if time.time() > deadline:
            continue
            
        blocks = _local_search(blocks, v, k, t, rng, deadline)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # If we hit the theoretical lower bound or a known good value, we could stop
            if best_size <= 69:
                break
                
    return best_blocks
