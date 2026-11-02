import itertools
import time
import random
from collections import defaultdict

def _greedy_cover_fast(v, k, t, rng, deadline):
    """
    Constructs a covering using a greedy heuristic with randomization.
    Optimized for speed by pre-computing uncovered counts per pair.
    """
    # Map triple to count (1 if uncovered, 0 if covered)
    # We use a set for membership testing for simplicity first, 
    # but for gain calculation we need to know which triples are uncovered.
    
    # A more efficient structure:
    # For each pair (a, b), keep a set of points x such that (a, b, x) is uncovered.
    # This allows O(1) lookup for gain calculation.
    
    # Initialize pair_uncovered: dict[tuple[int, int], set[int]]
    pair_uncovered = defaultdict(set)
    for a in range(v):
        for b in range(a + 1, v):
            # For each x, if (a, b, x) is a valid triple, it's initially uncovered
            # We don't need to store x explicitly in the set initially if we assume all are uncovered.
            # But we need to remove them as they get covered.
            # Let's initialize with all possible x not in {a, b}
            # Actually, we can just use a set of triples for simplicity and rely on the fact that
            # v=24 is small. C(24,3) = 2024.
            pass
            
    # Let's stick to the set of uncovered triples but optimize the gain loop.
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered triple to start the block
        # Using tuple(uncovered) for choice is O(N) which is fine for 2024 items
        start_triple = rng.choice(tuple(uncovered))
        block = list(start_triple)
        
        # Greedily add points to the block
        while len(block) < k:
            if time.time() > deadline:
                break
                
            # Calculate gain for each candidate point
            # Gain of x = number of pairs (a, b) in block such that (a, b, x) is in uncovered
            
            best_gain = -1
            best_points = []
            block_set = set(block)
            
            # Pre-compute pairs in current block
            block_pairs = list(itertools.combinations(block, 2))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for a, b in block_pairs:
                    # Create sorted triple
                    # Since a < b (from combinations), we just need to insert x
                    # Or just sort the tuple
                    # tuple(sorted((a, b, x))) is fast enough for small k
                    triple = tuple(sorted((a, b, x)))
                    if triple in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain:
                    best_points.append(x)
            
            # If no gain, pick random point not in block
            if not best_points:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    x = rng.choice(candidates)
                else:
                    break
            else:
                # Randomly pick from best points to add diversity
                x = rng.choice(best_points)
            
            block.append(x)
        
        if len(block) < k:
            # If we couldn't fill the block, skip it or pad? 
            # The problem requires blocks of size k.
            # If we break early, we should probably just stop or handle carefully.
            # Given v=24, k=7, we should always be able to fill.
            pass
            
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered triples
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return [list(b) for b in blocks]

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
    Returns the reduced list of blocks.
    """
    if not blocks:
        return blocks
    
    # Map each triple to the list of blocks that cover it
    triple_to_blocks = defaultdict(list)
    
    # Build the map
    for i, block in enumerate(blocks):
        for triple in itertools.combinations(block, t):
            triple_to_blocks[tuple(triple)].append(i)
    
    # Active flags
    active = [True] * len(blocks)
    
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if not active[i]:
                continue
            # Check if block i is essential
            is_essential = False
            block = blocks[i]
            for triple in itertools.combinations(block, t):
                # Check if any other active block covers this triple
                # triple_to_blocks[t] contains all block indices originally.
                # We need to check if any *other* active block is in the list.
                # To speed this up, we can just check the count of active blocks covering it.
                # But maintaining counts dynamically is complex.
                # For v=24, k=7, num blocks ~70-80.
                # Triple count per block is C(7,3)=35.
                # Total checks: 80 * 35 * (check list length ~2-3) is fast.
                
                covering_blocks = triple_to_blocks[tuple(triple)]
                for j in covering_blocks:
                    if j != i and active[j]:
                        is_essential = False
                        break
                if is_essential:
                    break
            
            if not is_essential:
                # Remove this block
                active[i] = False
                changed = True
                
    return [blocks[i] for i in range(len(blocks)) if active[i]]

def _local_search_remove(blocks, v, k, t, rng, deadline):
    """
    Attempts to further reduce the number of blocks by finding a block that can be removed
    and its uncovered triples covered by existing blocks (which is what _reduce_covering does).
    This function is currently just a wrapper or placeholder for more complex moves like
    swapping blocks. For now, we rely on _reduce_covering.
    """
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_blocks = None
    best_size = float('inf')
    
    # Initial deterministic run
    rng = random.Random(42)
    blocks = _greedy_cover_fast(v, k, t, rng, deadline)
    blocks = _reduce_covering(blocks, v, k, t)
    best_blocks = blocks
    best_size = len(blocks)
    
    # Try multiple randomized runs
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _greedy_cover_fast(v, k, t, rng, deadline)
        
        if time.time() > deadline:
            continue
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            if best_size <= 69:
                break
                
    return best_blocks
