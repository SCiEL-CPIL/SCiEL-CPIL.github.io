import itertools
import time
import random
from collections import defaultdict

def _precompute_triples(v, t):
    """Pre-compute all t-subsets as sorted tuples and map to indices."""
    triples = []
    triple_to_idx = {}
    for combo in itertools.combinations(range(v), t):
        idx = len(triples)
        triples.append(combo)
        triple_to_idx[combo] = idx
    return triples, triple_to_idx

def _greedy_fast(v, k, t, seed, triples, triple_to_idx, all_points):
    """Fast greedy using bitmasks for uncovered tracking."""
    rng = random.Random(seed)
    n_triples = len(triples)
    
    # Use a list of booleans for uncovered tracking (faster than set for small n)
    uncovered = [True] * n_triples
    uncovered_count = n_triples
    blocks = []
    
    # Pre-compute for each point, which triples it participates in
    # For t=3: point x is in triples (x,a,b) where a,b > x or a,b < x
    # Better: precompute triple indices for each pair of points
    # For adding point x to block B, gain = |{triples containing x and 2 points in B}|
    
    # Precompute: for each point x, list of (triple_idx, other_two_points)
    # Actually, for gain calculation when adding x to block with points p1..pm:
    # gain = number of uncovered triples (x, pi, pj) for all pairs pi,pj in block
    
    # Precompute triple lookup: triple_to_idx[(a,b,c)] where a<b<c
    # For speed, use a 2D array: pair_to_triples[p1][p2] = list of (triple_idx, third_point)
    # But that's complex. Instead, precompute for each pair (a,b) the list of triples containing them.
    
    # Actually, let's precompute: for each point x, and for each pair (i,j) in block,
    # we need to check if (x,i,j) is an uncovered triple.
    # We can precompute a 3D lookup or use the triple_to_idx dict.
    
    # For t=3, precompute pair_triples: for each pair (a,b), list of (c, triple_idx) where a<b<c or a<c<b etc.
    # Simpler: just use triple_to_idx with sorted tuples.
    
    while uncovered_count > 0:
        # Pick a random uncovered triple
        # Find an uncovered triple index
        while True:
            idx = rng.randint(0, n_triples - 1)
            if uncovered[idx]:
                break
        
        triple = triples[idx]
        block = list(triple)
        block_set = set(block)
        
        # Mark this triple as covered
        uncovered[idx] = False
        uncovered_count -= 1
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain: how many uncovered triples are covered by adding x
                # For t=3: triples (x, bi, bj) for all pairs bi,bj in block
                gain = 0
                if t == 3:
                    n_block = len(block)
                    for i in range(n_block):
                        for j in range(i + 1, n_block):
                            a, b = block[i], block[j]
                            # Sort (x, a, b) to use triple_to_idx
                            tri = tuple(sorted((x, a, b)))
                            tri_idx = triple_to_idx[tri]
                            if uncovered[tri_idx]:
                                gain += 1
                else:
                    for comb in itertools.combinations(block, t - 1):
                        tri = tuple(sorted(list(comb) + [x]))
                        tri_idx = triple_to_idx[tri]
                        if uncovered[tri_idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if candidates:
                next_point = rng.choice(candidates)
            else:
                next_point = rng.choice([x for x in all_points if x not in block_set])
            
            block.append(next_point)
            block_set.add(next_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered: mark all triples in this block as covered
        for tri in itertools.combinations(block, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            if uncovered[tri_idx]:
                uncovered[tri_idx] = False
                uncovered_count -= 1
        
        # Early termination if no more uncovered
        if uncovered_count == 0:
            break
    
    return blocks

def _local_search_fast(v, k, t, blocks, triples, triple_to_idx, deadline):
    """Try to remove blocks if the covering remains valid, using coverage counts."""
    blocks = [list(b) for b in blocks]
    n_triples = len(triples)
    
    # Build coverage count array
    cov_count = [0] * n_triples
    for b in blocks:
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            cov_count[tri_idx] += 1
            
    # Identify critical triples (count == 1)
    critical_triples = set()
    for i in range(n_triples):
        if cov_count[i] == 1:
            critical_triples.add(i)

    improved = True
    rng = random.Random(42)
    
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            # Try removing block i
            b = blocks[i]
            block_triples = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
            
            # Check if any triple in this block is critical (count 1)
            # If so, removing it makes count 0 -> invalid
            valid = True
            for tri_idx in block_triples:
                if tri_idx in critical_triples:
                    valid = False
                    break
            
            if valid:
                # Removal successful
                for tri_idx in block_triples:
                    cov_count[tri_idx] -= 1
                    if cov_count[tri_idx] == 1:
                        critical_triples.add(tri_idx)
                    elif cov_count[tri_idx] == 2:
                        critical_triples.discard(tri_idx)
                
                blocks.pop(i)
                improved = True
                break
            # If invalid, do nothing (coverage wasn't changed)
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Pre-compute
    triples, triple_to_idx = _precompute_triples(v, t)
    all_points = list(range(v))
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # Check if we have enough time for a full greedy + local search
        if time.time() + 5.0 > deadline:
            break
        
        blocks = _greedy_fast(v, k, t, seed, triples, triple_to_idx, all_points)
        
        # Apply local search if we have time
        if time.time() + 10.0 < deadline:
            blocks = _local_search_fast(v, k, t, blocks, triples, triple_to_idx, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        # Stop early if we hit theoretical lower bound
        if best_len == 69:
            break
        
        seed += 1
    
    return best_blocks
