import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute pair to triples mapping for faster gain calculation
    pair_to_triples = {}
    for tri_idx, tri in enumerate(triples):
        for i in range(t):
            for j in range(i+1, t):
                pair = (tri[i], tri[j])
                if pair not in pair_to_triples:
                    pair_to_triples[pair] = []
                pair_to_triples[pair].append(tri_idx)
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 115.0
    
    rng = random.Random(42)
    
    def greedy_construct(rng):
        """Greedy construction with randomized choices."""
        counts = np.zeros(num_triples, dtype=np.int32)
        blocks = []
        
        while True:
            # Find uncovered triples
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Pick a random uncovered triple
            target_idx = rng.choice(uncovered)
            target_triple = triples[target_idx]
            
            # Start block with this triple
            block = list(target_triple)
            block_set = set(block)
            
            # Greedily add points to maximize new coverage
            while len(block) < k:
                best_gain = -1
                best_points = []
                
                for x in points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            pair = (block[i], block[j])
                            if pair in pair_to_triples:
                                for tri_idx in pair_to_triples[pair]:
                                    if x in triples[tri_idx] and counts[tri_idx] == 0:
                                        gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_points:
                    chosen = rng.choice(best_points)
                else:
                    candidates = [x for x in points if x not in block_set]
                    chosen = rng.choice(candidates)
                
                block.append(chosen)
                block_set.add(chosen)
            
            block.sort()
            blocks.append(block)
            
            # Update counts
            for tri in itertools.combinations(block, t):
                counts[triple_to_idx[tuple(tri)]] += 1
        
        return blocks, counts
    
    def local_search_deletion(blocks, counts):
        """Remove redundant blocks."""
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                essential = False
                for tri in itertools.combinations(b, t):
                    idx = triple_to_idx[tuple(tri)]
                    if counts[idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    for tri in itertools.combinations(b, t):
                        counts[triple_to_idx[tuple(tri)]] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks, counts
    
    def local_search_swap(blocks, counts, rng, max_iters=100):
        """Try swapping blocks to reduce total count."""
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() - start_time < time_limit:
            improved = False
            iterations += 1
            
            for i in range(len(blocks)):
                # Temporarily remove block i
                temp_counts = counts.copy()
                for tri in itertools.combinations(blocks[i], t):
                    temp_counts[triple_to_idx[tuple(tri)]] -= 1
                
                uncovered = np.where(temp_counts == 0)[0]
                if len(uncovered) == 0:
                    # Block is redundant
                    for tri in itertools.combinations(blocks[i], t):
                        counts[triple_to_idx[tuple(tri)]] -= 1
                    blocks.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a replacement that covers uncovered with minimal new overlap
                    best_block = None
                    best_new_triples = float('inf')
                    
                    # Try some random candidates
                    for _ in range(20):
                        cand = sorted(rng.sample(points, k))
                        cand_set = set(cand)
                        
                        # Check coverage of uncovered triples
                        covers = True
                        for unc_idx in uncovered:
                            tri = triples[unc_idx]
                            if not all(p in cand_set for p in tri):
                                covers = False
                                break
                        
                        if not covers:
                            continue
                        
                        # Count new triples covered
                        new_count = 0
                        for tri in itertools.combinations(cand, t):
                            idx = triple_to_idx[tuple(tri)]
                            if temp_counts[idx] == 0:
                                new_count += 1
                        
                        if new_count < best_new_triples:
                            best_new_triples = new_count
                            best_block = cand
                    
                    if best_block is not None:
                        # Replace
                        blocks[i] = best_block
                        for tri in itertools.combinations(best_block, t):
                            counts[triple_to_idx[tuple(tri)]] += 1
                        improved = True
        
        return blocks, counts
    
    iteration = 0
    while time.time() - start_time < time_limit:
        # Greedy construction
        blocks, counts = greedy_construct(rng)
        
        # Local search: deletion
        blocks, counts = local_search_deletion(blocks, counts)
        
        # Local search: swap
        if time.time() - start_time < time_limit - 5:
            blocks, counts = local_search_swap(blocks, counts, rng)
        
        # Update best
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
        
        iteration += 1
    
    return best_blocks
