import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_triples = list(itertools.combinations(points, t))
    
    block_triples = {b: list(itertools.combinations(b, t)) for b in all_blocks}
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    # Pre-compute blocks containing each triple
    blocks_by_triple = {tr: [] for tr in all_triples}
    for b in all_blocks:
        for tr in block_triples[b]:
            blocks_by_triple[tr].append(b)

    def greedy_init(seed):
        rng = random.Random(seed)
        remaining_triples = set(all_triples)
        blocks = []
        
        while remaining_triples:
            target_triple = rng.choice(list(remaining_triples))
            candidates = blocks_by_triple[target_triple]
            
            best_block = None
            best_gain = -1
            for b in candidates:
                gain = 0
                for tr in block_triples[b]:
                    if tr in remaining_triples:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            blocks.append(best_block)
            for tr in block_triples[best_block]:
                remaining_triples.discard(tr)
                
        return blocks

    def reduce_and_search(blocks, time_limit):
        current_blocks = set(blocks)
        counts = [0] * len(all_triples)
        for b in current_blocks:
            for tr in block_triples[b]:
                counts[triple_to_idx[tr]] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Remove redundant blocks
            redundant = []
            for b in current_blocks:
                is_redundant = True
                for tr in block_triples[b]:
                    if counts[triple_to_idx[tr]] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant.append(b)
            
            if redundant:
                for b in redundant:
                    current_blocks.remove(b)
                    for tr in block_triples[b]:
                        counts[triple_to_idx[tr]] -= 1
                improved = True
                continue
            
            # 2. Try Swaps
            for b_out in list(current_blocks):
                if time.time() > time_limit:
                    break
                
                # Find uncovered triples if b_out removed
                uncovered = []
                for tr in block_triples[b_out]:
                    idx = triple_to_idx[tr]
                    if counts[idx] == 1:
                        uncovered.append(tr)
                
                if not uncovered:
                    continue # Should be redundant, but safety check
                
                # Find candidate blocks that cover all uncovered triples
                # Intersection of blocks_by_triple[tr] for tr in uncovered
                if len(uncovered) == 1:
                    candidates = blocks_by_triple[uncovered[0]]
                else:
                    # Start with the smallest list to speed up intersection
                    min_triple = min(uncovered, key=lambda tr: len(blocks_by_triple[tr]))
                    candidates = set(blocks_by_triple[min_triple])
                    for tr in uncovered:
                        if tr == min_triple:
                            continue
                        candidates.intersection_update(blocks_by_triple[tr])
                        if not candidates:
                            break
                
                found_swap = False
                for b_in in candidates:
                    if b_in in current_blocks:
                        continue
                    
                    # Perform swap
                    current_blocks.remove(b_out)
                    current_blocks.add(b_in)
                    
                    # Update counts
                    for tr in block_triples[b_in]:
                        counts[triple_to_idx[tr]] += 1
                    # Note: counts for b_out were NOT decremented in the "uncovered" check above,
                    # we just looked at them. We need to decrement them now for the actual swap.
                    for tr in block_triples[b_out]:
                        counts[triple_to_idx[tr]] -= 1
                        
                    improved = True
                    found_swap = True
                    break
                
                if found_swap:
                    break # Restart loop to check redundancies

        return list(current_blocks)

    best_blocks = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 10.0:
        blocks = greedy_init(seed)
        blocks = reduce_and_search(blocks, time.time() + 5.0)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            if best_size <= 76:
                # We can try to run more local search on this specific good solution
                pass
        seed += 1
    
    if best_blocks is None:
        best_blocks = greedy_init(0)
    
    if time.time() < deadline - 2.0:
        best_blocks = reduce_and_search(best_blocks, deadline - 2.0)
        
    return [list(b) for b in best_blocks]
