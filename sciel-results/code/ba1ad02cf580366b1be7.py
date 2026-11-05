import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_triples = list(itertools.combinations(range(v), t))
    best_blocks = None
    best_len = float('inf')
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts)
        
        blocks = greedy_cover(v, k, t, all_triples)
        blocks = remove_redundant_blocks(blocks, v, t)
        
        # Quick check for distinctness and validity of points
        if len(set(tuple(sorted(b)) for b in blocks)) == len(blocks):
             if len(blocks) < best_len:
                 best_len = len(blocks)
                 best_blocks = blocks
        
    if best_blocks is None:
        return []
    return best_blocks

def greedy_cover(v, k, t, all_triples):
    uncovered = set(all_triples)
    blocks = []
    
    while uncovered:
        seed = random.choice(list(uncovered))
        block = list(seed)
        
        while len(block) < k:
            block_combos = list(itertools.combinations(block, t-1))
            block_set = set(block)
            
            best_x = -1
            max_gain = -1
            for x in range(v):
                if x in block_set: continue
                gain = 0
                for comb in block_combos:
                    triple = tuple(sorted(comb + (x,)))
                    if triple in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_x = x
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return blocks

def remove_redundant_blocks(blocks, v, t):
    if not blocks:
        return blocks
        
    triple_counts = {}
    for block in blocks:
        for triple in itertools.combinations(block, t):
            triple_counts[triple] = triple_counts.get(triple, 0) + 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            redundant = True
            for triple in itertools.combinations(block, t):
                if triple_counts[triple] == 1:
                    redundant = False
                    break
            
            if redundant:
                for triple in itertools.combinations(block, t):
                    triple_counts[triple] -= 1
                changed = True
            else:
                new_blocks.append(block)
        blocks = new_blocks
        
    return blocks
