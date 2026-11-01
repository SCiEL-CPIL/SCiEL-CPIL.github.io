import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    all_points = list(range(v))
    all_triples = list(itertools.combinations(range(v), t))
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    def greedy_cover(rng):
        block_cache = {}
        uncovered = [True] * n_triples
        uncovered_count = n_triples
        blocks = []
        
        while uncovered_count > 0:
            best_block = None
            best_gain = -1
            best_tris = None
            
            uncovered_indices = [i for i in range(n_triples) if uncovered[i]]
            if not uncovered_indices:
                break
                
            n_anchors = min(len(uncovered_indices), 6)
            anchor_indices = rng.sample(uncovered_indices, n_anchors)
            
            for ai in anchor_indices:
                anchor = all_triples[ai]
                anchor_set = set(anchor)
                remaining = [p for p in all_points if p not in anchor_set]
                
                for combo in itertools.combinations(remaining, k - t):
                    block = tuple(sorted(anchor_set | set(combo)))
                    if block not in block_cache:
                        block_cache[block] = [triple_index[tr] for tr in itertools.combinations(block, t)]
                    tris = block_cache[block]
                    gain = sum(uncovered[i] for i in tris)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                        best_tris = tris
            
            blocks.append(list(best_block))
            for idx in best_tris:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        
        return blocks
    
    def remove_redundant(blocks):
        coverage = [0] * n_triples
        for block in blocks:
            for tr in itertools.combinations(block, t):
                coverage[triple_index[tr]] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for block in blocks:
                can_remove = True
                for tr in itertools.combinations(block, t):
                    if coverage[triple_index[tr]] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for tr in itertools.combinations(block, t):
                        coverage[triple_index[tr]] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
        return blocks
    
    best_blocks = None
    best_count = float('inf')
    attempts = 0
    
    while time.time() < deadline:
        rng = random.Random(attempts)
        blocks = greedy_cover(rng)
        blocks = remove_redundant(blocks)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
        attempts += 1
    
    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = greedy_cover(rng)
        best_blocks = remove_redundant(best_blocks)
    
    return best_blocks
