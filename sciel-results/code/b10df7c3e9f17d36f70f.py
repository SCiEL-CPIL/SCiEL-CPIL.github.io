import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    all_points = list(range(v))
    all_triples = list(itertools.combinations(all_points, t))
    n_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    def get_block_triple_indices(block):
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]

    def greedy_cover():
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            anchor_triple = random.choice(list(uncovered))
            anchor_set = set(anchor_triple)
            
            best_block = None
            best_gain = -1
            
            remaining_points = [p for p in all_points if p not in anchor_set]
            
            for cand_rest in itertools.combinations(remaining_points, k - t):
                cand_block = tuple(sorted(list(anchor_set) + list(cand_rest)))
                
                gain = 0
                for triple in itertools.combinations(cand_block, t):
                    if triple_to_idx[triple] in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = cand_block
            
            if best_block:
                blocks.append(best_block)
                for triple in itertools.combinations(best_block, t):
                    idx = triple_to_idx[triple]
                    if idx in uncovered:
                        uncovered.remove(idx)
                        
        return blocks

    def trim(blocks):
        counts = [0] * n_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                counts[triple_to_idx[triple]] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for block in blocks:
                redundant = True
                for triple in itertools.combinations(block, t):
                    if counts[triple_to_idx[triple]] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for triple in itertools.combinations(block, t):
                        counts[triple_to_idx[triple]] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
            
        return blocks

    def local_search(blocks):
        counts = [0] * n_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                counts[triple_to_idx[triple]] += 1
        
        block_triples = [get_block_triple_indices(b) for b in blocks]
        n_blocks = len(blocks)
        
        # Try to swap blocks
        while time.time() < deadline - 2:
            improved = False
            for _ in range(20): # Fewer swaps per iteration to allow more restarts
                if time.time() > deadline - 2:
                    break
                i = random.randint(0, n_blocks - 1)
                new_block = tuple(sorted(random.sample(all_points, k)))
                if new_block == blocks[i]:
                    continue
                
                # Tentative swap
                for idx in block_triples[i]:
                    counts[idx] -= 1
                new_triples = get_block_triple_indices(new_block)
                for idx in new_triples:
                    counts[idx] += 1
                
                valid = all(counts[idx] > 0 for idx in block_triples[i])
                
                if valid:
                    blocks[i] = new_block
                    block_triples[i] = new_triples
                    improved = True
                else:
                    for idx in new_triples:
                        counts[idx] -= 1
                    for idx in block_triples[i]:
                        counts[idx] += 1
            
            if improved:
                # Re-trim
                # Rebuild counts
                counts = [0] * n_triples
                for block in blocks:
                    for triple in itertools.combinations(block, t):
                        counts[triple_to_idx[triple]] += 1
                
                new_blocks = []
                for block in blocks:
                    redundant = True
                    for triple in itertools.combinations(block, t):
                        if counts[triple_to_idx[triple]] <= 1:
                            redundant = False
                            break
                    if redundant:
                        for triple in itertools.combinations(block, t):
                            counts[triple_to_idx[triple]] -= 1
                    else:
                        new_blocks.append(block)
                
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    block_triples = [get_block_triple_indices(b) for b in blocks]
                    n_blocks = len(blocks)
                else:
                    blocks = new_blocks
            else:
                # No improvement, break local search
                break
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline - 5:
        blocks = greedy_cover()
        blocks = trim(blocks)
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len <= 74:
            break

    return best_blocks
