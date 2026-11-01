import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    # Map t-subsets to integer IDs for faster lookup and counting
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)
    
    best_blocks = None
    best_len = float('inf')
    
    def random_greedy_cover():
        # uncovered[i] is 1 if triple i is uncovered
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        
        while True:
            # Check if all covered
            if not np.any(uncovered):
                break
            
            # Pick a random uncovered triple
            seed_idx = None
            while True:
                candidate = random.randint(0, num_triples - 1)
                if uncovered[candidate]:
                    seed_idx = candidate
                    break
            
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            
            # Extend block to size k
            while len(block) < k:
                best_gain = -1
                best_points = []
                block_set = set(block)
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in all_points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for pair in block_pairs:
                        triple = tuple(sorted((x,) + pair))
                        idx = triple_to_idx[triple]
                        if uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_points:
                    block.append(random.choice(best_points))
                else:
                    available = [x for x in all_points if x not in block_set]
                    if available:
                        block.append(random.choice(available))
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0
        
        return blocks

    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # triple_count[idx] = number of blocks covering triple idx
        triple_count = np.zeros(num_triples, dtype=np.int32)
        
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                triple_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[tuple(triple)]
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                for i, block in enumerate(blocks):
                    if i not in remove_set:
                        new_blocks.append(block)
                
                # Update counts
                for i in to_remove:
                    block = blocks[i]
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                
                blocks = new_blocks
                changed = True
                
        return blocks

    # Main loop
    while time.time() < deadline:
        blocks = random_greedy_cover()
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks if best_blocks is not None else []
