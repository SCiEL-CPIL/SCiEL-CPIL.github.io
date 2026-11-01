import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    # Map t-subsets to integer IDs for faster lookup and counting
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)
    
    # Precompute bitsets: for each point x, which triples contain x
    # triple_bits[x] is a bitset where bit i is 1 if triple i contains point x
    triple_bits = []
    for x in range(v):
        bits = np.zeros(num_triples, dtype=np.uint8)
        for i, triple in enumerate(all_t_subsets):
            if x in triple:
                bits[i] = 1
        triple_bits.append(bits)
    
    # Precompute for each pair (a,b) with a<b, the bitset of triples containing both a and b
    # For t=3, this is all triples containing both a and b
    pair_bits = {}
    for a in range(v):
        for b in range(a+1, v):
            bits = triple_bits[a] & triple_bits[b]
            pair_bits[(a,b)] = bits
    
    best_blocks = None
    best_len = float('inf')
    
    def random_greedy_cover():
        # uncovered is a bitset: bit i is 1 if triple i is uncovered
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        
        while True:
            # Check if all covered
            if not np.any(uncovered):
                break
            
            # Pick a random uncovered triple as seed
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = random.choice(uncovered_indices.tolist())
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            block_set = set(block)
            
            # Extend block to size k using bitset-based gain computation
            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break
                
                candidates = [x for x in all_points if x not in block_set]
                if not candidates:
                    break
                
                # Compute gain for each candidate x using bitsets
                # For t=3: gain(x) = number of uncovered triples containing x and at least one pair from block
                # = sum over pairs (a,b) in block of |uncovered & triple_bits[x] & triple_bits[a] & triple_bits[b]|
                # But we need triples containing x and exactly one pair from block (the pair + x forms the triple)
                
                # For t=3, each candidate x forms triples with each pair in block
                # gain(x) = sum over pairs (a,b) in block of uncovered[triple(x,a,b)]
                # We can compute this as: for each pair (a,b) in block, get the triple index and check uncovered
                
                # More efficient: precompute which triples each candidate would cover
                # For candidate x, the triples it covers with current block are:
                # For each pair (a,b) in block, the triple (min(x,a,b), mid(x,a,b), max(x,a,b))
                
                gains = np.zeros(len(candidates), dtype=np.int32)
                
                for x_i, x in enumerate(candidates):
                    for a in block:
                        for b in block:
                            if a == b or a >= b:
                                continue
                            # Form triple with x, a, b
                            triple = tuple(sorted((x, a, b)))
                            idx = triple_to_idx[triple]
                            gains[x_i] += uncovered[idx]
                
                # Pick the candidate with highest gain, break ties randomly
                max_gain = np.max(gains)
                if max_gain <= 0:
                    # No gain, pick randomly
                    block.append(random.choice(candidates))
                else:
                    best_indices = np.where(gains == max_gain)[0]
                    chosen = random.choice(best_indices.tolist())
                    block.append(candidates[chosen])
                
                block_set.add(block[-1])
            
            if len(block) < k:
                # Fill with random points if we ran out of time
                block_set = set(block)
                while len(block) < k:
                    available = [x for x in all_points if x not in block_set]
                    if not available:
                        break
                    block.append(random.choice(available))
                    block_set.add(block[-1])
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered using bitsets
            # For each triple in the block, set uncovered[triple] = 0
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
    
    # Main loop: run multiple randomized greedy constructions
    while time.time() < deadline - 5.0:
        blocks = random_greedy_cover()
        if not blocks:
            continue
        
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
    return best_blocks if best_blocks is not None else []
