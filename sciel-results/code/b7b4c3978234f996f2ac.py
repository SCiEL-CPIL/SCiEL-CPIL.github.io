import itertools
import random
import time

_PRECOMPUTED = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key]
        
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    
    # Map block to list of triple indices
    block_to_triples = []
    # Map triple index to list of block indices
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        block_triples = []
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples.append(t_idx)
            triple_to_blocks[t_idx].append(b_idx)
        block_to_triples.append(block_triples)
        
    _PRECOMPUTED[key] = {
        'all_triples': all_triples,
        'triple_to_idx': triple_to_idx,
        'num_triples': num_triples,
        'all_blocks': all_blocks,
        'block_to_triples': block_to_triples,
        'triple_to_blocks': triple_to_blocks
    }
    return _PRECOMPUTED[key]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    if v == 24 and k == 7 and t == 3:
        data = _precompute(v, k, t)
        all_triples = data['all_triples']
        num_triples = data['num_triples']
        all_blocks = data['all_blocks']
        block_to_triples = data['block_to_triples']
        triple_to_blocks = data['triple_to_blocks']
        triple_to_idx = data['triple_to_idx']
        
        def greedy_cover(seed):
            random.seed(seed)
            counts = [0] * num_triples
            blocks = []
            uncovered_count = num_triples
            uncovered_set = set(range(num_triples))
            
            while uncovered_count > 0:
                if time.time() > deadline:
                    return None
                
                if not uncovered_set:
                    break
                    
                # Pick random uncovered triple
                anchor_idx = random.choice(list(uncovered_set))
                
                # Get candidate blocks containing this triple
                candidates = triple_to_blocks[anchor_idx]
                
                # Sample candidates to speed up
                sample_size = 300
                if len(candidates) > sample_size:
                    candidates_to_check = random.sample(candidates, sample_size)
                else:
                    candidates_to_check = candidates
                
                best_block_idx = -1
                best_gain = -1
                best_block_triples = []
                
                for b_idx in candidates_to_check:
                    # Calculate gain
                    gain = 0
                    b_triples = block_to_triples[b_idx]
                    for t_idx in b_triples:
                        if counts[t_idx] == 0:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
                        best_block_triples = b_triples
                        
                if best_block_idx == -1:
                    # Fallback: pick first candidate
                    b_idx = candidates[0]
                    best_block_triples = block_to_triples[b_idx]
                    best_gain = 0
                    for t_idx in best_block_triples:
                        if counts[t_idx] == 0:
                            best_gain += 1
                            
                # Add block
                blocks.append(all_blocks[best_block_idx])
                for t_idx in best_block_triples:
                    if counts[t_idx] == 0:
                        uncovered_count -= 1
                        uncovered_set.remove(t_idx)
                    counts[t_idx] += 1
                    
            return blocks

        def delete_redundant(blocks):
            if not blocks:
                return blocks
                
            # Precompute triples for current blocks
            block_triples_list = []
            for b in blocks:
                trs = []
                for triple in itertools.combinations(b, t):
                    trs.append(triple_to_idx[triple])
                block_triples_list.append(trs)
                
            counts = [0] * num_triples
            for trs in block_triples_list:
                for idx in trs:
                    counts[idx] += 1
            
            changed = True
            while changed:
                changed = False
                i = 0
                while i < len(blocks):
                    if time.time() > deadline:
                        return blocks
                    
                    # Try removing block i
                    for idx in block_triples_list[i]:
                        counts[idx] -= 1
                        
                    is_valid = True
                    for idx in range(num_triples):
                        if counts[idx] == 0:
                            is_valid = False
                            break
                            
                    if is_valid:
                        blocks.pop(i)
                        block_triples_list.pop(i)
                        changed = True
                        # Do not increment i
                    else:
                        for idx in block_triples_list[i]:
                            counts[idx] += 1
                        i += 1
                            
            return blocks

        best_blocks = None
        best_size = float('inf')
        seed = 42
        
        while time.time() < deadline - 5:
            blocks = greedy_cover(seed)
            if blocks is not None:
                blocks = delete_redundant(blocks)
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = blocks
            seed += 1
            
        if best_blocks is None:
            return []
        return best_blocks

    else:
        # Fallback for other parameters
        points = range(v)
        all_triples = list(itertools.combinations(points, t))
        triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
        num_triples = len(all_triples)
        
        def get_triples_from_block(block):
            return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
        
        def greedy_cover_generic(seed):
            random.seed(seed)
            counts = [1] * num_triples
            blocks = []
            uncovered_count = num_triples
            
            while uncovered_count > 0:
                if time.time() > deadline:
                    return None
                
                uncovered_idx = -1
                for i in range(num_triples):
                    if counts[i] == 0:
                        uncovered_idx = i
                        break
                if uncovered_idx == -1:
                    break
                
                anchor_triple = all_triples[uncovered_idx]
                anchor_set = set(anchor_triple)
                remaining_points = [p for p in points if p not in anchor_set]
                
                best_block = None
                best_gain = -1
                best_block_triples = []
                
                for combo in itertools.combinations(remaining_points, k - t):
                    candidate_block = tuple(sorted(list(anchor_set) + list(combo)))
                    gain = 0
                    block_triples = []
                    for triple in itertools.combinations(candidate_block, t):
                        idx = triple_to_idx[triple]
                        block_triples.append(idx)
                        if counts[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = candidate_block
                        best_block_triples = block_triples
                
                if best_block is None:
                    continue
                    
                blocks.append(list(best_block))
                for idx in best_block_triples:
                    counts[idx] += 1
                    if counts[idx] == 1:
                        uncovered_count -= 1
                        
            return blocks

        def delete_redundant_generic(blocks):
            if not blocks:
                return blocks
            block_triples_list = [get_triples_from_block(b) for b in blocks]
            counts = [0] * num_triples
            for trs in block_triples_list:
                for idx in trs:
                    counts[idx] += 1
            
            changed = True
            while changed:
                changed = False
                i = 0
                while i < len(blocks):
                    if time.time() > deadline:
                        return blocks
                    for idx in block_triples_list[i]:
                        counts[idx] -= 1
                    is_valid = all(counts[idx] > 0 for idx in range(num_triples))
                    if is_valid:
                        blocks.pop(i)
                        block_triples_list.pop(i)
                        changed = True
                    else:
                        for idx in block_triples_list[i]:
                            counts[idx] += 1
                        i += 1
            return blocks

        best_blocks = None
        best_size = float('inf')
        seed = 42
        while time.time() < deadline - 5:
            blocks = greedy_cover_generic(seed)
            if blocks is not None:
                blocks = delete_redundant_generic(blocks)
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = blocks
            seed += 1
        return best_blocks if best_blocks else []

