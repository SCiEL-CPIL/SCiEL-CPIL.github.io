import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        return _simple_greedy(v, k, t)
        
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute
    triple_to_id = {}
    id_to_triple = []
    idx = 0
    for triple in itertools.combinations(range(v), t):
        triple_to_id[triple] = idx
        id_to_triple.append(triple)
        idx += 1
    num_triples = len(id_to_triple)
    
    block_to_id = {}
    id_to_block = []
    block_triples = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    idx = 0
    for block in itertools.combinations(range(v), k):
        block_to_id[block] = idx
        id_to_block.append(block)
        
        t_ids = []
        for sub in itertools.combinations(block, t):
            tid = triple_to_id[sub]
            t_ids.append(tid)
            triple_to_blocks[tid].append(idx)
            
        block_triples.append(t_ids)
        idx += 1
        
    data = {
        'v': v, 'k': k, 't': t,
        'triple_to_id': triple_to_id,
        'id_to_triple': id_to_triple,
        'block_to_id': block_to_id,
        'id_to_block': id_to_block,
        'block_triples': block_triples,
        'triple_to_blocks': triple_to_blocks,
        'num_triples': num_triples
    }
    
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        # Greedy
        block_ids = _greedy_fast(data, random.randint(0, 10**6))
        
        # Local Search
        block_ids = _local_search(data, block_ids, deadline)
        
        if len(block_ids) < best_count:
            best_count = len(block_ids)
            best_blocks = block_ids
            
    if best_blocks is None:
        # Fallback
        blocks = _simple_greedy(v, k, t)
        return blocks
        
    return [list(data['id_to_block'][bid]) for bid in best_blocks]

def _greedy_fast(data, seed):
    v = data['v']
    k = data['k']
    t = data['t']
    triple_to_blocks = data['triple_to_blocks']
    block_triples = data['block_triples']
    id_to_block = data['id_to_block']
    num_triples = data['num_triples']
    
    rng = random.Random(seed)
    counts = np.zeros(num_triples, dtype=np.int16)
    current_blocks = []
    current_set = set()
    uncovered = set(range(num_triples))
    
    while uncovered:
        target_tid = rng.choice(list(uncovered))
        candidates = triple_to_blocks[target_tid]
        
        best_bid = -1
        best_gain = -1
        
        for bid in candidates:
            if bid in current_set:
                continue
            gain = 0
            for tid in block_triples[bid]:
                if counts[tid] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_bid = bid
            elif gain == best_gain:
                if rng.random() < 0.5:
                    best_bid = bid
                    
        if best_bid == -1:
            break
            
        current_blocks.append(best_bid)
        current_set.add(best_bid)
        
        for tid in block_triples[best_bid]:
            if counts[tid] == 0:
                uncovered.discard(tid)
            counts[tid] += 1
            
    return current_blocks

def _local_search(data, block_ids, deadline):
    v = data['v']
    k = data['k']
    t = data['t']
    id_to_triple = data['id_to_triple']
    id_to_block = data['id_to_block']
    block_to_id = data['block_to_id']
    block_triples = data['block_triples']
    triple_to_blocks = data['triple_to_blocks']
    num_triples = data['num_triples']
    
    counts = np.zeros(num_triples, dtype=np.int16)
    current_blocks = list(block_ids)
    current_set = set(current_blocks)
    
    for bid in current_blocks:
        for tid in block_triples[bid]:
            counts[tid] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove
        new_blocks = []
        for bid in current_blocks:
            redundant = True
            for tid in block_triples[bid]:
                if counts[tid] == 1:
                    redundant = False
                    break
            if redundant:
                for tid in block_triples[bid]:
                    counts[tid] -= 1
                improved = True
            else:
                new_blocks.append(bid)
                
        current_blocks = new_blocks
        current_set = set(current_blocks)
        
        if not current_blocks:
            break
            
        # Phase 2: Replace
        # Try to replace blocks to enable removal
        for i in range(len(current_blocks)):
            bid = current_blocks[i]
            
            # Find critical triples
            critical_tids = []
            for tid in block_triples[bid]:
                if counts[tid] == 1:
                    critical_tids.append(tid)
                    
            if not critical_tids:
                continue
                
            # Get points
            critical_points = set()
            for tid in critical_tids:
                critical_points.update(id_to_triple[tid])
                
            if len(critical_points) > k:
                continue
                
            # Generate candidates
            # Points to choose: k - len(critical_points)
            # From: v - len(critical_points)
            remaining_points = [p for p in range(v) if p not in critical_points]
            need = k - len(critical_points)
            
            found_replacement = False
            for combo in itertools.combinations(remaining_points, need):
                new_block_tuple = tuple(sorted(critical_points.union(combo)))
                
                # Check if this block is already in the set (and not the one we are replacing)
                # If it is, and it's not bid, then we are duplicating a block? 
                # Blocks must be distinct.
                # If new_block_tuple is in current_set (and != id_to_block[bid]), skip.
                # Actually, if it's in current_set, it's already covered. 
                # But we are replacing bid with new_block.
                # If new_block is already in the set (as a different block), 
                # then we would have two identical blocks? No, blocks are distinct.
                # So if new_block_tuple is in current_set, and it's not bid, 
                # then adding it would create a duplicate? 
                # Wait, we are REMOVING bid and ADDING new_block.
                # If new_block is already in the set (as another block j), 
                # then after swap, we have block j and new_block (which is same as j). 
                # So we would have duplicate blocks. Invalid.
                # So skip if new_block_tuple in current_set and id_to_block[bid] != new_block_tuple.
                
                if new_block_tuple in current_set:
                    if id_to_block[bid] != new_block_tuple:
                        continue
                    else:
                        # It's the same block, no change
                        continue
                        
                new_bid = block_to_id[new_block_tuple]
                
                # Check validity:
                # All triples in bid that are NOT in new_bid must have count >= 2 (so count-1 >= 1)
                # All triples in new_bid are fine (count increases or stays same)
                
                valid = True
                # Triples in bid
                for tid in block_triples[bid]:
                    if tid not in block_triples[new_bid]: # Optimization: check membership
                        # Actually, checking membership in a list is slow.
                        # Better: 
                        # We know new_block_tuple. 
                        # We can check if the triple is in new_block_tuple.
                        # But we have tid. 
                        # We can precompute a set for block_triples? 
                        # block_triples is a list of lists. 
                        # Converting to set for every check is slow.
                        # Let's just check: 
                        # Is triple id_to_triple[tid] a subset of new_block_tuple?
                        # This is O(3) per triple. 35 triples * 3 = 105 ops. Fast.
                        if not set(id_to_triple[tid]).issubset(new_block_tuple):
                            if counts[tid] < 2:
                                valid = False
                                break
                if not valid:
                    continue
                    
                # Perform swap
                # Remove bid
                for tid in block_triples[bid]:
                    counts[tid] -= 1
                # Add new_bid
                for tid in block_triples[new_bid]:
                    counts[tid] += 1
                    
                current_blocks[i] = new_bid
                current_set.remove(bid)
                current_set.add(new_bid)
                
                improved = True
                found_replacement = True
                break # Restart Phase 1
                
            if found_replacement:
                break # Restart Phase 1
                
    return current_blocks

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        target = min(uncovered)
        best_block = None
        best_gain = -1
        remaining_points = [p for p in range(v) if p not in target]
        need = k - t
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block
        blocks.append(best_block)
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks
