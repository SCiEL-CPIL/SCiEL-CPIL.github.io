import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Precompute for each point, which t-subsets contain it
    point_to_t_subs = [[] for _ in range(v)]
    for sub_idx, sub in enumerate(all_t_subs):
        for p in sub:
            point_to_t_subs[p].append(sub_idx)
    
    # Precompute t-subset indices for pairs of points
    # For each pair (a,b), store list of t-subset indices that contain both a and b
    pair_to_t_subs = {}
    for sub_idx, sub in enumerate(all_t_subs):
        for i in range(t):
            for j in range(i+1, t):
                key = (sub[i], sub[j])
                if key not in pair_to_t_subs:
                    pair_to_t_subs[key] = []
                pair_to_t_subs[key].append(sub_idx)
    
    # Helper to get indices of t-subsets of a block
    def get_block_t_sub_indices(block):
        return [t_sub_index[sub] for sub in itertools.combinations(block, t)]
    
    # Greedy construction with fast gain calculation using precomputed structures
    def greedy_construct(seed):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered_mask = bytearray([1]) * n_t_subs  # 1 if uncovered
        
        while True:
            # Find any uncovered subset
            uncovered_count = sum(uncovered_mask)
            if uncovered_count == 0:
                break
            
            # Pick random uncovered subset
            uncovered_indices = np.nonzero(uncovered_mask)[0]
            idx = int(random.choice(uncovered_indices))
            target_sub = all_t_subs[idx]
            
            block = list(target_sub)
            block_set = set(block)
            
            while len(block) < k:
                best_p = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                
                for p in candidates:
                    # Gain = number of uncovered t-subsets of (block ∪ {p}) that contain p
                    # These are formed by choosing t-1 elements from block
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        rest_tuple = tuple(rest)
                        # Find t-subsets containing p and all elements in rest
                        # This means finding t-subsets that contain p and (t-1) elements from block
                        # We can iterate over pairs or use a different approach
                        # For each (t-1)-subset of block, find t-subsets containing p and that (t-1)-subset
                        # Actually, we need t-subsets of block∪{p} that contain p
                        # These are: {p} ∪ S where S is a (t-1)-subset of block
                        for s in itertools.combinations(block, t-1):
                            sub = tuple(sorted(s + (p,)))
                            sub_idx = t_sub_index.get(sub, -1)
                            if sub_idx != -1 and uncovered_mask[sub_idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                    elif gain == best_gain and best_p != -1:
                        if random.random() < 0.5:
                            best_p = p
                
                if best_p == -1:
                    break
                block.append(best_p)
                block_set.add(best_p)
            
            block.sort()
            blocks.append(block)
            
            # Update counts and uncovered mask
            for sub in itertools.combinations(block, t):
                idx = t_sub_index[sub]
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered_mask[idx] = 0
                    
        return blocks

    # Improved local search with faster operations
    def local_search(initial_blocks, time_limit):
        blocks = [list(b) for b in initial_blocks]
        counts = [0] * n_t_subs
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[t_sub_index[sub]] += 1
                
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        
        # Precompute for each block, the set of t-subset indices it covers
        # and for each point in block, which of those t-subsets contain that point
        
        while time.time() < time_limit:
            # Phase 1: Remove redundant blocks
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    b = blocks[i]
                    # Check if all t-subsets of b have count > 1
                    removable = True
                    for sub in itertools.combinations(b, t):
                        if counts[t_sub_index[sub]] <= 1:
                            removable = False
                            break
                    if removable:
                        # Remove block i
                        for sub in itertools.combinations(b, t):
                            counts[t_sub_index[sub]] -= 1
                        blocks.pop(i)
                        changed = True
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best_blocks = [list(x) for x in blocks]
                        break  # Restart loop as indices changed
            
            if time.time() >= time_limit:
                break
            
            # Phase 2: Random swaps and block replacements
            num_swaps = 500
            for _ in range(num_swaps):
                if time.time() >= time_limit:
                    break
                if not blocks:
                    break
                    
                # Try a swap: replace one point in a random block
                i = random.randint(0, len(blocks)-1)
                b = blocks[i]
                x = random.choice(b)
                
                # Check if removing x would uncover any t-subsets
                valid = True
                for sub in itertools.combinations(b, t):
                    if x in sub and counts[t_sub_index[sub]] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Find best replacement point y
                    best_y = -1
                    best_gain = -1
                    candidates = [p for p in range(v) if p not in b]
                    
                    for y in candidates:
                        # Gain = number of currently uncovered t-subsets that would be covered
                        # by adding y to b (i.e., t-subsets of b∪{y} containing y that are uncovered)
                        gain = 0
                        for sub in itertools.combinations(b, t):
                            new_sub = tuple(sorted((list(sub) + [y])))
                            if len(set(new_sub)) == t:  # y not already in sub
                                sub_idx = t_sub_index.get(new_sub, -1)
                                if sub_idx != -1 and counts[sub_idx] == 0:
                                    gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_y = y
                        elif gain == best_gain and best_y != -1:
                            if random.random() < 0.3:
                                best_y = y
                    
                    if best_y != -1 and best_gain > 0:
                        # Perform swap: remove x, add best_y
                        # Decrement counts for t-subsets of b containing x
                        for sub in itertools.combinations(b, t):
                            if x in sub:
                                counts[t_sub_index[sub]] -= 1
                        
                        # Update block
                        b.remove(x)
                        b.append(best_y)
                        b.sort()
                        blocks[i] = b
                        
                        # Increment counts for t-subsets of new b containing best_y
                        for sub in itertools.combinations(b, t):
                            if best_y in sub:
                                counts[t_sub_index[sub]] += 1
                        
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best_blocks = [list(x) for x in blocks]
                
                # Occasionally try to replace an entire block with a better one
                if random.random() < 0.1 and len(blocks) > 0:
                    i = random.randint(0, len(blocks)-1)
                    b = blocks[i]
                    
                    # Find a new block that covers more uncovered subsets
                    # Start from a random uncovered subset
                    uncovered_idx = [j for j in range(n_t_subs) if counts[j] == 0]
                    if uncovered_idx:
                        target_idx = random.choice(uncovered_idx)
                        target_sub = all_t_subs[target_idx]
                        
                        # Build a block around target_sub
                        new_block = list(target_sub)
                        new_block_set = set(new_block)
                        
                        while len(new_block) < k:
                            best_p = -1
                            best_gain = -1
                            candidates = [p for p in range(v) if p not in new_block_set]
                            
                            for p in candidates:
                                gain = 0
                                for rest in itertools.combinations(new_block, t-1):
                                    sub = tuple(sorted(rest + (p,)))
                                    sub_idx = t_sub_index.get(sub, -1)
                                    if sub_idx != -1 and counts[sub_idx] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_p = p
                            
                            if best_p == -1:
                                break
                            new_block.append(best_p)
                            new_block_set.add(best_p)
                        
                        new_block.sort()
                        
                        # Check if replacing block i with new_block is beneficial
                        # Count how many new t-subsets new_block covers that b doesn't
                        old_subs = set(t_sub_index[sub] for sub in itertools.combinations(b, t))
                        new_subs = set(t_sub_index[sub] for sub in itertools.combinations(new_block, t))
                        
                        # Only replace if new_block covers some uncovered subsets
                        # that b doesn't cover, and doesn't lose coverage of currently covered subsets
                        # that would become uncovered
                        gain = sum(1 for s in new_subs if s not in old_subs and counts[s] == 0)
                        loss = sum(1 for s in old_subs if s not in new_subs and counts[s] == 1)
                        
                        if gain > loss:
                            # Replace block i with new_block
                            for sub in itertools.combinations(b, t):
                                counts[t_sub_index[sub]] -= 1
                            for sub in itertools.combinations(new_block, t):
                                counts[t_sub_index[sub]] += 1
                            blocks[i] = new_block
                            
                            if len(blocks) < best_len:
                                best_len = len(blocks)
                                best_blocks = [list(x) for x in blocks]
        
        return best_blocks

    # Run multiple greedy constructions and pick the best one for local search
    best_initial = None
    best_initial_len = float('inf')
    
    # Run greedy variants until we run out of time
    num_greedy = 0
    for i in range(15):
        if time.time() > deadline - 25:
            break
        blocks = greedy_construct(i)
        num_greedy += 1
        if len(blocks) < best_initial_len:
            best_initial_len = len(blocks)
            best_initial = blocks
            
    if best_initial is None:
        # Fallback: simple greedy
        best_initial = greedy_construct(0)
        
    # Local search on the best initial
    final_blocks = local_search(best_initial, deadline)
    
    return final_blocks
