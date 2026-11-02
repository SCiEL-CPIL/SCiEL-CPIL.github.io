import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Precompute for each point, which t-subsets contain it
    point_to_t_subs = [[] for _ in range(v)]
    for sub_idx, sub in enumerate(all_t_subs):
        for p in sub:
            point_to_t_subs[p].append(sub_idx)
    
    # Helper to get indices of t-subsets of a block
    def get_block_t_sub_indices(block):
        return [t_sub_index[sub] for sub in itertools.combinations(block, t)]
    
    # Precompute t-subset indices for all possible k-subsets is too expensive
    # Instead, precompute for each (t-1)-subset and point, the t-subset index
    # This allows fast gain calculation during greedy
    t_minus_1_subs = list(itertools.combinations(range(v), t-1))
    t_minus_1_index = {sub: i for i, sub in enumerate(t_minus_1_subs)}
    n_t_minus_1 = len(t_minus_1_subs)
    
    # For each (t-1)-subset, store which t-subsets it's contained in (with which point)
    # t_minus_1_to_t: dict from (t-1)-subset to list of (point, t_subset_index)
    t_minus_1_to_t = {}
    for sub_idx, sub in enumerate(all_t_subs):
        for i in range(t):
            # Remove the i-th element to get a (t-1)-subset
            rest = tuple(sub[:i] + sub[i+1:])
            if rest not in t_minus_1_to_t:
                t_minus_1_to_t[rest] = []
            t_minus_1_to_t[rest].append((sub[i], sub_idx))

    # 1. Greedy Construction with fast gain calculation
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
            # Use numpy for faster random selection from sparse set
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
                        if rest_tuple in t_minus_1_to_t:
                            for (pt, sub_idx) in t_minus_1_to_t[rest_tuple]:
                                if pt == p and uncovered_mask[sub_idx]:
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

    # The above greedy is a bit messy with the uncovered list.
    # Let's refine.
    
    def greedy_construct_v2(seed):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered = set(range(n_t_subs))
        
        while uncovered:
            # Pick random uncovered
            idx = random.choice(list(uncovered))
            target_sub = all_t_subs[idx]
            
            block = list(target_sub)
            
            while len(block) < k:
                best_p = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block]
                
                for p in candidates:
                    gain = 0
                    # t-subsets of block U {p} containing p
                    # choose t-1 from block
                    for rest in itertools.combinations(block, t-1):
                        sub = tuple(sorted(rest + (p,)))
                        sub_idx = t_sub_index.get(sub, -1)
                        if sub_idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                    elif gain == best_gain and best_p != -1:
                        if random.random() < 0.5:
                            best_p = p
                
                if best_p == -1: break
                block.append(best_p)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx = t_sub_index[sub]
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered.discard(idx)
                    
        return blocks

    # 2. Local Search
    def local_search(initial_blocks, time_limit):
        blocks = [list(b) for b in initial_blocks]
        counts = [0] * n_t_subs
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[t_sub_index[sub]] += 1
                
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        
        while time.time() < time_limit:
            # Try to remove redundant blocks
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
                        break # Restart loop as indices changed
            
            if time.time() >= time_limit:
                break
            
            # Perform random swaps
            # Number of swaps depends on time? Just do a batch.
            for _ in range(100):
                if not blocks: break
                i = random.randint(0, len(blocks)-1)
                b = blocks[i]
                x = random.choice(b)
                y = random.choice([p for p in range(v) if p not in b])
                
                # Check validity
                # Subsets of b containing x
                valid = True
                for sub in itertools.combinations(b, t):
                    if x in sub and counts[t_sub_index[sub]] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    # Decrement counts for subsets of b containing x
                    for sub in itertools.combinations(b, t):
                        if x in sub:
                            counts[t_sub_index[sub]] -= 1
                    
                    # Update block
                    b.remove(x)
                    b.append(y)
                    b.sort()
                    blocks[i] = b
                    
                    # Increment counts for subsets of new b containing y
                    for sub in itertools.combinations(b, t):
                        if y in sub:
                            counts[t_sub_index[sub]] += 1
        
        return best_blocks

    # Run multiple greedy constructions and pick the best one for local search
    best_initial = None
    best_initial_len = float('inf')
    
    # Run greedy variants until we run out of time
    num_greedy = 0
    for i in range(10):
        if time.time() > deadline - 30:
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
