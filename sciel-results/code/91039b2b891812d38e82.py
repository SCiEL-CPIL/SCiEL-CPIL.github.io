import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # 2. Greedy construction
    def greedy_construct(seed, time_limit=None):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered = set(range(n_t_subs))
        
        while uncovered and (time_limit is None or time.time() < time_limit):
            # Pick a random uncovered subset
            idx = random.choice(list(uncovered))
            target_sub = all_t_subs[idx]
            
            block = list(target_sub)
            
            while len(block) < k:
                best_p = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block]
                
                for p in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        sub = tuple(sorted(rest + (p,)))
                        sub_idx = t_sub_index.get(sub, -1)
                        if sub_idx is not None and sub_idx in uncovered:
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
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx2 = t_sub_index[sub]
                counts[idx2] += 1
                if counts[idx2] == 1:
                    uncovered.discard(idx2)
                    
        return blocks
    
    # 3. Local Search
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
            for _ in range(200):
                if time.time() >= time_limit:
                    break
                if not blocks:
                    break
                i = random.randint(0, len(blocks)-1)
                b = blocks[i]
                x = random.choice(b)
                y = random.choice([p for p in range(v) if p not in b])
                
                # Check validity
                valid = True
                for sub in itertools.combinations(b, t):
                    if x in sub and counts[t_sub_index[sub]] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
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
                    
                    # Try to remove redundant blocks after swap
                    changed2 = True
                    while changed2:
                        changed2 = False
                        for j in range(len(blocks)):
                            bj = blocks[j]
                            removable2 = True
                            for sub2 in itertools.combinations(bj, t):
                                if counts[t_sub_index[sub2]] <= 1:
                                    removable2 = False
                                    break
                            if removable2:
                                for sub2 in itertools.combinations(bj, t):
                                    counts[t_sub_index[sub2]] -= 1
                                blocks.pop(j)
                                changed2 = True
                                if len(blocks) < best_len:
                                    best_len = len(blocks)
                                    best_blocks = [list(x2) for x2 in blocks]
                                break
        
        return best_blocks
    
    # Run multiple greedy constructions with different seeds
    best_initial = None
    best_initial_len = float('inf')
    
    # Time budget allocation
    greedy_time_per = 15
    num_greedy = 4
    
    greedy_deadline = start_time + greedy_time_per * num_greedy + 5
    
    for i in range(num_greedy):
        if time.time() > greedy_deadline:
            break
        blocks = greedy_construct(i, time.time() + greedy_time_per)
        if len(blocks) < best_initial_len:
            best_initial_len = len(blocks)
            best_initial = blocks
    
    if best_initial is None:
        best_initial = greedy_construct(0)
    
    # Local search on the best initial
    final_blocks = local_search(best_initial, deadline)
    
    return final_blocks
