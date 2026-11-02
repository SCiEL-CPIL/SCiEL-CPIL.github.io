import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Helper to get indices of t-subsets of a block
    def get_block_t_sub_indices(block):
        return [t_sub_index[sub] for sub in itertools.combinations(block, t)]

    # 1. Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered = set(range(n_t_subs))
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered subset to start
            idx = random.choice(list(uncovered))
            target_sub = all_t_subs[idx]
            
            # Build a block around this subset
            block = list(target_sub)
            
            # Greedily extend to k elements
            while len(block) < k:
                best_p = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block]
                
                for p in candidates:
                    gain = 0
                    # Count how many new t-subsets this point adds
                    for rest in itertools.combinations(block, t-1):
                        sub = tuple(sorted(rest + (p,)))
                        sub_idx = t_sub_index.get(sub, -1)
                        if sub_idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                    elif gain == best_gain and best_p != -1 and random.random() < 0.3:
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

    # 2. Local Search
    def local_search(initial_blocks, time_limit):
        blocks = [list(b) for b in initial_blocks]
        counts = [0] * n_t_subs
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[t_sub_index[sub]] += 1
                
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        
        # Function to check if a block can be removed
        def can_remove(block_idx):
            b = blocks[block_idx]
            for sub in itertools.combinations(b, t):
                if counts[t_sub_index[sub]] <= 1:
                    return False
            return True

        # Function to remove a block
        def remove_block(block_idx):
            b = blocks[block_idx]
            for sub in itertools.combinations(b, t):
                counts[t_sub_index[sub]] -= 1
            blocks.pop(block_idx)

        # Main local search loop
        while time.time() < time_limit:
            # Phase 1: Remove redundant blocks
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    if can_remove(i):
                        remove_block(i)
                        changed = True
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best_blocks = [list(x) for x in blocks]
                        break # Restart loop as indices changed
            
            if time.time() >= time_limit:
                break
            
            # Phase 2: Try element swaps to enable more removals
            # We try to swap an element in a block with an element outside the block
            # if it doesn't uncover any t-subsets.
            swaps_attempted = 0
            max_swaps = 1000
            while swaps_attempted < max_swaps and time.time() < time_limit:
                swaps_attempted += 1
                if not blocks:
                    break
                i = random.randint(0, len(blocks)-1)
                b = blocks[i]
                x = random.choice(b)
                y = random.choice([p for p in range(v) if p not in b])
                
                # Check if swapping x with y is safe (doesn't uncover any t-subset)
                valid = True
                # Subsets of b containing x
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
                    
                    # Check for new redundancies
                    if len(blocks) > 0:
                        # Try to remove any block that became redundant
                        for j in range(len(blocks)):
                            if j != i and can_remove(j):
                                remove_block(j)
                                if len(blocks) < best_len:
                                    best_len = len(blocks)
                                    best_blocks = [list(x) for x in blocks]
                                break # Restart swap loop
                        if i < len(blocks) and can_remove(i):
                            remove_block(i)
                            if len(blocks) < best_len:
                                best_len = len(blocks)
                                best_blocks = [list(x) for x in blocks]
                            break # Restart swap loop
        
        return best_blocks

    # Strategy: Run greedy with a few seeds, then local search on the best
    best_initial = None
    best_initial_len = float('inf')
    
    # Time budget for greedy
    greedy_time = 20
    
    for i in range(3):
        if time.time() > deadline - greedy_time:
            break
        blocks = greedy_construct(i)
        if len(blocks) < best_initial_len:
            best_initial_len = len(blocks)
            best_initial = blocks
    
    if best_initial is None:
        best_initial = greedy_construct(0)
    
    # Local search on the best initial
    final_blocks = local_search(best_initial, deadline)
    
    return final_blocks
