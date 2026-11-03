import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # Precompute all t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    ts_set = set(all_ts)
    
    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(all_ts)
        blocks = []
        
        while uncovered:
            # Pick a starting subset
            if seed is None:
                first_uncovered = min(uncovered)
            else:
                # Faster random choice
                first_uncovered = random.choice(list(uncovered))
            
            block = list(first_uncovered)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                block_combos = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in block_combos:
                        s = tuple(sorted(rest + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if best_gain == len(block_combos):
                            break
                if best_x is None:
                    # Fallback, should not happen
                    for x in range(v):
                        if x not in block:
                            best_x = x
                            break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for s in itertools.combinations(block, t):
                if s in uncovered:
                    uncovered.remove(s)
                    
        return blocks

    def local_search(blocks):
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            
            # 1. Remove redundant blocks
            cover_count = {}
            for block in blocks:
                for s in itertools.combinations(block, t):
                    cover_count[s] = cover_count.get(s, 0) + 1
            
            for i in range(len(blocks)):
                block = blocks[i]
                is_redundant = True
                for s in itertools.combinations(block, t):
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s in itertools.combinations(block, t):
                        cover_count[s] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
            
            if changed:
                continue

            # 2. Try to replace pairs of blocks with a single block (Merge)
            # Only try if time permits and blocks > 1
            if len(blocks) > 1 and time.time() - start_time < time_limit - 10:
                # Recompute cover_count if it was modified or not initialized in this pass
                # Note: cover_count is valid from the redundancy check pass above if no changes were made
                n = len(blocks)
                merge_found = False
                # Randomize order to avoid bias?
                indices = list(range(n))
                random.shuffle(indices)
                
                for idx_i in indices:
                    if time.time() - start_time > time_limit - 5:
                        return blocks
                    i = idx_i
                    if i >= len(blocks): continue # Safety
                    
                    for idx_j in indices:
                        if time.time() - start_time > time_limit - 5:
                            return blocks
                        j = idx_j
                        if j <= i or j >= len(blocks): continue
                        
                        # Collect critical triples
                        crit_set = set()
                        
                        for s in itertools.combinations(blocks[i], t):
                            if cover_count[s] == 1:
                                crit_set.add(s)
                            elif cover_count[s] == 2:
                                if all(x in blocks[j] for x in s):
                                    crit_set.add(s)
                        
                        for s in itertools.combinations(blocks[j], t):
                            if cover_count[s] == 1:
                                crit_set.add(s)
                            elif cover_count[s] == 2:
                                if all(x in blocks[i] for x in s):
                                    crit_set.add(s)
                        
                        if not crit_set:
                            continue

                        if len(crit_set) > k:
                            continue

                        # Try to construct a new block B_new containing all crit_set elements
                        P = set()
                        for s in crit_set:
                            P.update(s)
                        
                        if len(P) > k:
                            continue

                        available = [x for x in range(v) if x not in P]
                        if len(available) < k - len(P):
                            continue
                        
                        extra = random.sample(available, k - len(P))
                        B_new = sorted(list(P) + extra)
                        
                        # Verify B_new is valid (distinct elements)
                        if len(set(B_new)) != k:
                            continue

                        # Perform merge
                        # Remove j and i (j > i? No, indices are from shuffled list, need care)
                        # Safer: create new list
                        new_blocks = []
                        for idx, b in enumerate(blocks):
                            if idx != i and idx != j:
                                new_blocks.append(b)
                        new_blocks.append(B_new)
                        
                        # Check if this merge actually reduces count and maintains coverage
                        # It reduces count by 1.
                        # We assume coverage is maintained by construction of crit_set.
                        
                        blocks = new_blocks
                        changed = True
                        merge_found = True
                        break
                    if merge_found:
                        break
            
        return blocks

    # Main logic
    best_blocks = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy + Local Search
    blocks = greedy_construct()
    blocks = local_search(blocks)
    best_blocks = blocks
    best_len = len(blocks)
    
    # 2. Random Restarts
    attempts = 0
    while time.time() - start_time < time_limit:
        seed = random.randint(0, 1000000)
        blocks = greedy_construct(seed)
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        attempts += 1
        if attempts > 200: # Safety
            break

    return best_blocks
