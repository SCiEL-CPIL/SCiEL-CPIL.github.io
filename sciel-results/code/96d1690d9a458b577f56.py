import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    m = len(all_t_subsets)
    
    best_blocks = None
    best_size = float('inf')
    
    # Time limit management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Helper to check if a block is a valid k-subset
    # (Assumed valid by construction)
    
    # Helper to get t-subsets of a block
    def get_block_t_subsets(block):
        return list(itertools.combinations(block, t))
        
    # Helper to check if a block covers a list of t-subsets
    def block_covers_subsets(block, subsets):
        block_set = set(block)
        for s in subsets:
            for x in s:
                if x not in block_set:
                    return False
        return True

    # Greedy Construction (Simplified for speed, similar to original)
    def greedy_construct():
        cov = [0] * m
        blocks = []
        
        while True:
            min_cov_val = min(cov)
            if min_cov_val > 0:
                break
            
            candidates = [i for i in range(m) if cov[i] == min_cov_val]
            idx = random.choice(candidates)
            seed_subset = all_t_subsets[idx]
            
            block = list(seed_subset)
            
            while len(block) < k:
                best_gain = -1
                best_x = -1
                candidates_x = [x for x in range(v) if x not in block]
                block_t_minus_1 = list(itertools.combinations(block, t-1))
                
                for x in candidates_x:
                    gain = 0
                    for s_prime in block_t_minus_1:
                        t_sub = tuple(sorted(s_prime + (x,)))
                        i = subset_to_idx[t_sub]
                        if cov[i] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x != -1:
                        if random.random() < 0.5:
                            best_x = x
                
                if best_x == -1:
                    best_x = random.choice(candidates_x)
                block.append(best_x)
            
            block.sort()
            for t_sub in itertools.combinations(block, t):
                i = subset_to_idx[t_sub]
                cov[i] += 1
            blocks.append(block)
            
        return blocks, cov

    # Local Search: Remove Redundant + Swap
    def local_search(blocks, cov):
        changed = True
        while changed:
            changed = False
            # 1. Remove redundant
            for i in range(len(blocks)):
                block = blocks[i]
                is_essential = False
                for t_sub in itertools.combinations(block, t):
                    idx = subset_to_idx[t_sub]
                    if cov[idx] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    for t_sub in itertools.combinations(block, t):
                        idx = subset_to_idx[t_sub]
                        cov[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
            
            if not changed:
                # 2. Try Swaps
                if len(blocks) > 0:
                    # Pick a random block to swap
                    # We want to swap a block that has many critical subsets? 
                    # Or just random.
                    # Let's try a few random blocks
                    for _ in range(10): # Try a few swaps per iteration
                        if time.time() - start_time > time_limit:
                            return blocks, cov
                            
                        i = random.randrange(len(blocks))
                        B = blocks[i]
                        
                        # Find critical subsets of B
                        critical = []
                        for t_sub in itertools.combinations(B, t):
                            idx = subset_to_idx[t_sub]
                            if cov[idx] == 1:
                                critical.append(t_sub)
                                
                        if not critical:
                            # Should have been removed
                            continue
                            
                        # Try to find a replacement B' that covers all critical subsets
                        # Generate random B'
                        # To increase success, we can bias B' to include points from critical subsets
                        # But simple random might be okay.
                        
                        # Optimization: If critical is large, it's hard to cover.
                        # If critical is small (e.g., 1-2 subsets), easier.
                        
                        # Let's try 20 random candidates
                        for _ in range(20):
                            # Generate random k-subset
                            B_prime = random.sample(range(v), k)
                            B_prime.sort()
                            
                            # Check if B_prime is already in blocks
                            # This is O(N*K) which is fine for small N
                            if B_prime in blocks:
                                continue
                                
                            # Check if B_prime covers all critical subsets
                            if block_covers_subsets(B_prime, critical):
                                # Replacement is valid
                                # Update cov: remove B, add B_prime
                                for t_sub in itertools.combinations(B, t):
                                    idx = subset_to_idx[t_sub]
                                    cov[idx] -= 1
                                for t_sub in itertools.combinations(B_prime, t):
                                    idx = subset_to_idx[t_sub]
                                    cov[idx] += 1
                                
                                blocks[i] = B_prime
                                changed = True
                                break # Restart removal/swap loop
                        if changed:
                            break
                            
        return blocks, cov

    # Main Loop
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        
        random.seed()
        
        blocks, cov = greedy_construct()
        blocks, cov = local_search(blocks, cov)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    return best_blocks
