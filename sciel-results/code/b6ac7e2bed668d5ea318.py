import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    N = 256
    nodes = list(range(2, N + 1))
    n_nodes = len(nodes)
    
    # Map value to index for easier bit manipulation? 
    # Actually, values are 2..256. We can use a bitmask of size 257.
    # Bit i represents if number i is in the set.
    
    # Precompute conflicts
    # conflict_mask[i] is a bitmask of nodes j such that lcm(i, j) <= 256
    conflict_mask = [0] * (N + 1)
    for i in nodes:
        for j in nodes:
            if i == j: continue
            g = gcd(i, j)
            if (i * j) // g <= 256:
                conflict_mask[i] |= (1 << j)
                
    all_nodes_mask = 0
    for i in nodes:
        all_nodes_mask |= (1 << i)
        
    # Precompute weights
    weights = [0.0] * (N + 1)
    for i in nodes:
        weights[i] = 1.0 / i
        
    def get_score(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    def is_valid_swap_add(mask, node, exclude_mask):
        # Check if node is compatible with all nodes in (mask & ~exclude_mask)
        # i.e., (mask & ~exclude_mask) & conflict_mask[node] == 0
        return ((mask & ~exclude_mask) & conflict_mask[node]) == 0

    def local_search(mask):
        current_score = get_score(mask)
        improved = True
        while improved and (time.time() - start_time < deadline):
            improved = False
            
            # 1-for-1 Swap
            # Remove a, add b. 1/b > 1/a. b compat with S\{a}
            # Iterate a in S, b not in S
            # Optimization: Only try if 1/b > 1/a
            
            # 1-for-2 Swap
            # Remove a, add b, c. 1/b + 1/c > 1/a. b, c compat with each other and S\{a}
            
            # 2-for-1 Swap
            # Remove a, b, add c. 1/c > 1/a + 1/b. c compat with S\{a,b}
            
            # 2-for-2 Swap
            # Remove a, b, add c, d. 1/c + 1/d > 1/a + 1/b. c,d compat with each other and S\{a,b}
            
            # Efficient implementation requires iterating subsets.
            # Given N=256, S is likely small (maybe 10-20 nodes).
            # Number of nodes in S is small.
            
            nodes_in_S = [i for i in nodes if (mask >> i) & 1]
            nodes_not_in_S = [i for i in nodes if not ((mask >> i) & 1)]
            
            # 1-for-1
            for a in nodes_in_S:
                for b in nodes_not_in_S:
                    if weights[b] <= weights[a]: continue
                    if is_valid_swap_add(mask, b, 1 << a):
                        new_mask = (mask & ~(1 << a)) | (1 << b)
                        new_score = get_score(new_mask)
                        if new_score > current_score + 1e-12:
                            mask = new_mask
                            current_score = new_score
                            improved = True
                            nodes_in_S = [i for i in nodes if (mask >> i) & 1]
                            nodes_not_in_S = [i for i in nodes if not ((mask >> i) & 1)]
                            break
                if improved: break
            if improved: continue
            
            # 1-for-2
            for a in nodes_in_S:
                # Find pairs b, c in not_in_S
                # b < c
                for i_idx in range(len(nodes_not_in_S)):
                    b = nodes_not_in_S[i_idx]
                    for c in nodes_not_in_S[i_idx+1:]:
                        if weights[b] + weights[c] <= weights[a]: continue
                        # Check b, c compat
                        if (conflict_mask[b] >> c) & 1: continue
                        # Check compat with S \ {a}
                        if is_valid_swap_add(mask, b, 1 << a) and is_valid_swap_add(mask, c, 1 << a):
                            new_mask = (mask & ~(1 << a)) | (1 << b) | (1 << c)
                            new_score = get_score(new_mask)
                            if new_score > current_score + 1e-12:
                                mask = new_mask
                                current_score = new_score
                                improved = True
                                nodes_in_S = [i for i in nodes if (mask >> i) & 1]
                                nodes_not_in_S = [i for i in nodes if not ((mask >> i) & 1)]
                                break
                    if improved: break
                if improved: break
            if improved: continue
            
            # 2-for-1
            for i_idx in range(len(nodes_in_S)):
                a = nodes_in_S[i_idx]
                for b in nodes_in_S[i_idx+1:]:
                    # Find c in not_in_S
                    for c in nodes_not_in_S:
                        if weights[c] <= weights[a] + weights[b]: continue
                        if is_valid_swap_add(mask, c, (1 << a) | (1 << b)):
                            new_mask = (mask & ~((1 << a) | (1 << b))) | (1 << c)
                            new_score = get_score(new_mask)
                            if new_score > current_score + 1e-12:
                                mask = new_mask
                                current_score = new_score
                                improved = True
                                nodes_in_S = [i for i in nodes if (mask >> i) & 1]
                                nodes_not_in_S = [i for i in nodes if not ((mask >> i) & 1)]
                                break
                    if improved: break
                if improved: break
            if improved: continue
            
            # 2-for-2 (Optional, might be slow if S is large, but S is small)
            # Skip for now to ensure speed, 1-for-2 is usually the key.
            
        return mask

    best_mask = 0
    best_score = 0.0
    
    # Initial Greedy
    # Sort by weight desc
    sorted_nodes = sorted(nodes, key=lambda x: -weights[x])
    for _ in range(20): # 20 restarts
        if time.time() - start_time > deadline: break
        
        # Randomized Greedy
        random.shuffle(sorted_nodes)
        mask = 0
        for n in sorted_nodes:
            if (mask & conflict_mask[n]) == 0:
                mask |= (1 << n)
        
        mask = local_search(mask)
        score = get_score(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    # Convert mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx)
        m ^= lsb
        
    return result
