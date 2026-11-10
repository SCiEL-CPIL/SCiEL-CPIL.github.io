import time
import random

def construct():
    deadline = time.time() + 85
    N = 256
    
    # Precompute conflict matrix using bitmasks for speed
    # adj[i] is a bitmask where bit j is set if i and j conflict
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    def can_add(element, current_mask):
        # Check if element is compatible with all elements in current_mask
        # adj[element] contains bits for conflicting elements
        return (adj[element] & current_mask) == 0

    def greedy_fill(S_mask, adj, N):
        # Add any element that can be added, iterating until stable
        improved = True
        while improved:
            improved = False
            for e in range(1, N + 1):
                if (S_mask >> e) & 1:
                    continue
                if can_add(e, S_mask):
                    S_mask |= (1 << e)
                    improved = True
        return S_mask

    def local_search(S_mask, adj, N, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = [i for i in range(1, N + 1) if (S_mask >> i) & 1]
            random.shuffle(S_list)
            
            # Try 1-for-1 swaps (remove one, add one)
            for remove_elem in S_list:
                if time.time() > deadline:
                    break
                S_without = S_mask & ~(1 << remove_elem)
                for add_elem in range(1, N + 1):
                    if (S_without >> add_elem) & 1:
                        continue
                    if can_add(add_elem, S_without):
                        new_S = S_without | (1 << add_elem)
                        # Try to greedily fill the new set
                        new_S = greedy_fill(new_S, adj, N)
                        if bin(new_S).count('1') > bin(S_mask).count('1'):
                            S_mask = new_S
                            improved = True
                            break
                if improved:
                    break
            
            # Try 1-for-2 swaps (remove one, add two)
            if not improved:
                for remove_elem in S_list:
                    if time.time() > deadline:
                        break
                    S_without = S_mask & ~(1 << remove_elem)
                    candidates = []
                    for add_elem in range(1, N + 1):
                        if (S_without >> add_elem) & 1:
                            continue
                        if can_add(add_elem, S_without):
                            candidates.append(add_elem)
                    
                    for i in range(len(candidates)):
                        if time.time() > deadline:
                            break
                        for j in range(i + 1, len(candidates)):
                            c1, c2 = candidates[i], candidates[j]
                            if (adj[c1] >> c2) & 1:
                                continue
                            new_S = S_without | (1 << c1) | (1 << c2)
                            new_S = greedy_fill(new_S, adj, N)
                            if bin(new_S).count('1') > bin(S_mask).count('1'):
                                S_mask = new_S
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try 1-for-3 swaps (remove one, add three)
            if not improved:
                for remove_elem in S_list:
                    if time.time() > deadline:
                        break
                    S_without = S_mask & ~(1 << remove_elem)
                    candidates = []
                    for add_elem in range(1, N + 1):
                        if (S_without >> add_elem) & 1:
                            continue
                        if can_add(add_elem, S_without):
                            candidates.append(add_elem)
                    
                    # Only try 1-for-3 if we have enough candidates
                    if len(candidates) < 3:
                        continue
                        
                    for i in range(len(candidates)):
                        if time.time() > deadline:
                            break
                        for j in range(i + 1, len(candidates)):
                            if time.time() > deadline:
                                break
                            c1, c2 = candidates[i], candidates[j]
                            if (adj[c1] >> c2) & 1:
                                continue
                            for k in range(j + 1, len(candidates)):
                                c3 = candidates[k]
                                if (adj[c1] >> c3) & 1 or (adj[c2] >> c3) & 1:
                                    continue
                                new_S = S_without | (1 << c1) | (1 << c2) | (1 << c3)
                                new_S = greedy_fill(new_S, adj, N)
                                if bin(new_S).count('1') > bin(S_mask).count('1'):
                                    S_mask = new_S
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        return S_mask

    best_mask = 0
    best_size = 0
    
    random.seed(42)
    
    # Multiple random restarts with greedy construction
    # We use a mix of strategies:
    # 1. Start with all odds (known valid base)
    # 2. Randomized greedy order
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
            
        S_mask = 0
        
        # Strategy A: Start with odds
        if attempt % 2 == 0:
            for e in range(1, N + 1, 2):
                S_mask |= (1 << e)
        else:
            # Strategy B: Randomized greedy
            elements = list(range(1, N + 1))
            random.shuffle(elements)
            for e in elements:
                if can_add(e, S_mask):
                    S_mask |= (1 << e)
        
        # Greedy fill
        S_mask = greedy_fill(S_mask, adj, N)
        
        # Local Search
        S_mask = local_search(S_mask, adj, N, deadline - 5)
        
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask
            
    # Final polish
    final_mask = greedy_fill(best_mask, adj, N)
    if bin(final_mask).count('1') > best_size:
        best_mask = final_mask
        best_size = bin(final_mask).count('1')
        
    result = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
    return result

construct()
