import time
import random
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    triples_with = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcds[a][b]
            for c in range(b + 1, N + 1):
                gbc = gcds[b][c]
                if gab == gbc:
                    gca = gcds[c][a]
                    if gca == gab:
                        bad_triples.append((a, b, c))
                        # Store pairs for each element
                        triples_with[a].append((b, c))
                        triples_with[b].append((a, c))
                        triples_with[c].append((a, b))

    start_time = time.time()
    deadline = start_time + 55
    random.seed(42)
    
    def is_valid_add(S_mask, x):
        """
        Check if adding x to S (represented by S_mask) creates a bad triple.
        Assumes S is currently valid.
        Only need to check triples containing x.
        """
        for y, z in triples_with[x]:
            if (S_mask & (1 << y)) and (S_mask & (1 << z)):
                return False
        return True

    def greedy_build(order):
        S_mask = 0
        for x in order:
            if is_valid_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def tabu_search(S_mask, max_iter=10000):
        current_size = bin(S_mask).count("1")
        best_size = current_size
        best_mask = S_mask
        
        # Tabu lists: store (element, time_added/removed)
        # We use a simple list of (element, iteration)
        tabu_add = []  # Elements that were just added, cannot be removed
        tabu_remove = [] # Elements that were just removed, cannot be added
        tabu_len = 10
        
        not_in_S = [x for x in range(1, N + 1) if not (S_mask & (1 << x))]
        in_S = [x for x in range(1, N + 1) if (S_mask & (1 << x))]
        
        iter_count = 0
        stall_count = 0
        
        while time.time() < deadline and iter_count < max_iter:
            iter_count += 1
            improved = False
            
            # Clean up tabu lists
            if iter_count > 100:
                tabu_add = [(e, t) for e, t in tabu_add if iter_count - t < tabu_len]
                tabu_remove = [(e, t) for e, t in tabu_remove if iter_count - t < tabu_len]
            
            # Phase 1: Try to add
            # Shuffle not_in_S for randomness
            random.shuffle(not_in_S)
            for x in not_in_S:
                if x in [e for e, _ in tabu_remove]:
                    continue
                if is_valid_add(S_mask, x):
                    S_mask |= (1 << x)
                    current_size += 1
                    in_S.append(x)
                    not_in_S.remove(x)
                    tabu_add.append((x, iter_count))
                    improved = True
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = S_mask
                    break
            
            if improved:
                stall_count = 0
                continue
            
            # Phase 2: Try to swap
            # If we can't add, try swapping
            # Pick a random element to remove from in_S
            if not in_S:
                break
                
            # Bias towards removing elements that block many others?
            # For now, random
            random.shuffle(in_S)
            swap_done = False
            for u in in_S:
                if u in [e for e, _ in tabu_add]:
                    continue
                
                S_temp = S_mask & ~(1 << u)
                
                # Try to find a valid v to add to S_temp
                # Prioritize v that was blocked by u?
                # Just shuffle not_in_S
                random.shuffle(not_in_S)
                for v in not_in_S:
                    if v in [e for e, _ in tabu_remove]:
                        continue
                    if is_valid_add(S_temp, v):
                        # Perform swap
                        S_mask = S_temp | (1 << v)
                        in_S.remove(u)
                        in_S.append(v)
                        not_in_S.remove(v)
                        not_in_S.append(u)
                        
                        tabu_add.append((v, iter_count))
                        tabu_remove.append((u, iter_count))
                        
                        swap_done = True
                        break
                
                if swap_done:
                    break
            
            if not swap_done:
                stall_count += 1
                if stall_count > 50:
                    # Perturb: Remove 2 random elements
                    if len(in_S) > 2:
                        u1, u2 = random.sample(in_S, 2)
                        S_mask &= ~(1 << u1) & ~(1 << u2)
                        current_size -= 2
                        in_S.remove(u1)
                        in_S.remove(u2)
                        not_in_S.extend([u1, u2])
                        tabu_remove.extend([(u1, iter_count), (u2, iter_count)])
                        stall_count = 0
                    else:
                        break

        return best_mask, best_size

    best_set_mask = 0
    best_size = 0
    
    # Conflict counts for greedy ordering
    conflict_counts = [len(triples_with[i]) for i in range(N + 1)]
    
    # Multiple restarts
    orders = []
    orders.append(sorted(range(1, N + 1), key=lambda x: conflict_counts[x]))
    orders.append(sorted(range(1, N + 1), key=lambda x: -conflict_counts[x]))
    orders.append(list(range(1, N + 1)))
    random.shuffle(orders[2])
    
    for run in range(20):
        if time.time() > deadline - 5:
            break
            
        if run < len(orders):
            order = orders[run]
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
        S_mask = greedy_build(order)
        S_mask, size = tabu_search(S_mask)
        
        if size > best_size:
            best_size = size
            best_set_mask = S_mask

    result = [x for x in range(1, N + 1) if (best_set_mask & (1 << x))]
    
    if not result:
        return [1]
        
    return result
