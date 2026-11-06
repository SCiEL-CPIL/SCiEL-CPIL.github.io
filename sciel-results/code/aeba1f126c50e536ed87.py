import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 511
    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512.0 - a)
        
    # Precompute coprime masks
    # mask[a] is an integer bitset where bit b is 1 if gcd(a, b) == 1
    # We only care about bits 1..511
    mask = [0] * (N + 1)
    for a in range(1, N + 1):
        m = 0
        for b in range(1, N + 1):
            if gcd(a, b) == 1:
                m |= (1 << b)
        mask[a] = m
        
    # Greedy initialization
    # Sort candidates by weight descending (which is a ascending? No, w[a] = 1/(512-a), so larger a -> larger w)
    # So iterate a from N down to 1
    chosen = []
    chosen_mask = 0
    for a in range(N, 0, -1):
        # Check if a is coprime to all in chosen
        # (mask[a] & chosen_mask) == chosen_mask
        if (mask[a] & chosen_mask) == chosen_mask:
            chosen.append(a)
            chosen_mask |= (1 << a)
            
    best_score = sum(w[a] for a in chosen)
    best_chosen = list(chosen)
    best_mask = chosen_mask
    
    def get_score(chosen_list):
        return sum(w[a] for a in chosen_list)

    # Local Search
    # We will try to improve by swapping.
    # To speed up, we can precompute lists of candidates.
    
    # Random seed for reproducibility if needed, but we want to explore
    random.seed(42)
    
    # Helper to check if a set of new nodes is coprime to a set of old nodes
    # new_nodes: list, old_mask: int
    # We need new_nodes to be pairwise coprime and coprime to old_mask
    
    def can_add_set(new_nodes, old_mask):
        # Check pairwise coprime in new_nodes
        new_mask = 0
        for n in new_nodes:
            if new_mask and (mask[n] & new_mask) != new_mask:
                # This check is slightly wrong. mask[n] has 1 where coprime.
                # We need (mask[n] & new_mask) == new_mask
                if (mask[n] & new_mask) != new_mask:
                    return False
            new_mask |= (1 << n)
        # Check coprime to old_mask
        # For each n in new_nodes, (mask[n] & old_mask) == old_mask
        for n in new_nodes:
            if (mask[n] & old_mask) != old_mask:
                return False
        return True

    # Main search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try 1-for-1 swaps
        for i in range(len(best_chosen)):
            u = best_chosen[i]
            u_mask = (1 << u)
            rest_mask = best_mask ^ u_mask
            
            # Find best v not in best_chosen such that v is coprime to rest_mask
            # and w[v] > w[u]
            best_v = -1
            best_v_w = -1.0
            # Iterate through all possible v
            # Optimization: only consider v with w[v] > w[u]
            for v in range(1, N + 1):
                if v == u:
                    continue
                if best_mask & (1 << v):
                    continue
                if w[v] <= w[u] + 1e-9:
                    continue
                # Check if v is coprime to rest_mask
                if (mask[v] & rest_mask) == rest_mask:
                    if w[v] > best_v_w:
                        best_v_w = w[v]
                        best_v = v
            
            if best_v != -1:
                # Perform swap
                new_chosen = list(best_chosen)
                new_chosen[i] = best_v
                new_mask = best_mask ^ u_mask | (1 << best_v)
                new_score = best_score - w[u] + w[best_v]
                if new_score > best_score + 1e-9:
                    best_chosen = new_chosen
                    best_mask = new_mask
                    best_score = new_score
                    improved = True
                    # Restart search as state changed significantly
                    break
        
        if time.time() > deadline:
            break
            
        if improved:
            continue
            
        # Try 1-for-2 swaps
        # Remove u, add v, w
        for i in range(len(best_chosen)):
            u = best_chosen[i]
            u_mask = (1 << u)
            rest_mask = best_mask ^ u_mask
            
            # We need to find v, w such that:
            # 1. v, w not in best_chosen
            # 2. v, w coprime to rest_mask
            # 3. v, w coprime to each other
            # 4. w[v] + w[w] > w[u]
            
            # To make this fast, precompute candidates coprime to rest_mask
            # candidates = [x for x in range(1, N+1) if not (best_mask & (1<<x)) and (mask[x] & rest_mask) == rest_mask]
            # This list might be small.
            candidates = []
            for x in range(1, N + 1):
                if best_mask & (1 << x):
                    continue
                if (mask[x] & rest_mask) == rest_mask:
                    candidates.append(x)
                    
            # Now find pair v, w in candidates such that gcd(v,w)==1 and w[v]+w[w] > w[u]
            # Sort candidates by weight descending
            candidates.sort(key=lambda x: w[x], reverse=True)
            
            found_pair = False
            for idx_v in range(len(candidates)):
                v = candidates[idx_v]
                if w[v] < w[u]: # Even with another 0, can't beat if w[v] is too small? No, w is positive.
                    # Actually if w[v] + max_w < w[u], break
                    if w[v] + (w[candidates[0]] if candidates else 0) <= w[u]:
                        break
                for idx_w in range(idx_v + 1, len(candidates)):
                    w_node = candidates[idx_w]
                    if w[v] + w[w_node] <= w[u] + 1e-9:
                        break # Since sorted, further w_node will be smaller
                    if (mask[v] & (1 << w_node)) == (1 << w_node): # gcd(v, w_node) == 1
                        # Perform swap
                        new_chosen = list(best_chosen)
                        new_chosen.pop(i)
                        new_chosen.append(v)
                        new_chosen.append(w_node)
                        new_mask = rest_mask | (1 << v) | (1 << w_node)
                        new_score = best_score - w[u] + w[v] + w[w_node]
                        if new_score > best_score + 1e-9:
                            best_chosen = new_chosen
                            best_mask = new_mask
                            best_score = new_score
                            improved = True
                            found_pair = True
                            break
                if found_pair:
                    break
            
            if found_pair:
                break # Restart main loop
        
        if time.time() > deadline:
            break

    return best_chosen
