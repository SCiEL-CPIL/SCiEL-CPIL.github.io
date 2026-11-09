import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize with Paley(17) graph
    n_paley = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * n_paley
    for i in range(n_paley):
        for j in range(i + 1, n_paley):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_n = n_paley
    best_adj = adj.copy()
    
    # Try to extend up to n=40
    for target_n in range(n_paley + 1, 41):
        if time.time() > deadline:
            break
            
        # Time limit for this specific extension
        target_deadline = time.time() + 2.0
        
        success = False
        # Try multiple random neighbor sets
        for attempt in range(1000):
            if time.time() > target_deadline or time.time() > deadline:
                break
                
            # Generate random neighbor mask S
            # S is subset of {0, ..., target_n-1}
            S_mask = random.getrandbits(target_n)
            
            # Check 1: Triangle in G[S]
            if has_triangle_in_mask(adj, S_mask):
                continue
                
            # Check 2: Ind Set 6 in G[U] where U = {0..target_n-1} \ S
            all_mask = (1 << target_n) - 1
            U_mask = all_mask ^ S_mask
            
            if has_ind_set_size_k(adj, U_mask, 6):
                continue
                
            # Valid!
            # Add new vertex
            adj.append(S_mask)
            success = True
            best_n = target_n
            best_adj = adj.copy()
            break
            
        if not success:
            # Could not extend to target_n
            break
            
    # Convert best_adj to edges
    edges = []
    n = best_n
    for i in range(n):
        mask = best_adj[i]
        for j in range(i + 1, n):
            if mask & (1 << j):
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}

def has_triangle_in_mask(adj_list, S_mask):
    S = S_mask
    while S:
        u_bit = S & -S
        u = u_bit.bit_length() - 1
        S ^= u_bit
        
        candidates = adj_list[u] & S_mask
        candidates &= ~((1 << (u + 1)) - 1)
        
        while candidates:
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            candidates ^= v_bit
            
            common = adj_list[u] & adj_list[v] & S_mask
            if common:
                return True
    return False

def greedy_ind_set(adj_list, U_mask, k):
    count = 0
    current_U = U_mask
    while current_U and count < k:
        bit = current_U & -current_U
        v = bit.bit_length() - 1
        count += 1
        current_U &= ~(adj_list[v] | (1 << v))
        
    return count >= k

def has_ind_set_size_k(adj_list, U_mask, k):
    if k == 0:
        return True
    if U_mask.bit_count() < k:
        return False
    if k == 1:
        return U_mask != 0
    
    if greedy_ind_set(adj_list, U_mask, k):
        return True
        
    max_deg = -1
    v = -1
    temp = U_mask
    while temp:
        bit = temp & -temp
        idx = bit.bit_length() - 1
        temp ^= bit
        deg = (adj_list[idx] & U_mask).bit_count()
        if deg > max_deg:
            max_deg = deg
            v = idx
            
    if v == -1:
        return False
        
    new_U_include = U_mask & ~(adj_list[v] | (1 << v))
    if has_ind_set_size_k(adj_list, new_U_include, k - 1):
        return True
        
    new_U_exclude = U_mask & ~(1 << v)
    if has_ind_set_size_k(adj_list, new_U_exclude, k):
        return True
        
    return False
