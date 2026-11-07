import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 256
    
    # Precompute bad_with[v][w]
    # bad_with[v][w] is a bitset of u such that {v, u, w} is a bad triple.
    # A triple {a,b,c} is bad if a|bc or b|ac or c|ab.
    bad_with = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    bad_with[a][b] |= (1 << c)
                    bad_with[a][c] |= (1 << b)
                    bad_with[b][a] |= (1 << c)
                    bad_with[b][c] |= (1 << a)
                    bad_with[c][a] |= (1 << b)
                    bad_with[c][b] |= (1 << a)
                    
    initial_available = 0
    for i in range(2, N + 1):
        initial_available |= (1 << i)
        
    best_size = 0
    best_set_bits = 0
    
    def bit_count(x):
        return bin(x).count('1')
        
    def lowest_bit_index(x):
        if x == 0: return -1
        return (x & -x).bit_length() - 1
        
    def remove_lowest_bit(x):
        return x & (x - 1)
        
    # Greedy initialization
    def greedy_init(available, chosen):
        while available:
            v = lowest_bit_index(available)
            chosen |= (1 << v)
            available &= ~(1 << v)
            
            mask_to_remove = 0
            w_iter = chosen & ~(1 << v)
            while w_iter:
                w = lowest_bit_index(w_iter)
                mask_to_remove |= bad_with[v][w]
                w_iter = remove_lowest_bit(w_iter)
            
            available &= ~mask_to_remove
            
        return chosen

    initial_chosen = greedy_init(initial_available, 0)
    best_size = bit_count(initial_chosen)
    best_set_bits = initial_chosen
    
    # Branch and Bound
    def solve(available, chosen, depth):
        nonlocal best_size, best_set_bits, deadline
        
        if time.time() > deadline:
            return
            
        if bit_count(chosen) + bit_count(available) <= best_size:
            return
            
        if available == 0:
            if bit_count(chosen) > best_size:
                best_size = bit_count(chosen)
                best_set_bits = chosen
            return
            
        # Select vertex with max removal impact
        best_v = -1
        max_removal = -1
        a_iter = available
        while a_iter:
            v = lowest_bit_index(a_iter)
            a_iter = remove_lowest_bit(a_iter)
            
            mask_to_remove = 0
            w_iter = chosen
            while w_iter:
                w = lowest_bit_index(w_iter)
                mask_to_remove |= bad_with[v][w]
                w_iter = remove_lowest_bit(w_iter)
            
            removals = bit_count(available & mask_to_remove)
            if removals > max_removal:
                max_removal = removals
                best_v = v
                
        if best_v == -1:
            return
            
        v = best_v
        
        # Branch 1: Include v
        new_available = available & ~(1 << v)
        
        mask_to_remove = 0
        w_iter = chosen
        while w_iter:
            w = lowest_bit_index(w_iter)
            mask_to_remove |= bad_with[v][w]
            w_iter = remove_lowest_bit(w_iter)
            
        new_available &= ~mask_to_remove
        
        solve(new_available, chosen | (1 << v), depth + 1)
        
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude v
        new_available2 = available & ~(1 << v)
        solve(new_available2, chosen, depth + 1)
        
    solve(initial_available, 0, 0)
    
    result = []
    bits = best_set_bits
    while bits:
        idx = lowest_bit_index(bits)
        result.append(idx)
        bits = remove_lowest_bit(bits)
        
    return result
