import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    N = 256
    # Squares up to 2*256 = 512
    squares = set()
    for k in range(1, int(math.isqrt(2 * N)) + 1):
        squares.add(k * k)
        
    # Valid vertices: 2*v not in squares
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    M = len(valid_vertices)
    
    # Precompute conflicts for each v in valid_vertices
    # conflicts[v] is a bitmask of u in valid_vertices such that u+v is square
    conflicts = {v: 0 for v in valid_vertices}
    
    for i in range(len(valid_vertices)):
        u = valid_vertices[i]
        for j in range(i + 1, len(valid_vertices)):
            v = valid_vertices[j]
            if u + v in squares:
                conflicts[u] |= (1 << v)
                conflicts[v] |= (1 << u)
                
    def greedy_init(order):
        S = 0
        for v in order:
            if (S & conflicts[v]) == 0:
                S |= (1 << v)
        return S
        
    def local_search(S, time_limit):
        current_size = bin(S).count('1')
        best_S = S
        best_size = current_size
        
        v_list = valid_vertices[:]
        
        while time.time() < time_limit:
            # Try to add any valid vertex
            added = False
            # Shuffle to break symmetry
            random.shuffle(v_list)
            for v in v_list:
                if not (S & (1 << v)):
                    if (S & conflicts[v]) == 0:
                        S |= (1 << v)
                        current_size += 1
                        added = True
                        # Continue adding in this pass? 
                        # If we add one, others might become invalid.
                        # But if we add a valid one, it doesn't invalidate others that were already valid?
                        # No, adding v adds conflicts[v] to the "occupied" set.
                        # So other candidates might become invalid.
                        # So we should break and restart the outer loop to re-evaluate all.
                        # Or, we can just break and let the next iteration handle it.
                        break
            
            if not added:
                # Try a kick: remove a random vertex, then greedily add
                if current_size > 0:
                    # Pick a random vertex from S
                    bits = []
                    temp = S
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        temp ^= lsb
                    
                    if bits:
                        u = random.choice(bits)
                        S_new = S & ~(1 << u)
                        size_new = current_size - 1
                        
                        # Greedily add from S_new
                        random.shuffle(v_list)
                        for v in v_list:
                            if not (S_new & (1 << v)):
                                if (S_new & conflicts[v]) == 0:
                                    S_new |= (1 << v)
                                    size_new += 1
                                    
                        if size_new > best_size:
                            best_S = S_new
                            best_size = size_new
                            S = S_new
                            current_size = size_new
                        elif size_new == best_size:
                            # Accept with small probability to escape local opt
                            if random.random() < 0.1:
                                S = S_new
                                current_size = size_new
                        else:
                            # Even if worse, move to S_new to explore?
                            # Yes, simulated annealing style
                            S = S_new
                            current_size = size_new
            # If added, loop continues to try adding more
            # If kicked, loop continues
            
        return best_S

    best_global_S = 0
    best_global_size = 0
    
    # Initial greedy with natural order
    S0 = greedy_init(valid_vertices)
    best_global_S = S0
    best_global_size = bin(S0).count('1')
    
    # Local search on S0
    S_ls = local_search(S0, deadline - 5.0)
    size_ls = bin(S_ls).count('1')
    if size_ls > best_global_size:
        best_global_S = S_ls
        best_global_size = size_ls
        
    # Random restarts
    while time.time() < deadline - 2.0:
        order = valid_vertices[:]
        random.shuffle(order)
        S_r = greedy_init(order)
        size_r = bin(S_r).count('1')
        
        t_start = time.time()
        S_ls_r = local_search(S_r, t_start + 2.0)
        size_ls_r = bin(S_ls_r).count('1')
        
        if size_ls_r > best_global_size:
            best_global_S = S_ls_r
            best_global_size = size_ls_r
            
    result = []
    temp = best_global_S
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(idx)
        temp ^= lsb
        
    return result
