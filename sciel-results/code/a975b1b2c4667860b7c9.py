"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Build conflict graph using bitsets
    adj = [0] * n
    
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_size = 0
    best_set = []
    
    # Initial solution: all odd numbers
    initial_mask = 0
    for i in range(n):
        if (i + 1) % 2 == 1:
            initial_mask |= (1 << i)
            best_size += 1
    best_set = []
    temp = initial_mask
    while temp:
        lsb = temp & (-temp)
        idx = lsb.bit_length() - 1
        best_set.append(idx + 1)
        temp ^= lsb
    
    def popcount(x):
        count = 0
        while x:
            count += 1
            x &= x - 1
        return count
    
    def solve(candidates, current_size, current_set_mask):
        nonlocal best_size, best_set
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                temp = current_set_mask
                s = []
                while temp:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    s.append(idx + 1)
                    temp ^= lsb
                best_set = s
            return
        
        if current_size + popcount(candidates) <= best_size:
            return
        
        # Pick vertex with max degree in candidate subgraph
        cand_list = []
        temp = candidates
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            cand_list.append(idx)
            temp ^= lsb
        
        if not cand_list:
            return
        
        best_v = cand_list[0]
        max_deg = -1
        for v in cand_list:
            deg = popcount(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        v = best_v
        v_bit = 1 << v
        
        # Branch 1: include v
        new_cand = candidates & ~v_bit & ~adj[v]
        new_mask = current_set_mask | v_bit
        solve(new_cand, current_size + 1, new_mask)
        
        # Branch 2: exclude v
        new_cand2 = candidates & ~v_bit
        solve(new_cand2, current_size, current_set_mask)
    
    all_cand = (1 << n) - 1
    solve(all_cand, 0, 0)
    
    return best_set

# EVOLVE-BLOCK-END
