import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > deadline

    def is_valid_bitset(n, adj):
        # adj is a list of ints, bit i set if edge (v, i) exists
        # Check triangles
        for u in range(n):
            mask_u = adj[u]
            # Iterate over neighbors of u
            v_mask = mask_u
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if u < v:
                    # Check common neighbors
                    common = mask_u & adj[v]
                    if common:
                        return False
        
        # Check independent set of size 10
        # Equivalent to clique of size 10 in complement
        # Complement adjacency: ~adj[u] & ((1<<n)-1) & ~(1<<u)
        # Let's just search for independent set in G
        
        def find_independent_set(size, current_mask, avail_mask):
            # current_mask: vertices in the set
            # avail_mask: vertices that can be added (independent from current)
            
            if current_mask.bit_count() == size:
                return True
            if avail_mask == 0:
                return False
            if current_mask.bit_count() + avail_mask.bit_count() < size:
                return False
            
            # Heuristic: pick vertex with max degree in avail
            # Or min?
            # For independent set, picking a vertex removes its neighbors from avail.
            # Removing a vertex with many neighbors in avail shrinks avail more.
            # So pick vertex with max degree in avail.
            
            # To find max degree, we need to check all v in avail
            # This is O(n) per step.
            
            best_v = -1
            max_deg = -1
            m = avail_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                # Degree of v in avail: popcount(adj[v] & avail_mask)
                deg = (adj[v] & avail_mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            v_bit = 1 << best_v
            
            # Branch 1: Include best_v
            # New avail: avail_mask minus best_v minus neighbors of best_v
            new_avail = avail_mask & ~v_bit & ~adj[best_v]
            if find_independent_set(size, current_mask | v_bit, new_avail):
                return True
            
            # Branch 2: Exclude best_v
            new_avail = avail_mask & ~v_bit
            if find_independent_set(size, current_mask, new_avail):
                return True
            
            return False

        if n >= 10:
            if find_independent_set(10, 0, (1<<n)-1):
                return False
        return True

    def local_search(n, time_limit):
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        best_cost = float('inf')
        best_adj = None
        current_cost = float('inf')
        
        def get_cost(adj_list):
            tri_count = 0
            for u in range(n):
                mask_u = adj_list[u]
                v_mask = mask_u
                while v_mask:
                    lsb = v_mask & -v_mask
                    v = lsb.bit_length() - 1
                    v_mask ^= lsb
                    if u < v:
                        tri_count += (mask_u & adj_list[v]).bit_count()
            
            # Estimate alpha (greedy)
            avail = (1 << n) - 1
            alpha_est = 0
            while avail:
                # Pick vertex with min degree in avail
                min_deg = float('inf')
                v_pick = -1
                m = avail
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    deg = (adj_list[v] & avail).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        v_pick = v
                if v_pick == -1: break
                alpha_est += 1
                v_bit = 1 << v_pick
                avail &= ~v_bit
                avail &= ~adj_list[v_pick]
            
            return 10000 * tri_count + alpha_est

        current_cost = get_cost(adj)
        if current_cost == 0 and is_valid_bitset(n, adj):
            return adj
        
        while elapsed() < time_limit:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            
            u_bit = 1 << u
            v_bit = 1 << v
            
            if adj[u] & v_bit:
                adj[u] &= ~v_bit
                adj[v] &= ~u_bit
            else:
                adj[u] |= v_bit
                adj[v] |= u_bit
            
            new_cost = get_cost(adj)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
                    if best_cost == 0 and is_valid_bitset(n, best_adj):
                        return best_adj
            else:
                # Revert
                if adj[u] & v_bit:
                    adj[u] &= ~v_bit
                    adj[v] &= ~u_bit
                else:
                    adj[u] |= v_bit
                    adj[v] |= u_bit
                    
            if check_deadline():
                break
        
        return best_adj if best_cost < float('inf') else None

    for n in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28]:
        if check_deadline():
            break
            
        time_for_n = min(10, deadline - elapsed())
        if time_for_n <= 0:
            break
            
        res = local_search(n, start_time + time_for_n)
        if res:
            if is_valid_bitset(n, res):
                edges = []
                for u in range(n):
                    mask = res[u]
                    v_mask = mask
                    while v_mask:
                        lsb = v_mask & -v_mask
                        v = lsb.bit_length() - 1
                        v_mask ^= lsb
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
        
        if n == 28:
            return {'n': 28, 'edges': [[i, j] for i in range(28) for j in range(i + 1, 28) if (j - i) % 28 % 3 == 1]}
            
    return {'n': 28, 'edges': [[i, j] for i in range(28) for j in range(i + 1, 28) if (j - i) % 28 % 3 == 1]}
