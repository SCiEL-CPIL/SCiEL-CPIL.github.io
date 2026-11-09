import math
import time

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 512
    
    # Build Incompatibility Graph
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            lcm = i * j // g
            if lcm > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Greedy Initial Solution
    # Sort by degree in incompat graph (ascending)
    degrees = [adj[i].bit_count() for i in range(1, N + 1)]
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: degrees[x])
    
    current_set = set()
    available = set(range(1, N + 1))
    for v in vertices:
        if v in available:
            current_set.add(v)
            # Remove v and incompatible neighbors
            # We can just iterate available
            for u in list(available):
                if u == v or (adj[v] >> u) & 1:
                    available.remove(u)
                    
    global_best_set = set(current_set)
    global_best_size = len(current_set)
    
    initial_U = 0
    for i in range(1, N + 1):
        initial_U |= (1 << i)
        
    def branch_and_bound(U, P, k):
        nonlocal global_best_set, global_best_size
        
        if time.time() - start_time > timeout:
            return
            
        if k > global_best_size:
            global_best_size = k
            global_best_set = set(P)
            
        if not U:
            return
            
        if k + U.bit_count() <= global_best_size:
            return
            
        # Select vertex with max degree in U
        max_deg = -1
        best_v = -1
        best_v_bit = 0
        u = U
        while u:
            v_bit = u & -u
            v = v_bit.bit_length() - 1
            deg = (adj[v] & U).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
                best_v_bit = v_bit
            u &= u - 1
            
        if best_v == -1:
            return
            
        # Branch 1: Include
        new_U_include = U & ~(adj[best_v] | best_v_bit)
        P.add(best_v)
        branch_and_bound(new_U_include, P, k + 1)
        P.remove(best_v)
        
        # Branch 2: Exclude
        new_U_exclude = U & ~best_v_bit
        branch_and_bound(new_U_exclude, P, k)

    branch_and_bound(initial_U, set(), 0)
    
    return list(global_best_set)
