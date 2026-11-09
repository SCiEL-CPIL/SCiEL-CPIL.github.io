import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def is_valid(n, adj):
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    c = common
                    while c:
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        if adj[k] & common:
                            return False
        # Check Ind6 (K6 in complement)
        full_mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
            
        def has_k_in_comp(target_size, mask):
            if target_size == 1:
                return mask != 0
            if target_size == 0:
                return True
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                next_mask = (mask ^ lsb) & comp_adj[v]
                if has_k_in_comp(target_size - 1, next_mask):
                    return True
            return False

        return not has_k_in_comp(6, full_mask)

    # Construct 32-vertex blow-up of Clebsch
    # Clebsch vertices: 0..15
    # Clebsch edges: Hamming distance 1 or 4
    clebsch_edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            xor = i ^ j
            # Popcount
            if bin(xor).count('1') == 1 or bin(xor).count('1') == 4:
                clebsch_edges.append((i, j))
                
    # Blow up with k=2
    # Vertex v in Clebsch becomes 2v and 2v+1 in new graph
    n = 32
    adj = [0] * n
    for u, v in clebsch_edges:
        # Connect all pairs between I_u and I_v
        for i in range(2):
            for j in range(2):
                nu = 2 * u + i
                nv = 2 * v + j
                adj[nu] |= (1 << nv)
                adj[nv] |= (1 << nu)
                
    best_n = n
    best_adj = adj
    
    # Try to extend to 33, 34, 35
    for target_n in range(33, 36):
        if time.time() > deadline - 10.0:
            break
        found = False
        # Use local search starting from best_adj
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            # Initialize by adding random vertices to best_adj
            adj = best_adj[:] + [0] * (target_n - best_n)
            # Randomize new edges
            for i in range(best_n, target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            # Randomize edges between old and new
            for i in range(best_n):
                for j in range(best_n, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            if is_valid(target_n, adj):
                found = True
                best_n = target_n
                best_adj = adj
                break
            
            # Local search
            for step in range(500):
                if time.time() > deadline - 2.0:
                    break
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v: continue
                # Flip
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                if is_valid(target_n, adj):
                    found = True
                    best_n = target_n
                    best_adj = adj
                    break
                # Revert
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            if found:
                break

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
