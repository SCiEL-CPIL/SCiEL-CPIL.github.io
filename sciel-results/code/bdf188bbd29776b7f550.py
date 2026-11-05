import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Tries to find a valid graph with n=33 (or higher) using local search.
    Falls back to the known valid n=32 construction (8 disjoint K4s).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 35s buffer for safety/overhead

    # Fallback: 8 disjoint K4s, n=32
    n_fallback = 32
    edges_fallback = []
    for block_idx in range(8):
        start = block_idx * 4
        verts = [start, start + 1, start + 2, start + 3]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                edges_fallback.append([verts[i], verts[j]])

    # Try to find n=33
    n = 33

    def check_c5(adj, n):
        """
        Returns True if the graph is C5-free.
        adj[i] is an integer where bit j is 1 if i~j.
        Checks for any C5 subgraph.
        """
        # A C5 exists if there is a path u-a-b-c-v where u~v, and internal vertices are distinct and not u,v.
        # For each edge (u,v), check if there is a path v-a-b-c-u.
        # a in N(v)\{u}, c in N(u)\{v}, b in N(a) & N(c) \ {u,v,a,c}.
        
        for u in range(n):
            nu = adj[u]
            nv_bits = nu
            while nv_bits:
                lsb = nv_bits & -nv_bits
                v = lsb.bit_length() - 1
                nv_bits ^= lsb
                
                if u >= v:
                    continue
                
                s_a = adj[v] & ~(1 << u)
                s_c = adj[u] & ~(1 << v)
                
                a_bits = s_a
                while a_bits:
                    lsb_a = a_bits & -a_bits
                    a = lsb_a.bit_length() - 1
                    a_bits ^= lsb_a
                    
                    if a == u or a == v:
                        continue
                        
                    c_bits = s_c
                    while c_bits:
                        lsb_c = c_bits & -c_bits
                        c = lsb_c.bit_length() - 1
                        c_bits ^= lsb_c
                        
                        if a == c:
                            continue
                        
                        common = adj[a] & adj[c]
                        mask = (1 << u) | (1 << v) | (1 << a) | (1 << c)
                        if common & ~mask:
                            return False
        return True

    def has_independent_set_9_fast(adj, n):
        """
        Returns True if there is an independent set of size 9.
        Uses branch and bound with early termination.
        """
        def dfs(candidates, size):
            if size >= 9:
                return True
            if candidates == 0:
                return False
            
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            # Branch 1: Include v - new candidates must not be adjacent to v
            new_cands = candidates & ~adj[v]
            if dfs(new_cands, size + 1):
                return True
            
            # Branch 2: Exclude v
            if dfs(candidates, size):
                return False
            
            return False
        
        return dfs((1 << n) - 1, 0)

    def try_find_n33():
        rng = random.Random(42)
        
        # Initial setup: 8 K4s
        adj = [0] * n
        for block_idx in range(8):
            start = block_idx * 4
            for i in range(4):
                for j in range(i+1, 4):
                    u = start + i
                    v = start + j
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Reset adj to base
            adj = [0] * n
            for block_idx in range(8):
                start = block_idx * 4
                for i in range(4):
                    for j in range(i+1, 4):
                        u = start + i
                        v = start + j
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Randomly add edges from 32 to base
            for block_idx in range(8):
                start = block_idx * 4
                verts = [start, start+1, start+2, start+3]
                rng.shuffle(verts)
                k = rng.randint(2, 4)
                for v in verts[:k]:
                    adj[32] |= (1 << v)
                    adj[v] |= (1 << 32)
            
            num_inter_edges = rng.randint(0, 15)
            for _ in range(num_inter_edges):
                b1 = rng.randint(0, 7)
                b2 = rng.randint(0, 7)
                if b1 == b2:
                    continue
                u = b1 * 4 + rng.randint(0, 3)
                v = b2 * 4 + rng.randint(0, 3)
                if not (adj[u] & (1 << v)):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not check_c5(adj, n):
                continue
            
            if not has_independent_set_9_fast(adj, n):
                edges = []
                for u_idx in range(n):
                    v_bits = adj[u_idx]
                    while v_bits:
                        lsb = v_bits & -v_bits
                        v_idx = lsb.bit_length() - 1
                        v_bits ^= lsb
                        if u_idx < v_idx:
                            edges.append([u_idx, v_idx])
                return {'n': n, 'edges': edges}
            
            for step in range(800):
                if time.time() > deadline:
                    break
                
                if rng.random() < 0.6:
                    b1 = rng.randint(0, 7)
                    b2 = rng.randint(0, 7)
                    if b1 == b2:
                        continue
                    u = b1 * 4 + rng.randint(0, 3)
                    v = b2 * 4 + rng.randint(0, 3)
                    if u == 32 or v == 32:
                        continue
                
                    if adj[u] & (1 << v):
                        continue
                    
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    if not check_c5(adj, n):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        if not has_independent_set_9_fast(adj, n):
                            edges = []
                            for u_idx in range(n):
                                v_bits = adj[u_idx]
                                while v_bits:
                                    lsb = v_bits & -v_bits
                                    v_idx = lsb.bit_length() - 1
                                    v_bits ^= lsb
                                    if u_idx < v_idx:
                                        edges.append([u_idx, v_idx])
                            return {'n': n, 'edges': edges}
                else:
                    b1 = rng.randint(0, 7)
                    b2 = rng.randint(0, 7)
                    if b1 == b2:
                        continue
                    u = b1 * 4 + rng.randint(0, 3)
                    v = b2 * 4 + rng.randint(0, 3)
                    if u == 32 or v == 32:
                        continue
                    
                    if not (adj[u] & (1 << v)):
                        continue
                    
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    if not has_independent_set_9_fast(adj, n):
                        edges = []
                        for u_idx in range(n):
                            v_bits = adj[u_idx]
                            while v_bits:
                                lsb = v_bits & -v_bits
                                v_idx = lsb.bit_length() - 1
                                v_bits ^= lsb
                                if u_idx < v_idx:
                                    edges.append([u_idx, v_idx])
                        return {'n': n, 'edges': edges}
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return None

    try:
        result = try_find_n33()
        if result:
            return result
    except Exception:
        pass
        
    return {'n': n_fallback, 'edges': edges_fallback}

if __name__ == "__main__":
    r = construct()
    print(r['n'])
