import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Fallback: Paley graph on 37
    def get_paley():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = 37
    best_edges = get_paley()[1]
    
    n = 42
    full_mask = (1 << n) - 1
    
    def count_triangles(mask, adj):
        # Fast triangle count in induced subgraph
        if mask.bit_count() < 3:
            return 0
        
        count = 0
        # Iterate vertices in mask
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            
            # Neighbors of u in mask
            neigh = mask & adj[u]
            
            # Count edges in neigh
            # Each edge (v, w) in neigh forms a triangle with u
            # We count each edge once by ensuring v < w?
            # Or just count bits in (neigh & adj[v]) and divide by 2?
            # Let's just count edges in the subgraph induced by neigh
            e_count = 0
            m2 = neigh
            while m2:
                lsb2 = m2 & -m2
                v = lsb2.bit_length() - 1
                m2 ^= lsb2
                # Count neighbors of v in neigh
                # This counts edge (v,w) and (w,v)
                e_count += (neigh & adj[v]).bit_count()
            
            # e_count is 2 * (#edges in neigh)
            # Each triangle is counted 3 times (once for each vertex u)
            # So total triangles = sum(e_count/2) / 3
            count += e_count // 2
            
        return count // 3

    def get_cadj(adj):
        return [full_mask ^ adj[u] ^ (1 << u) for u in range(n)]

    def evaluate_cost(adj):
        c5 = 0
        i5 = 0
        for combo in combinations(range(n), 5):
            u0, u1, u2, u3, u4 = combo
            is_clique = True
            is_indep = True
            if not (adj[u0] >> u1) & 1: is_clique = False
            if is_clique and not (adj[u0] >> u2) & 1: is_clique = False
            if is_clique and not (adj[u0] >> u3) & 1: is_clique = False
            if is_clique and not (adj[u0] >> u4) & 1: is_clique = False
            if is_clique and not (adj[u1] >> u2) & 1: is_clique = False
            if is_clique and not (adj[u1] >> u3) & 1: is_clique = False
            if is_clique and not (adj[u1] >> u4) & 1: is_clique = False
            if is_clique and not (adj[u2] >> u3) & 1: is_clique = False
            if is_clique and not (adj[u2] >> u4) & 1: is_clique = False
            if is_clique and not (adj[u3] >> u4) & 1: is_clique = False
            if is_clique: c5 += 1
            
            if (adj[u0] >> u1) & 1: is_indep = False
            if is_indep and (adj[u0] >> u2) & 1: is_indep = False
            if is_indep and (adj[u0] >> u3) & 1: is_indep = False
            if is_indep and (adj[u0] >> u4) & 1: is_indep = False
            if is_indep and (adj[u1] >> u2) & 1: is_indep = False
            if is_indep and (adj[u1] >> u3) & 1: is_indep = False
            if is_indep and (adj[u1] >> u4) & 1: is_indep = False
            if is_indep and (adj[u2] >> u3) & 1: is_indep = False
            if is_indep and (adj[u2] >> u4) & 1: is_indep = False
            if is_indep and (adj[u3] >> u4) & 1: is_indep = False
            if is_indep: i5 += 1
        return c5 + i5

    def delta_cost_add_edge(u, v, adj, cadj):
        common_neigh = adj[u] & adj[v]
        new_c5 = count_triangles(common_neigh, adj)
        
        common_non_neigh = cadj[u] & cadj[v]
        destroyed_i5 = count_triangles(common_non_neigh, cadj)
        
        return new_c5 - destroyed_i5

    def delta_cost_remove_edge(u, v, adj, cadj):
        common_neigh = adj[u] & adj[v]
        destroyed_c5 = count_triangles(common_neigh, adj)
        
        common_non_neigh = cadj[u] & cadj[v]
        new_i5 = count_triangles(common_non_neigh, cadj)
        
        return -destroyed_c5 + new_i5

    # Initialize
    paley_n, paley_edges = get_paley()
    adj = [0] * n
    for u, v in paley_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    random.seed(42)
    for i in range(37, n):
        for j in range(37):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    cadj = get_cadj(adj)
    current_cost = evaluate_cost(adj)
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        is_edge = bool((adj[u] >> v) & 1)
        
        if is_edge:
            delta = delta_cost_remove_edge(u, v, adj, cadj)
        else:
            delta = delta_cost_add_edge(u, v, adj, cadj)
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            if is_edge:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            cadj[u] = full_mask ^ adj[u] ^ (1 << u)
            cadj[v] = full_mask ^ adj[v] ^ (1 << v)
            
            current_cost += delta
            
            if current_cost == 0:
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                return n, edges
        
        T *= cooling

    return best_n, best_edges
