import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute known 35-vertex graph (Paley-like or known record)
    # We will try to find a valid 35-vertex graph first, then extend to 36.
    
    def is_valid(n, adj):
        # Fast K4 check
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Check if any edge exists among common neighbors
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj[x] & (common & ~(1 << x)):
                            return False
        # Fast I6 check using nested loops with early termination
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                for w in range(v + 1, n):
                    if (adj[u] >> w) & 1 or (adj[v] >> w) & 1:
                        continue
                    for x in range(w + 1, n):
                        if (adj[u] >> x) & 1 or (adj[v] >> x) & 1 or (adj[w] >> x) & 1:
                            continue
                        for y in range(x + 1, n):
                            if (adj[u] >> y) & 1 or (adj[v] >> y) & 1 or (adj[w] >> y) & 1 or (adj[x] >> y) & 1:
                                continue
                            for z in range(y + 1, n):
                                if (adj[u] >> z) & 1 or (adj[v] >> z) & 1 or (adj[w] >> z) & 1 or (adj[x] >> z) & 1 or (adj[y] >> z) & 1:
                                    continue
                                return False
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges_in_mask(mask, adj):
        """Count edges in induced subgraph on vertices in mask"""
        total_deg = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            total_deg += (adj[u] & mask).bit_count()
        return total_deg // 2

    def count_indep_4_in_mask(mask, adj):
        """Count independent sets of size 4 in induced subgraph on vertices in mask"""
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
        
        count = 0
        n_v = len(vertices)
        if n_v < 4:
            return 0
        
        for i in range(n_v):
            for j in range(i + 1, n_v):
                if (adj[vertices[i]] >> vertices[j]) & 1:
                    continue
                for k in range(j + 1, n_v):
                    if (adj[vertices[i]] >> vertices[k]) & 1 or (adj[vertices[j]] >> vertices[k]) & 1:
                        continue
                    for l in range(k + 1, n_v):
                        if (adj[vertices[i]] >> vertices[l]) & 1 or (adj[vertices[j]] >> vertices[l]) & 1 or (adj[vertices[k]] >> vertices[l]) & 1:
                            continue
                        count += 1
        return count

    def delta_violations(u, v, adj, n, is_edge):
        """
        Returns the change in total violations if edge (u,v) is flipped.
        """
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        
        non_common = ((1 << n) - 1) & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
        
        e_count = count_edges_in_mask(common, adj)
        i4_count = count_indep_4_in_mask(non_common, adj)
        
        if is_edge:
            return -e_count + i4_count
        else:
            return e_count - i4_count

    def local_search(adj, n, deadline, max_rounds=20):
        """Perform local search to minimize violations with time-bounded iterations"""
        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            best_delta = 0
            best_u = -1
            best_v = -1
            
            for u in range(n):
                for v in range(u + 1, n):
                    is_edge = bool((adj[u] >> v) & 1)
                    d = delta_violations(u, v, adj, n, is_edge)
                    if d < best_delta:
                        best_delta = d
                        best_u = u
                        best_v = v
                        improved = True
                        
            if best_u != -1:
                if (adj[best_u] >> best_v) & 1:
                    adj[best_u] &= ~(1 << best_v)
                    adj[best_v] &= ~(1 << best_u)
                else:
                    adj[best_u] |= (1 << best_v)
                    adj[best_v] |= (1 << best_u)
            else:
                break
        return adj

    best_graph = None
    best_n = 0

    # Strategy: 
    # 1. Try to find a valid 35-vertex graph quickly (using random init + local search).
    # 2. If found, try to extend it to 36 vertices by adding a vertex and local searching.
    # 3. If 36 is found, return it.
    # 4. If not, return the best 35-vertex graph found.
    # 5. If no 35-vertex graph is found, fall back to smaller n.

    # First, secure a valid 35-vertex graph
    found_35 = False
    adj_35 = None
    
    # Try a few random initializations for n=35
    for r in range(5):
        if time.time() > deadline:
            break
        adj = [0] * 35
        for u in range(35):
            for v in range(u + 1, 35):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        adj = local_search(adj, 35, deadline, max_rounds=5)
        
        if is_valid(35, adj):
            adj_35 = adj
            found_35 = True
            break

    if found_35:
        best_n = 35
        best_graph = adj_35
        
        # Try to extend to 36
        # Add a new vertex 35
        # Its adjacency to 0..34 is initially random
        for r in range(10):
            if time.time() > deadline:
                break
            
            adj_36 = adj_35[:] + [0]
            # Randomly connect vertex 35 to others
            for v in range(35):
                if random.random() < 0.5:
                    adj_36[v] |= (1 << 35)
                    adj_36[35] |= (1 << v)
            
            # Local search on the 36-vertex graph
            # To speed up, we can restrict search to edges involving the new vertex first?
            # Or just full local search.
            adj_36 = local_search(adj_36, 36, deadline, max_rounds=10)
            
            if is_valid(36, adj_36):
                best_n = 36
                best_graph = adj_36
                break

    if best_graph is None:
        # Fallback: Try to find a valid graph for smaller n
        for target_n in [34, 33, 32, 30, 25, 20, 17]:
            if time.time() > deadline:
                break
            found = False
            for r in range(5):
                if time.time() > deadline:
                    break
                adj = [0] * target_n
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if random.random() < 0.5:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                adj = local_search(adj, target_n, deadline, max_rounds=5)
                
                if is_valid(target_n, adj):
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break
            if found:
                break

    if best_graph is None:
        # Ultimate fallback: Paley 17
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(res['n'])
