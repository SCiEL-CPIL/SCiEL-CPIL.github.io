import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def is_valid(n, adj):
        # Rigorous K4 check: for each edge (u,v), check if any common neighbor x has an edge to another common neighbor y
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
                        # Check if x is connected to any other vertex in common
                        if adj[x] & (common & ~(1 << x)):
                            return False
        # Rigorous I6 check
        for combo in combinations(range(n), 6):
            is_indep = True
            for i in range(6):
                for j in range(i + 1, 6):
                    u, v = combo[i], combo[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
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

    def make_paley_graph(p):
        """Create a Paley graph of prime order p (p ≡ 1 mod 4)"""
        adj = [0] * p
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(adj, n, deadline):
        """Perform local search to minimize violations"""
        improved = True
        while improved and time.time() < deadline:
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
        return adj

    best_graph = None
    best_n = 0

    # Strategy 1: Try to extend a valid 35-vertex graph to 36
    # First, find a valid 35-vertex graph
    for target_n in [36, 35]:
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 15
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            if target_n == 36:
                # Start from a 35-vertex Paley-like structure and add a vertex
                # Use Paley 29 as base and try to extend
                if r == 0:
                    # Use a structured initialization: Paley 29 + 7 more vertices
                    base_p = 29
                    adj = make_paley_graph(base_p)
                    # Extend to 36 by adding vertices with random connections
                    adj = adj + [0] * (target_n - base_p)
                    for u in range(base_p, target_n):
                        for v in range(target_n):
                            if v != u and random.random() < 0.45:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                else:
                    # Random initialization
                    adj = [0] * target_n
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if random.random() < 0.5:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
            else:
                # For n=35, use Paley 29 + extension or random
                if r == 0:
                    base_p = 29
                    adj = make_paley_graph(base_p)
                    adj = adj + [0] * (target_n - base_p)
                    for u in range(base_p, target_n):
                        for v in range(target_n):
                            if v != u and random.random() < 0.45:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                elif r == 1:
                    # Try Paley 13 + Paley 13 + 9
                    adj = [0] * target_n
                    # Create blocks
                    for block in range(2):
                        start = block * 13
                        paley13 = make_paley_graph(13)
                        for i in range(13):
                            for j in range(13):
                                if i != j and (paley13[i] >> j) & 1:
                                    adj[start + i] |= (1 << (start + j))
                    # Connect blocks and remaining vertices
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if (u < 13 and v >= 26) or (u >= 13 and u < 26 and v >= 26) or (u >= 26 and v >= 26):
                                if random.random() < 0.5:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                            elif (u < 13 and 13 <= v < 26) or (13 <= u < 26 and v >= 26):
                                if random.random() < 0.4:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                else:
                    adj = [0] * target_n
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if random.random() < 0.5:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
            
            # Local search
            adj = local_search(adj, target_n, deadline)
            
            if is_valid(target_n, adj):
                best_graph = adj
                best_n = target_n
                found = True
                break
        
        if found:
            break

    # Strategy 2: If we didn't find n>=35, try smaller n with more robust search
    if best_n < 35:
        for target_n in range(34, 16, -1):
            if time.time() > deadline:
                break
            
            found = False
            max_restarts = 10
            for r in range(max_restarts):
                if time.time() > deadline:
                    break
                
                adj = [0] * target_n
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if random.random() < 0.5:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                adj = local_search(adj, target_n, deadline)
                
                if is_valid(target_n, adj):
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break
            
            if found:
                break

    if best_graph is None:
        # Fallback to Paley 17
        adj = make_paley_graph(17)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
