import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for the whole graph
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        # Use bitmask representation for clique part edges
        # edge_mask[u] is a bitmask where bit v is set if edge (min(u,v), max(u,v)) exists
        edge_mask = [0] * k
        for u in range(k):
            for v in range(u + 1, k):
                edge_mask[u] |= (1 << v)
                edge_mask[v] |= (1 << u)
        
        partition = []
        
        # Process independent set vertices
        # Sort by neighborhood size ascending - smaller neighborhoods first
        # This preserves edge availability for larger neighborhoods
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)))
        
        for v in I_sorted:
            N_v = adj[v] & set(C)
            
            if not N_v:
                continue
            
            # Need to cover all edges from v to N_v with cliques containing v
            # A clique {v} ∪ S where S ⊆ N_v requires S to be a clique in G[C]
            # i.e., all pairs in S must have edges in edge_mask
            
            while N_v:
                # Find a maximal clique in the induced subgraph on N_v
                # Using bitmask-based common neighbor computation
                
                # Start with vertex of max degree in N_v
                max_deg_vertex = max(N_v, key=lambda x: bin(edge_mask[x] & ((1 << k) - 1)).count('1'))
                current_clique = {max_deg_vertex}
                
                # Common neighbors of current_clique in N_v
                # Start with neighbors of max_deg_vertex in N_v
                common_mask = edge_mask[max_deg_vertex]
                # Mask to only consider vertices in N_v
                nv_mask = 0
                for u in N_v:
                    nv_mask |= (1 << u)
                common_mask &= nv_mask
                
                # Greedily extend the clique
                while True:
                    # Find best candidate from common_mask
                    best_candidate = None
                    best_score = -1
                    
                    # Iterate through bits in common_mask
                    temp = common_mask
                    while temp:
                        # Get lowest set bit
                        lsb = temp & (-temp)
                        w = lsb.bit_length() - 1
                        temp ^= lsb
                        
                        # Score: degree in remaining graph (edges to other vertices in N_v)
                        score = bin(edge_mask[w] & nv_mask).count('1')
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = w
                    
                    if best_candidate is None:
                        break
                    
                    current_clique.add(best_candidate)
                    # Update common_mask: intersect with neighbors of best_candidate
                    common_mask &= edge_mask[best_candidate]
                
                # Add clique to partition
                clique_list = sorted(list(current_clique | {v}))
                partition.append(clique_list)
                
                # Remove edges of current_clique from edge_mask
                clique_list_sorted = sorted(current_clique)
                for i in range(len(clique_list_sorted)):
                    u = clique_list_sorted[i]
                    for j in range(i + 1, len(clique_list_sorted)):
                        w = clique_list_sorted[j]
                        edge_mask[u] &= ~(1 << w)
                        edge_mask[w] &= ~(1 << u)
                
                # Remove current_clique from N_v
                N_v -= current_clique
        
        # Remaining edges in C: partition into cliques
        # Find all remaining edges
        remaining_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if edge_mask[u] & (1 << v):
                    remaining_edges.append((u, v))
        
        while remaining_edges:
            # Pick edge with maximum common neighborhood
            best_edge = None
            best_common = -1
            
            for e in remaining_edges:
                u, v = e
                # Common neighbors: vertices w where both (u,w) and (v,w) exist
                common_mask = edge_mask[u] & edge_mask[v]
                common_count = bin(common_mask).count('1')
                if common_count > best_common:
                    best_common = common_count
                    best_edge = e
            
            if best_edge is None:
                best_edge = remaining_edges[0]
            
            u, v = best_edge
            current_clique = {u, v}
            
            # Common neighbors of u and v
            common_mask = edge_mask[u] & edge_mask[v]
            
            # Greedily extend
            while True:
                best_candidate = None
                best_score = -1
                
                temp = common_mask
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    temp ^= lsb
                    
                    # Score by degree
                    score = bin(edge_mask[w]).count('1')
                    if score > best_score:
                        best_score = score
                        best_candidate = w
                
                if best_candidate is None:
                    break
                
                current_clique.add(best_candidate)
                common_mask &= edge_mask[best_candidate]
            
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
            
            # Remove edges
            clique_list_sorted = sorted(current_clique)
            for i in range(len(clique_list_sorted)):
                u2 = clique_list_sorted[i]
                for j in range(i + 1, len(clique_list_sorted)):
                    v2 = clique_list_sorted[j]
                    edge_mask[u2] &= ~(1 << v2)
                    edge_mask[v2] &= ~(1 << u2)
            
            # Remove covered edges from remaining_edges
            clique_set = set(current_clique)
            new_remaining = []
            for e in remaining_edges:
                if e[0] in clique_set and e[1] in clique_set:
                    continue
                new_remaining.append(e)
            remaining_edges = new_remaining
        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results
