import time

def construct():
    start_time = time.time()
    
    def get_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        # adj_mask[i] is an integer bitmask of neighbors of i
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def solve_instance(n, p, seed):
        adj_mask = get_graph(n, p, seed)
        
        # Check if graph is empty
        if not any(adj_mask):
            return []
            
        cliques = []
        
        # Helper to iterate through set bits
        def iter_bits(x):
            while x:
                lsb = x & -x
                yield lsb.bit_length() - 1
                x ^= lsb

        # Helper: find edge (u,v) with minimum common neighbors in remaining graph
        def find_min_common_edge(adj_mask, n):
            best_u, best_v = -1, -1
            best_common = float('inf')
            for u in range(n):
                if adj_mask[u] == 0:
                    continue
                for v in iter_bits(adj_mask[u]):
                    if v <= u:
                        continue
                    common = (adj_mask[u] & adj_mask[v]).bit_count()
                    if common < best_common:
                        best_common = common
                        best_u, best_v = u, v
            return best_u, best_v, best_common

        # Greedy clique expansion starting from a seed edge
        def expand_clique(seed_u, seed_v, adj_mask):
            C = [seed_u, seed_v]
            # Candidates must be neighbors of both seed_u and seed_v
            U = adj_mask[seed_u] & adj_mask[seed_v]
            
            while U:
                best_u = -1
                best_score = -1
                
                for u in iter_bits(U):
                    # Score: how many neighbors does u have within U?
                    score = (adj_mask[u] & U).bit_count()
                    if score > best_score:
                        best_score = score
                        best_u = u
                
                if best_u == -1:
                    break
                    
                C.append(best_u)
                U &= adj_mask[best_u]
                
            return C

        # Remove all edges within clique C from adj_mask
        def remove_clique_edges(C, adj_mask):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)

        # Local improvement: try to merge two cliques if their union is a clique
        def try_merge(cliques, adj_mask, n):
            changed = True
            while changed and (time.time() - start_time < 110):
                changed = False
                new_cliques = []
                i = 0
                while i < len(cliques):
                    merged = False
                    C_i = cliques[i]
                    for j in range(i + 1, len(cliques)):
                        C_j = cliques[j]
                        # Check if union is a clique
                        is_clique = True
                        # Every vertex in C_i must be connected to every vertex in C_j
                        for u in C_i:
                            for v in C_j:
                                if not (adj_mask[u] & (1 << v)):
                                    is_clique = False
                                    break
                            if not is_clique:
                                break
                        if is_clique:
                            # Merge: remove edges of both cliques, add union as one clique
                            new_cliques.append(C_i + C_j)
                            # Mark C_j as used
                            # We need to skip j in subsequent iterations
                            # Simple approach: rebuild the list
                            new_cliques = [C_i + C_j] + [cliques[k] for k in range(i+1, len(cliques)) if k != j]
                            i = 0  # Restart scanning
                            changed = True
                            break
                    if not merged:
                        new_cliques.append(C_i)
                        i += 1
                cliques = new_cliques
            return cliques

        while True:
            if time.time() - start_time > 115:
                break
                
            # Find edge with minimum common neighbors
            u, v, common = find_min_common_edge(adj_mask, n)
            
            if u == -1:
                break
                
            # Expand clique from this edge
            C = expand_clique(u, v, adj_mask)
            
            # Add clique to result
            cliques.append(C)
            
            # Remove edges within C from adj_mask
            remove_clique_edges(C, adj_mask)
        
        # Local improvement pass
        cliques = try_merge(cliques, adj_mask, n)
        
        return cliques

    results = []
    # Instance 1
    results.append(solve_instance(32, 0.6, 601))
    
    # Instance 2
    results.append(solve_instance(64, 0.55, 602))
    
    return results
