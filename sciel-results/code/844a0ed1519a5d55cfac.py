import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def iter_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    def solve_instance(n, p, seed):
        adj_mask = get_graph(n, p, seed)
        
        # Check if graph is empty
        if not any(adj_mask):
            return []
            
        cliques = []
        
        # Helper to find edge (u,v) with minimum common neighbors in remaining graph
        # This helps prioritize "tight" edges that constrain the clique structure
        def find_min_common_edge(adj_mask, n):
            best_u, best_v = -1, -1
            best_common = float('inf')
            for u in range(n):
                mask_u = adj_mask[u]
                if mask_u == 0:
                    continue
                for v in iter_bits(mask_u):
                    if v <= u:
                        continue
                    # Common neighbors of u and v
                    common = (adj_mask[u] & adj_mask[v]).bit_count()
                    if common < best_common:
                        best_common = common
                        best_u, best_v = u, v
            return best_u, best_v

        # Greedy clique expansion starting from a seed edge (u, v)
        def expand_clique(seed_u, seed_v, adj_mask):
            C = [seed_u, seed_v]
            # Candidates must be neighbors of both seed_u and seed_v
            U = adj_mask[seed_u] & adj_mask[seed_v]
            
            while U:
                best_u = -1
                best_score = -1
                
                for u in iter_bits(U):
                    # Score: how many neighbors does u have within the current candidate set U?
                    # This encourages picking vertices that keep the candidate set "dense"
                    score = (adj_mask[u] & U).bit_count()
                    if score > best_score:
                        best_score = score
                        best_u = u
                
                if best_u == -1:
                    break
                    
                C.append(best_u)
                # Update candidates: must be connected to new vertex best_u
                U &= adj_mask[best_u]
                
            return C

        # Remove all edges within clique C from adj_mask
        def remove_clique_edges(C, adj_mask):
            for i in range(len(C)):
                u = C[i]
                for j in range(i + 1, len(C)):
                    v = C[j]
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)

        # Local improvement: try to merge two cliques if their union is a clique
        # This is a simple O(K^2) pass. K is expected to be small.
        def try_merge(cliques, adj_mask, n):
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                new_cliques = []
                i = 0
                while i < len(cliques):
                    C_i = cliques[i]
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        C_j = cliques[j]
                        # Check if union C_i U C_j is a clique
                        # Every vertex in C_i must be connected to every vertex in C_j
                        is_clique = True
                        for u in C_i:
                            mask_u = adj_mask[u]
                            for v in C_j:
                                if not (mask_u & (1 << v)):
                                    is_clique = False
                                    break
                            if not is_clique:
                                break
                        if is_clique:
                            # Merge C_i and C_j
                            new_cliques.append(C_i + C_j)
                            # Rebuild the rest of the list excluding i and j
                            # We are currently processing i, and found j.
                            # The new list will have the merged clique at position i.
                            # We need to keep cliques[k] for k > j, and k < i are already in new_cliques.
                            # Actually, the logic above appends to new_cliques.
                            # Let's restart the outer loop to be safe and simple, 
                            # as K is small.
                            changed = True
                            # Reconstruct cliques list for next iteration of while changed
                            # We can just break out and let the outer while loop restart with the modified list?
                            # No, `try_merge` modifies `cliques` in place or returns new.
                            # Let's return the new list immediately to restart the check.
                            remaining = [cliques[k] for k in range(i+1, len(cliques)) if k != j]
                            return [C_i + C_j] + remaining
                    if not merged:
                        new_cliques.append(C_i)
                        i += 1
                cliques = new_cliques
            return cliques

        while True:
            if time.time() > deadline:
                break
                
            # Find the "bottleneck" edge
            u, v = find_min_common_edge(adj_mask, n)
            
            if u == -1:
                break
                
            # Expand clique from this edge
            C = expand_clique(u, v, adj_mask)
            
            # Add clique to result
            cliques.append(C)
            
            # Remove edges within C from adj_mask
            remove_clique_edges(C, adj_mask)
        
        # Local improvement pass
        if time.time() < deadline:
            cliques = try_merge(cliques, adj_mask, n)
        
        return cliques

    results = []
    # Instance 1
    results.append(solve_instance(32, 0.6, 601))
    
    # Instance 2
    results.append(solve_instance(64, 0.55, 602))
    
    return results
