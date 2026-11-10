"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n = 16
    # Pre-compute all 8-subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    # Pre-compute masks for fast intersection
    subset_masks = []
    for subset in subsets_8:
        mask = 0
        for v in subset:
            mask |= (1 << v)
        subset_masks.append(mask)

    def is_q3_free(adj):
        """
        Check if the graph defined by adj (list of bitmasks) is Q3-free.
        Q3 is isomorphic to K_{4,4} minus a perfect matching.
        Properties:
        1. 8 vertices.
        2. Bipartite with parts of size 4.
        3. 3-regular (each vertex degree 3).
        4. Connected (implied by 3-regular bipartite on 8 vertices).
        
        So we check:
        1. Degrees in subset are all 3.
        2. Subgraph is bipartite.
        """
        for i in range(len(subsets_8)):
            subset = subsets_8[i]
            mask = subset_masks[i]
            
            # 1. Check degrees
            degs = [0] * 8
            edge_count = 0
            for k, v in enumerate(subset):
                nb = adj[v] & mask
                deg = nb.bit_count()
                degs[k] = deg
                edge_count += deg
            edge_count //= 2
            
            # Q3 has 12 edges. If edge count is not 12, skip.
            # Also, if any degree is not 3, skip.
            if edge_count != 12:
                continue
            if any(d != 3 for d in degs):
                continue
            
            # 2. Check Bipartite
            # Build local adjacency for the 8 vertices
            # Map global index to local 0..7
            # local_adj[k] is bitmask of neighbors within the subset
            local_adj = [0] * 8
            for k, v in enumerate(subset):
                nb = adj[v] & mask
                # Convert global neighbor bitmask to local
                local_nb = 0
                for j, u in enumerate(subset):
                    if nb & (1 << u):
                        local_nb |= (1 << j)
                local_adj[k] = local_nb
            
            # BFS to check bipartite
            color = [-1] * 8
            # Since it's 3-regular and 8 vertices, it must be connected if bipartite?
            # Actually, if it's 3-regular bipartite, it is connected.
            # But let's just check if it can be 2-colored.
            # Start from vertex 0.
            color[0] = 0
            queue = [0]
            head = 0
            is_bipartite = True
            while head < len(queue):
                u = queue[head]
                head += 1
                # Iterate neighbors in local graph
                nb_mask = local_adj[u]
                while nb_mask:
                    # Get lowest set bit
                    lsb = nb_mask & -nb_mask
                    v_idx = lsb.bit_length() - 1
                    nb_mask ^= lsb
                    
                    if color[v_idx] == -1:
                        color[v_idx] = 1 - color[u]
                        queue.append(v_idx)
                    elif color[v_idx] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
            
            if is_bipartite:
                # If it's 3-regular, bipartite, and 8 vertices, it is Q3.
                return False
        
        return True

    best_edges = []
    best_count = 0
    
    # Strategy: Greedy construction with multiple restarts
    # We can also try to improve the greedy result with local search (add/remove)
    
    for restart in range(200): # Increase restarts since check is fast
        if time.time() > deadline:
            break
            
        # Generate all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        adj = [0] * n
        current_edges = []
        
        for u, v in all_edges:
            # Try adding edge (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges.append([u, v])
            
            if not is_q3_free(adj):
                # Roll back
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_edges.pop()
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            # Optional: Local improvement
            # Try to add edges that were skipped? 
            # Or try to swap edges?
            # For now, just record the best greedy result.
            
    return best_edges

# EVOLVE-BLOCK-END
