import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 256
    # Squares up to 256+256=512. sqrt(512) ~ 22.6
    squares = set(k*k for k in range(1, 23))
    
    # 1. Filter out nodes that are individually invalid (2*a is a square)
    # If 2*a is a square, a cannot be in any valid set because a+a=2a is a square.
    valid_nodes = []
    for i in range(1, N + 1):
        if 2 * i not in squares:
            valid_nodes.append(i)
            
    n = len(valid_nodes)
    if n == 0:
        return [1]
        
    # Map valid nodes to indices 0..n-1
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
    
    # 2. Build Complement Graph Adjacency List (Bitmask)
    # Edge exists in complement if sum is NOT a square.
    # We want Max Clique in this complement graph.
    # adj[i] is a bitmask of neighbors of i in the complement graph.
    adj = [0] * n
    
    for i in range(n):
        u = valid_nodes[i]
        mask = 0
        for j in range(i + 1, n):
            v = valid_nodes[j]
            s = u + v
            r = math.isqrt(s)
            if r * r != s:
                # Not a square sum -> Edge in complement
                mask |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = mask
        
    # 3. Maximum Clique via Branch and Bound with Greedy Coloring
    # We use bitmasks for candidates.
    
    best_clique_size = 0
    best_clique_indices = []
    
    # Correct Greedy Coloring Implementation
    def get_coloring(candidates):
        uncolored = candidates
        color_assign = {}
        max_color = 0
        
        while uncolored:
            max_color += 1
            # Find maximal independent set in H[uncolored]
            # Independent in H means no edges in H.
            # Edges in H are defined by adj.
            # So we pick v, then remove v and all neighbors of v in H.
            temp = uncolored
            color_mask = 0
            while temp:
                lsb = temp & (-temp)
                v_idx = lsb.bit_length() - 1
                color_mask |= lsb
                # Remove v and its H-neighbors from temp
                temp = temp & ~(lsb | adj[v_idx])
            
            # Assign max_color to all vertices in color_mask
            cm = color_mask
            while cm:
                lsb = cm & (-cm)
                v_idx = lsb.bit_length() - 1
                color_assign[v_idx] = max_color
                cm = cm & (cm - 1)
                
            uncolored = uncolored & ~color_mask
            
        return max_color, color_assign

    # Branch and Bound
    def expand(candidates, current_clique):
        nonlocal best_clique_size, best_clique_indices
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if len(current_clique) > best_clique_size:
                best_clique_size = len(current_clique)
                best_clique_indices = current_clique[:]
            return
            
        # Get coloring bound
        max_color, color_assign = get_coloring(candidates)
        
        # If current clique size + max possible (max_color) <= best, prune
        if len(current_clique) + max_color <= best_clique_size:
            return
            
        # Order candidates by decreasing color
        # Extract indices from candidates
        cand_indices = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            cand_indices.append(idx)
            tmp = tmp & (tmp - 1)
            
        cand_indices.sort(key=lambda x: color_assign[x], reverse=True)
        
        for i, v in enumerate(cand_indices):
            if time.time() > deadline:
                return
                
            # Pruning based on remaining vertices
            # The max color of the remaining vertices (v and those after it) is an upper bound
            # Since we sorted by color, the max color for the suffix starting at i is color_assign[v]?
            # Actually, Tomita's algorithm uses the color of the current vertex as the bound for the subtree rooted at it.
            
            if len(current_clique) + color_assign[v] <= best_clique_size:
                break # No need to check subsequent vertices with lower/equal color
                
            # Include v
            # New candidates must be neighbors of v in H (complement graph)
            new_candidates = candidates & adj[v]
            
            expand(new_candidates, current_clique + [v])
            
            # Exclude v: remove v from candidates for the next iteration of the loop
            candidates = candidates & ~(1 << v)
            
    # Initial greedy lower bound
    def greedy_clique():
        clique = []
        cands = (1 << n) - 1
        while cands:
            # Pick vertex with highest degree in induced subgraph
            best_v = -1
            best_deg = -1
            tmp = cands
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                deg = (cands & adj[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                tmp = tmp & (tmp - 1)
                
            if best_v == -1: break
            
            clique.append(best_v)
            # Intersect with neighbors of best_v
            cands = cands & adj[best_v]
            
        return clique
        
    init_clique = greedy_clique()
    best_clique_size = len(init_clique)
    best_clique_indices = init_clique[:]
    
    # Run BnB
    # Start with all valid nodes
    all_cands = (1 << n) - 1
    expand(all_cands, [])
    
    # Convert indices back to values
    result = [valid_nodes[i] for i in best_clique_indices]
    result.sort()
    
    # Final verification
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                # Should not happen if logic is correct, but fallback to single element
                return [1]
                
    if not result:
        return [1]
        
    return result
