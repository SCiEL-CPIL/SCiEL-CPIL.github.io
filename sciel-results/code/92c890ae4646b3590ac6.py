import time
import random

def _is_triangle_free(adj_masks):
    n = 16
    for u in range(n):
        v_mask = adj_masks[u]
        while v_mask:
            v_bit = v_mask & -v_mask
            v = v_bit.bit_length() - 1
            v_mask ^= v_bit
            if u < v:
                if adj_masks[u] & adj_masks[v]:
                    return False
    return True

def _is_3_colorable(adj_masks):
    n = 16
    degrees = [bin(mask).count('1') for mask in adj_masks]
    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    
    color = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = 0
        neighbors = adj_masks[v]
        while neighbors:
            u_bit = neighbors & -neighbors
            u = u_bit.bit_length() - 1
            neighbors ^= u_bit
            if color[u] != -1:
                forbidden |= (1 << color[u])
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    deadline = time.time() + 115
    n = 16
    max_edges = n * (n - 1) // 2  # 120
    
    def get_edge_count(adj):
        return sum(bin(m).count('1') for m in adj) // 2
    
    def build_grotzsch():
        """Build Grötzsch graph (Mycielski of C5) on vertices 0-10."""
        adj = [0] * 16
        # C5: 0-1-2-3-4-0
        for u, v in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Copies 5-9
        for u, v in [(5,1),(5,4),(6,0),(6,2),(7,1),(7,3),(8,2),(8,4),(9,3),(9,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Apex 10
        for v in range(5, 10):
            adj[10] |= (1 << v)
            adj[v] |= (1 << 10)
        return adj
    
    def build_turan_triangle_free(adj, start_v):
        """
        Add a new vertex start_v connected to as many existing vertices as possible
        while maintaining triangle-freeness. Use greedy coloring-based approach.
        """
        # To add vertex v, we can connect it to vertices that don't share a common neighbor
        # with each other (otherwise triangle). So we need to find a large independent set
        # in the "conflict graph" where two existing vertices are connected if they share
        # a neighbor.
        
        # Actually, a simpler approach: we can connect v to any set of vertices S such that
        # no two vertices in S have a common neighbor. This means S must be an independent
        # set in the graph where u~v if N(u) ∩ N(v) ≠ ∅.
        
        # Let's build the conflict graph
        conflict = [0] * n
        for u in range(n):
            for w in range(n):
                if u == w:
                    continue
                # u and w conflict if they share a common neighbor
                if adj[u] & adj[w]:
                    conflict[u] |= (1 << w)
                    conflict[w] |= (1 << u)
        
        # Now find maximum independent set in the conflict graph (on existing vertices only)
        # For a 16-vertex graph, we can do this with backtracking
        best_size = 0
        best_set = 0
        
        def find_max_independent(idx, current_set, remaining):
            nonlocal best_size, best_set
            if remaining == 0:
                size = bin(current_set).count('1')
                if size > best_size:
                    best_size = size
                    best_set = current_set
                return
            
            # Branch: include idx or not
            # Include idx
            new_set = current_set | (1 << idx)
            new_remaining = remaining & ~conflict[idx]
            find_max_independent(idx + 1, new_set, new_remaining)
            
            # Exclude idx
            find_max_independent(idx + 1, current_set, remaining)
        
        find_max_independent(0, 0, (1 << n) - 1)
        
        # Connect start_v to the vertices in best_set
        for v in range(n):
            if best_set & (1 << v):
                adj[start_v] |= (1 << v)
                adj[v] |= (1 << start_v)
        
        return best_size
    
    # Start with Grötzsch graph
    current_adj = build_grotzsch()
    current_edges = get_edge_count(current_adj)
    best_adj = [m for m in current_adj]
    best_edges = current_edges
    
    # Add vertices 11-15 using the independent set approach
    for new_v in range(11, n):
        if time.time() > deadline:
            break
        build_turan_triangle_free(current_adj, new_v)
        current_edges = get_edge_count(current_adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [m for m in current_adj]
    
    # Now do local search to improve
    def get_forbidden_pairs(adj):
        """Return set of (u,v) pairs that would create a triangle if added."""
        forbidden = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    continue  # already an edge
                if adj[u] & adj[v]:
                    forbidden.add((u, v))
        return forbidden
    
    # Local search: try to add edges greedily, then do local optimization
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Phase 1: Greedily add edges that maintain triangle-freeness and 4-chromaticity
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            forbidden = get_forbidden_pairs(current_adj)
            # Try all non-forbidden, non-existing edges
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in forbidden:
                        continue
                    if current_adj[u] & (1 << v):
                        continue
                    candidates.append((u, v))
            
            random.shuffle(candidates)
            for u, v in candidates:
                if time.time() > deadline:
                    break
                new_adj = [m for m in current_adj]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                
                # Check 4-chromatic (not 3-colorable)
                if _is_3_colorable(new_adj):
                    continue
                
                # Accept
                current_adj = new_adj
                current_edges += 1
                added_any = True
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [m for m in current_adj]
                break  # Restart candidate list after each addition
        
        # Phase 2: Local search - try edge swaps and additions
        for step in range(300):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Add edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or (current_adj[u] & (1 << v)):
                    continue
                # Quick triangle check
                if current_adj[u] & current_adj[v]:
                    continue
                
                new_adj = [m for m in current_adj]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                
                if _is_3_colorable(new_adj):
                    continue
                
                new_edges = current_edges + 1
                current_adj = new_adj
                current_edges = new_edges
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [m for m in current_adj]
                    
            elif move_type < 0.8:
                # Remove edge
                edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        v_bit = v_mask & -v_mask
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit
                        if u < v:
                            edges.append((u, v))
                if not edges:
                    continue
                u, v = random.choice(edges)
                new_adj = [m for m in current_adj]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                if _is_3_colorable(new_adj):
                    continue
                
                current_adj = new_adj
                current_edges -= 1
                
            else:
                # Swap: remove one edge, add another
                edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        v_bit = v_mask & -v_mask
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit
                        if u < v:
                            edges.append((u, v))
                if not edges:
                    continue
                u1, v1 = random.choice(edges)
                
                new_adj = [m for m in current_adj]
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2 or (new_adj[u2] & (1 << v2)):
                    continue
                if new_adj[u2] & new_adj[v2]:
                    continue
                
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                
                if _is_3_colorable(new_adj):
                    continue
                
                current_adj = new_adj
                # Edge count stays same
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
