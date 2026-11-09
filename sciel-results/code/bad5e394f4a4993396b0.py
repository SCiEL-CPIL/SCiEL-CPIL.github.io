import time
import random
import math

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
    
    # Start with Grötzsch graph
    current_adj = build_grotzsch()
    current_edges = get_edge_count(current_adj)
    best_adj = [m for m in current_adj]
    best_edges = current_edges
    
    # Precompute: for each vertex, the set of vertices it CANNOT be connected to
    # (would create a triangle) given current adjacency
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
    
    def get_all_edges(adj):
        """Get list of all edges in the graph."""
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                if u < v:
                    edges.append((u, v))
        return edges
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it maintains triangle-freeness and 4-chromaticity."""
        if adj[u] & (1 << v):
            return None
        if adj[u] & adj[v]:
            return None
        new_adj = [m for m in adj]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        if _is_3_colorable(new_adj):
            return None
        return new_adj
    
    def try_remove_edge(adj, u, v):
        """Try to remove edge (u,v) if it maintains 4-chromaticity."""
        if not (adj[u] & (1 << v)):
            return None
        new_adj = [m for m in adj]
        new_adj[u] &= ~(1 << v)
        new_adj[v] &= ~(1 << u)
        if _is_3_colorable(new_adj):
            return None
        return new_adj
    
    def try_swap_edges(adj, u1, v1, u2, v2):
        """Try to swap edges: remove (u1,v1) and add (u2,v2)."""
        if not (adj[u1] & (1 << v1)):
            return None
        if adj[u2] & (1 << v2):
            return None
        if adj[u2] & adj[v2]:
            return None
        
        new_adj = [m for m in adj]
        new_adj[u1] &= ~(1 << v1)
        new_adj[v1] &= ~(1 << u1)
        new_adj[u2] |= (1 << v2)
        new_adj[v2] |= (1 << u2)
        
        if _is_3_colorable(new_adj):
            return None
        return new_adj
    
    # Local search: try to add edges greedily, then do local optimization
    for iteration in range(200):
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
                new_adj = try_add_edge(current_adj, u, v)
                if new_adj is None:
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
        for step in range(500):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                new_adj = try_add_edge(current_adj, u, v)
                if new_adj is None:
                    continue
                
                current_adj = new_adj
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [m for m in current_adj]
                    
            elif move_type < 0.6:
                # Remove edge
                edges = get_all_edges(current_adj)
                if not edges:
                    continue
                u, v = random.choice(edges)
                new_adj = try_remove_edge(current_adj, u, v)
                if new_adj is None:
                    continue
                
                current_adj = new_adj
                current_edges -= 1
                
            elif move_type < 0.85:
                # Swap: remove one edge, add another
                edges = get_all_edges(current_adj)
                if not edges:
                    continue
                u1, v1 = random.choice(edges)
                
                # Try multiple random candidates for the new edge
                found = False
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 == v2:
                        continue
                    new_adj = try_swap_edges(current_adj, u1, v1, u2, v2)
                    if new_adj is not None:
                        current_adj = new_adj
                        found = True
                        break
                if not found:
                    continue
                # Edge count stays same
            else:
                # Structured swap: try all possible swaps for a removed edge
                edges = get_all_edges(current_adj)
                if not edges:
                    continue
                u1, v1 = random.choice(edges)
                
                # Try all possible (u2,v2) pairs
                best_swap = None
                for u2 in range(n):
                    if time.time() > deadline:
                        break
                    for v2 in range(u2+1, n):
                        new_adj = try_swap_edges(current_adj, u1, v1, u2, v2)
                        if new_adj is not None:
                            # Check if this swap could enable future improvements
                            # by seeing if we can add an edge after the swap
                            new_edges_count = current_edges
                            # Try to add an edge after the swap
                            for u3 in range(n):
                                for v3 in range(u3+1, n):
                                    if new_adj[u3] & (1 << v3):
                                        continue
                                    if new_adj[u3] & new_adj[v3]:
                                        continue
                                    test_adj = try_add_edge(new_adj, u3, v3)
                                    if test_adj is not None:
                                        # This swap enables an edge addition
                                        best_swap = (new_adj, u3, v3)
                                        break
                                if best_swap:
                                    break
                            if best_swap:
                                break
                    if best_swap:
                        break
                
                if best_swap:
                    new_adj, u3, v3 = best_swap
                    current_adj = try_add_edge(new_adj, u3, v3)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [m for m in current_adj]
    
    # Final polish: try to add more edges to the best solution found
    if time.time() < deadline - 5:
        polish_adj = [m for m in best_adj]
        polish_edges = best_edges
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            forbidden = get_forbidden_pairs(polish_adj)
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in forbidden:
                        continue
                    if polish_adj[u] & (1 << v):
                        continue
                    candidates.append((u, v))
            
            for u, v in candidates:
                if time.time() > deadline - 2:
                    break
                new_adj = try_add_edge(polish_adj, u, v)
                if new_adj is not None:
                    polish_adj = new_adj
                    polish_edges += 1
                    improved = True
                    if polish_edges > best_edges:
                        best_edges = polish_edges
                        best_adj = [m for m in polish_adj]
                    break
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
