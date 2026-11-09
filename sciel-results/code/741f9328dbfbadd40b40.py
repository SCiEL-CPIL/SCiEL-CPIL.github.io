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

    current_adj = build_grotzsch()
    current_edges = get_edge_count(current_adj)
    best_adj = [m for m in current_adj]
    best_edges = current_edges
    
    # Add vertices 11, 12, 13, 14, 15
    for new_v in range(11, 16):
        if time.time() > deadline:
            break
            
        # Find all independent sets in current_adj[0:new_v]
        # We want to connect new_v to an independent set I
        # such that the new graph is not 3-colorable.
        # To maximize edges, we prefer large I.
        
        candidates = []
        # Generate independent sets of the current graph (vertices 0..new_v-1)
        # We can do this by backtracking on the subgraph
        
        def get_independent_sets(adj, n_vertices):
            sets = []
            def backtrack(start, current_set, mask):
                if start == n_vertices:
                    sets.append(current_set)
                    return
                # Option 1: don't include start
                backtrack(start + 1, current_set, mask)
                # Option 2: include start if it's independent
                if not (adj[start] & mask):
                    backtrack(start + 1, current_set | (1 << start), mask | (1 << start))
            
            backtrack(0, 0, 0)
            return sets
        
        indep_sets = get_independent_sets(current_adj, new_v)
        
        # Sort by size descending
        indep_sets.sort(key=lambda x: bin(x).count('1'), reverse=True)
        
        found = False
        for mask in indep_sets:
            if bin(mask).count('1') == 0:
                continue
                
            # Try adding new_v connected to vertices in mask
            new_adj = [m for m in current_adj]
            for u in range(new_v):
                if mask & (1 << u):
                    new_adj[u] |= (1 << new_v)
                    new_adj[new_v] |= (1 << u)
            
            if not _is_3_colorable(new_adj):
                current_adj = new_adj
                current_edges += bin(mask).count('1')
                found = True
                break
        
        if not found:
            # Fallback: connect to nothing
            current_adj = [m for m in current_adj]

    # Final local search
    def get_addable_edges(adj):
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    continue
                if adj[u] & adj[v]:
                    continue
                candidates.append((u, v))
        return candidates

    for step in range(5000):
        if time.time() > deadline:
            break
        
        addable = get_addable_edges(current_adj)
        if not addable:
            break
            
        # Randomly pick an edge to try adding
        u, v = random.choice(addable)
        new_adj = [m for m in current_adj]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if not _is_3_colorable(new_adj):
            current_adj = new_adj
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [m for m in current_adj]
        else:
            # Try to swap: remove a different edge to allow this one?
            # Or just reject.
            pass

    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
