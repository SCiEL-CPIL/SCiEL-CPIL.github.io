<<<<<<< SEARCH
def find_hamilton_cycle_fast(adj, n, deadline):
    """Fast Hamilton cycle finder with aggressive pruning."""
    # Quick degree check
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Fix vertex 0 as start
    path = [0]
    used = [False] * n
    used[0] = True
    
    # Pre-sort neighbors by degree for MRV heuristic
    neighbor_order = [list(adj[v]) for v in range(n)]
    for v in range(n):
        neighbor_order[v].sort(key=lambda x: len(adj[x]))
    
    def backtrack(depth):
        if time.time() > deadline:
            return None
        
        if depth == n:
            last = path[-1]
            if 0 in adj[last]:
                return path[:]
            return None
        
        last = path[-1]
        
        # Try neighbors in MRV order
        for nxt in neighbor_order[last]:
            if used[nxt]:
                continue
            
            # Pruning: if this is the second-to-last step, check closing edge feasibility
            if depth == n - 1:
                remaining = [v for v in range(n) if not used[v] and v != nxt]
                if len(remaining) == 1:
                    r = remaining[0]
                    if r not in adj[nxt] or 0 not in adj[r]:
                        continue
                elif len(remaining) == 0:
                    # nxt is the last vertex, must connect to 0
                    if 0 not in adj[nxt]:
                        continue
            
            # Additional pruning: check degree constraint for unvisited vertices
            # After adding nxt, ensure no unvisited vertex has fewer than 2 
            # available neighbors (counting edges to other unvisited + edges to path)
            if depth >= 2:
                # Count available degree for each unvisited vertex
                unvisited_check = [v for v in range(n) if not used[v] and v != nxt]
                if unvisited_check:
                    path_set_check = set(path)
                    valid = True
                    for v in unvisited_check:
                        # Count neighbors: unvisited (excluding v) + path
                        deg = 0
                        for w in adj[v]:
                            if w == nxt:
                                continue  # nxt is now used, but edge v-nxt exists
                            if used[w] and w != nxt:
                                deg += 1  # edge to path
                            elif not used[w] and w != v:
                                deg += 1  # edge to other unvisited
                        # Also count edge to nxt if it exists
                        if nxt in adj[v]:
                            deg += 1
                        if deg < 2:
                            valid = False
                            break
                    if not valid:
                        continue
            
            used[nxt] = True
            path.append(nxt)
            
            # Connectivity pruning: check that unvisited vertices are connected
            # to each other (through unvisited edges) and at least one connects to path
            if depth < n - 1:
                unvisited = [v for v in range(n) if not used[v]]
                if len(unvisited) > 0:
                    # Check connectivity of unvisited subgraph
                    visited_u = set()
                    stack = [unvisited[0]]
                    visited_u.add(unvisited[0])
                    while stack:
                        u = stack.pop()
                        for w in adj[u]:
                            if w in unvisited and w not in visited_u:
                                visited_u.add(w)
                                stack.append(w)
                    if len(visited_u) != len(unvisited):
                        path.pop()
                        used[nxt] = False
                        continue
                    
                    # Check that at least one unvisited vertex connects to path
                    path_set = set(path)
                    has_connection = False
                    for v in unvisited:
                        for w in adj[v]:
                            if w in path_set:
                                has_connection = True
                                break
                        if has_connection:
                            break
                    if not has_connection:
                        path.pop()
                        used[nxt] = False
                        continue
            
            result = backtrack(depth + 1)
            if result is not None:
                return result
            
            path.pop()
            used[nxt] = False
        
        return None
    
    return backtrack(1)
=======
def find_hamilton_cycle_fast(adj, n, deadline):
    """Fast Hamilton cycle finder with aggressive pruning using bitmasks."""
    # Quick degree check
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Build bitmask adjacency
    masks = bitmask_from_adj(adj)
    
    # Pre-compute sorted neighbor lists for MRV
    neighbor_order = [list(adj[v]) for v in range(n)]
    for v in range(n):
        neighbor_order[v].sort(key=lambda x: len(adj[x]))
    
    # Track which edges are "forced" - if a vertex has degree 2, both edges must be used
    # This can propagate constraints
    
    def get_unvisited_mask(used_mask):
        return MASK_ALL ^ used_mask
    
    def check_degree_feasible(used_mask, last_vertex):
        """Check that all unvisited vertices can still have degree >= 2 in the cycle."""
        unvisited = get_unvisited_mask(used_mask)
        if unvisited == 0:
            return True
        
        # For each unvisited vertex, count how many neighbors are available
        # (either unvisited or the last vertex in path, or vertex 0 if it's the start)
        for v in range(n):
            if not (unvisited & (1 << v)):
                continue
            # Count neighbors that are unvisited or are the last vertex
            avail = masks[v] & (unvisited | (1 << last_vertex))
            # Also, vertex 0 will be connected at the end, so count edge to 0
            if (masks[v] & 1) and v != 0:
                avail |= 1
            if bin(avail).count('1') < 2:
                return False
        return True
    
    def check_connectivity(used_mask):
        """Check that unvisited vertices form a connected subgraph (within unvisited edges)
        and at least one connects to the path."""
        unvisited = get_unvisited_mask(used_mask)
        if unvisited == 0:
            return True
        
        # Find first unvisited vertex
        first_v = -1
        for v in range(n):
            if unvisited & (1 << v):
                first_v = v
                break
        if first_v == -1:
            return True
        
        # BFS/DFS within unvisited vertices
        visited = 1 << first_v
        stack = [first_v]
        while stack:
            u = stack.pop()
            neighbors = masks[u] & unvisited & ~visited
            while neighbors:
                lsb = neighbors & (-neighbors)
                w = lsb.bit_length() - 1
                visited |= lsb
                stack.append(w)
                neighbors ^= lsb
        
        if visited != unvisited:
            return False
        
        # Check at least one unvisited vertex connects to used vertices (path)
        path_mask = used_mask
        for v in range(n):
            if unvisited & (1 << v):
                if masks[v] & path_mask:
                    return True
        return False
    
    def check_biconnectivity(used_mask):
        """Check that unvisited subgraph is biconnected (no articulation points)
        when combined with the path endpoints. This is a stronger check."""
        unvisited = get_unvisited_mask(used_mask)
        if unvisited == 0:
            return True
        
        # For biconnectivity, we need the unvisited subgraph plus connections
        # to path to remain connected after removing any single vertex.
        # Simplified: check no articulation point in unvisited subgraph
        # that would disconnect it.
        
        # Find all unvisited vertices
        uv_list = []
        for v in range(n):
            if unvisited & (1 << v):
                uv_list.append(v)
        
        if len(uv_list) <= 2:
            return True
        
        # Check articulation points in the induced subgraph on unvisited vertices
        # plus virtual edges to path endpoints
        # Simplified check: for each unvisited vertex, removing it shouldn't
        # disconnect the remaining unvisited vertices from each other or from path
        
        for rem in uv_list:
            remaining = unvisited & ~(1 << rem)
            if remaining == 0:
                continue
            
            # Check if remaining unvisited vertices are still connected
            # (considering edges between them and to path)
            first_r = -1
            for v in range(n):
                if remaining & (1 << v):
                    first_r = v
                    break
            
            visited_r = 1 << first_r
            stack_r = [first_r]
            while stack_r:
                u = stack_r.pop()
                # Neighbors in remaining, or neighbors in used_mask (path)
                # that can connect back
                neighbors = masks[u] & (remaining | used_mask) & ~visited_r
                # But we only traverse within remaining for connectivity
                neighbors_in_rem = masks[u] & remaining & ~visited_r
                while neighbors_in_rem:
                    lsb = neighbors_in_rem & (-neighbors_in_rem)
                    w = lsb.bit_length() - 1
                    visited_r |= lsb
                    stack_r.append(w)
                    neighbors_in_rem ^= lsb
            
            # Check all remaining vertices are reachable
            if visited_r != remaining:
                return False
            
            # Also check that remaining vertices can still connect to path
            # (at least one vertex in remaining has an edge to used_mask)
            has_path_conn = False
            for v in range(n):
                if remaining & (1 << v):
                    if masks[v] & used_mask:
                        has_path_conn = True
                        break
            if not has_path_conn:
                return False
        
        return True
    
    # Fix vertex 0 as start
    path = [0]
    used_mask = 1  # bit 0 set
    
    def backtrack(depth):
        if time.time() > deadline:
            return None
        
        if depth == n:
            last = path[-1]
            if masks[last] & 1:  # edge to vertex 0
                return path[:]
            return None
        
        last = path[-1]
        
        # Try neighbors in MRV order
        for nxt in neighbor_order[last]:
            if used_mask & (1 << nxt):
                continue
            
            # Check edge exists
            if not (masks[last] & (1 << nxt)):
                continue
            
            new_used_mask = used_mask | (1 << nxt)
            
            # Degree feasibility check
            if not check_degree_feasible(new_used_mask, nxt):
                continue
            
            # Connectivity check
            if depth < n - 1:
                if not check_connectivity(new_used_mask):
                    continue
            
            # For deeper levels, try biconnectivity (expensive but powerful)
            if depth >= 4 and depth < n - 2:
                if not check_biconnectivity(new_used_mask):
                    continue
            
            # Special case: if this is the last vertex to place
            if depth == n - 1:
                # nxt must connect back to 0
                if not (masks[nxt] & 1):
                    continue
                # Also check all other unvisited vertices (should be none since depth==n-1)
                # Actually at depth n-1, we're placing the (n-1)th vertex (0-indexed),
                # so one vertex remains. Wait, depth is the current path length.
                # path has depth elements. We're adding nxt, making it depth+1.
                # If depth+1 == n, then nxt is the last vertex.
                # Check: does the remaining unvisited vertex (if any) work?
                rem_mask = MASK_ALL ^ new_used_mask
                if rem_mask:
                    # One vertex remains, it must connect to both nxt and 0
                    rem_v = rem_mask.bit_length() - 1
                    if not (masks[nxt] & (1 << rem_v)):
                        continue
                    if not (masks[rem_v] & 1):
                        continue
                    # Place rem_v after nxt
                    path.append(nxt)
                    path.append(rem_v)
                    return path[:]
                else:
                    # nxt is the last vertex, must connect to 0
                    path.append(nxt)
                    return path[:]
            
            path.append(nxt)
            result = backtrack(depth + 1)
            if result is not None:
                return result
            path.pop()
        
        return None
    
    return backtrack(1)
>>>>>>> REPLACE

