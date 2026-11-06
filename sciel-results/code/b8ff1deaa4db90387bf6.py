"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit
    n = 16
    full_mask = (1 << n) - 1
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free using adjacency bitmasks."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    if ai & adj[j]:
                        return False
        return True
    
    def is_3_colorable(adj):
        """Check if graph is 3-colorable using backtracking with bitmasks and MRV.
        Returns True if 3-colorable, False if not (i.e., 4-chromatic or higher).
        """
        neighbors = [adj[i] for i in range(n)]
        
        # colors[v] = -1 (uncolored), 0, 1, or 2
        # We only need to track color_masks to determine availability
        color_masks = [0, 0, 0]
        
        def get_available_colors(v, color_masks):
            nmask = neighbors[v]
            avail = 7  # 0b111
            if nmask & color_masks[0]: avail &= ~1
            if nmask & color_masks[1]: avail &= ~2
            if nmask & color_masks[2]: avail &= ~4
            return avail
        
        def backtrack(uncolored_mask):
            if uncolored_mask == 0:
                return True
            
            # Find vertex with MRV (Minimum Remaining Values) among uncolored
            best_v = -1
            best_avail = 0
            best_count = 4
            
            m = uncolored_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                avail = get_available_colors(v, color_masks)
                cnt = bin(avail).count('1')
                if cnt == 0:
                    return False
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    best_avail = avail
                    if cnt == 1:
                        break
                m ^= lsb
            
            # Assign best_v
            v = best_v
            avail = best_avail
            bit_v = 1 << v
            
            # Try colors in avail
            a = avail
            while a:
                lsb = a & -a
                c = lsb.bit_length() - 1
                # Assign c to v
                color_masks[c] |= bit_v
                
                if backtrack(uncolored_mask ^ bit_v):
                    return True
                
                # Unassign
                color_masks[c] &= ~bit_v
                a ^= lsb
            
            return False
        
        return backtrack(full_mask)
    
    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def make_groetzsch():
        """Create the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
        adj = [0] * n
        # Original C5: vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        
        # Mycielski: add vertices 5-9 as copies of 0-4, and vertex 10 as universal to copies
        for i in range(5):
            # Copy i is vertex 5+i
            # Connect 5+i to neighbors of i in original
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            # Connect 5+i to vertex 10
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        return adj
    
    def greedy_fill(adj, deadline, seed):
        """Greedy fill: try to add edges in random order, keep if triangle-free and non-3-colorable."""
        random.seed(seed)
        cur_adj = adj[:]
        
        # All possible edges not yet in graph
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n) 
                     if not (cur_adj[i] & (1 << j))]
        random.shuffle(candidates)
        
        # Greedy: add edge if it doesn't create a triangle
        for u, v in candidates:
            if time.time() > deadline:
                break
            common = cur_adj[u] & cur_adj[v]
            if common == 0:
                cur_adj[u] |= (1 << v)
                cur_adj[v] |= (1 << u)
        
        return cur_adj
    
    def local_search(adj, deadline, seed):
        """Local search to maximize edges while maintaining triangle-free and non-3-colorable."""
        random.seed(seed)
        cur_adj = adj[:]
        current_count = count_edges(cur_adj)
        best_adj = cur_adj[:]
        best_count = current_count
        
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Try to add a random edge
            all_missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                          if not (cur_adj[i] & (1 << j))]
            
            if not all_missing:
                break
            
            random.shuffle(all_missing)
            improved = False
            
            for u, v in all_missing[:200]:  # Try up to 200 random edges
                # Check if adding (u,v) keeps triangle-free
                common = cur_adj[u] & cur_adj[v]
                if common:
                    continue
                
                # Add the edge
                cur_adj[u] |= (1 << v)
                cur_adj[v] |= (1 << u)
                
                # Check if still not 3-colorable
                if not is_3_colorable(cur_adj):
                    new_count = count_edges(cur_adj)
                    if new_count > current_count:
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = cur_adj[:]
                        improved = True
                        break
                    else:
                        # Roll back
                        cur_adj[u] &= ~(1 << v)
                        cur_adj[v] &= ~(1 << u)
                else:
                    # Roll back
                    cur_adj[u] &= ~(1 << v)
                    cur_adj[v] &= ~(1 << u)
            
            if not improved:
                no_improve_count += 1
                # Try removing an edge and adding two others (2-opt style)
                if no_improve_count % 5 == 0:
                    existing_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                                    if cur_adj[i] & (1 << j)]
                    if existing_edges:
                        random.shuffle(existing_edges)
                        for u, v in existing_edges[:30]:
                            # Remove (u,v)
                            cur_adj[u] &= ~(1 << v)
                            cur_adj[v] &= ~(1 << u)
                            
                            # Try to add a different edge
                            other_missing = [(a, b) for a in range(n) for b in range(a + 1, n) 
                                          if not (cur_adj[a] & (1 << b))]
                            random.shuffle(other_missing)
                            found = False
                            for a, b in other_missing[:100]:
                                common = cur_adj[a] & cur_adj[b]
                                if common:
                                    continue
                                cur_adj[a] |= (1 << b)
                                cur_adj[b] |= (1 << a)
                                if not is_3_colorable(cur_adj):
                                    new_count = count_edges(cur_adj)
                                    if new_count > current_count:
                                        current_count = new_count
                                        if current_count > best_count:
                                            best_count = current_count
                                            best_adj = cur_adj[:]
                                        found = True
                                        break
                                    else:
                                        cur_adj[a] &= ~(1 << b)
                                        cur_adj[b] &= ~(1 << a)
                                else:
                                    cur_adj[a] &= ~(1 << b)
                                    cur_adj[b] &= ~(1 << a)
                            
                            if not found:
                                # Restore removed edge
                                cur_adj[u] |= (1 << v)
                                cur_adj[v] |= (1 << u)
                            else:
                                no_improve_count = 0
                                break
            else:
                no_improve_count = 0
        
        return best_adj, best_count
    
    best_adj = None
    best_count = 0
    
    # Phase 1: Start with Grötzsch graph and do multiple restarts
    groetzsch = make_groetzsch()
    
    # Verify Grötzsch is valid
    if is_triangle_free(groetzsch) and not is_3_colorable(groetzsch):
        best_adj = groetzsch[:]
        best_count = count_edges(groetzsch)
    
    # Phase 2: Multiple restarts with greedy fill + local search
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Start from Grötzsch or a perturbed version
        if restart < 50:
            start_adj = groetzsch[:]
        else:
            # Perturb best_adj
            start_adj = best_adj[:] if best_adj is not None else groetzsch[:]
            existing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                       if start_adj[i] & (1 << j)]
            random.shuffle(existing)
            num_remove = random.randint(1, 3)
            for u, v in existing[:num_remove]:
                start_adj[u] &= ~(1 << v)
                start_adj[v] &= ~(1 << u)
        
        # Greedy fill
        filled_adj = greedy_fill(start_adj, deadline, seed=restart * 7 + 13)
        
        # Check if still non-3-colorable
        if not is_3_colorable(filled_adj):
            cnt = count_edges(filled_adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = filled_adj[:]
        
        # Local search from this solution
        if time.time() < deadline:
            ls_adj, ls_count = local_search(filled_adj, deadline, seed=restart * 7 + 17)
            if ls_count > best_count:
                best_count = ls_count
                best_adj = ls_adj
    
    # Phase 3: More aggressive search with different initial structures
    if time.time() < deadline:
        # Try starting from random triangle-free graphs
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Generate random triangle-free graph
            adj = [0] * n
            random_edges = all_edges[:]
            random.shuffle(random_edges)
            for u, v in random_edges:
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if non-3-colorable
            if not is_3_colorable(adj):
                cnt = count_edges(adj)
                if cnt > best_count:
                    best_count = cnt
                    best_adj = adj[:]
                
                # Local search
                if time.time() < deadline:
                    ls_adj, ls_count = local_search(adj, deadline, seed=restart * 11 + 23)
                    if ls_count > best_count:
                        best_count = ls_count
                        best_adj = ls_adj
    
    if best_adj is None:
        # Absolute fallback: return the Grötzsch graph on 11 vertices + 5 isolated
        adj = make_groetzsch()
        return adj_to_edges(adj)
    
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
