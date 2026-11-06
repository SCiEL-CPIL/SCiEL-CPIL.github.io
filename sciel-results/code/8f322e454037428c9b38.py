import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    n = 16
    # adj_mask[i] is a bitmask of neighbors of i
    # We will work with bitmasks for speed
    
    # 1. Initialize Grötzsch Core
    # Vertices 0-10
    # 0-4: Cycle
    # 5-9: Shadows
    # 10: Top
    
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)
        
    # Shadows
    # Edge (u, v) in cycle implies (5+u, v) and (5+v, u)
    cycle_edges = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle_edges:
        su = 5 + u
        sv = 5 + v
        add_edge(su, v)
        add_edge(sv, u)
        
    # Top 10 to 5-9
    for i in range(5):
        add_edge(10, 5 + i)
        
    # Core edges are fixed. We cannot remove them.
    core_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_mask[i] & (1 << j):
                core_edges.append((i, j))
                
    core_set = set(core_edges)
    
    # Potential edges: All edges not in core
    potential_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in core_set:
                potential_edges.append((i, j))
                
    def is_triangle_free_add(u, v, masks):
        # Check if u and v have common neighbors
        return (masks[u] & masks[v]) == 0
        
    def count_edges(masks):
        count = 0
        for i in range(n):
            count += bin(masks[i]).count('1')
        return count // 2
        
    best_edges = []
    best_count = 0
    best_masks = [0]*n
    
    # We will run multiple iterations of Greedy + Local Search
    # To speed up, we represent the graph as a list of bitmasks.
    
    # Helper to convert masks to edge list
    def masks_to_edges(masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if masks[i] & (1 << j):
                    edges.append([i, j])
        return edges
        
    # Initial greedy run to get a baseline
    # We will use random restarts
    
    num_restarts = 10000 # Will be limited by time
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        masks = [0] * n
        # Copy core
        for u, v in core_edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
        # Greedy Phase
        # Shuffle potential edges
        shuffled = potential_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if is_triangle_free_add(u, v, masks):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
        # Local Search Phase
        # Try to improve by swapping edges
        # We only consider non-core edges for removal
        current_edges_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if masks[i] & (1 << j):
                    current_edges_set.add((i, j))
                    
        improved = True
        while improved:
            improved = False
            # Try removing each non-core edge
            # To make this faster, we can shuffle the order of removal attempts
            non_core_in_graph = [e for e in current_edges_set if e not in core_set]
            random.shuffle(non_core_in_graph)
            
            for e in non_core_in_graph:
                u, v = e
                # Remove e
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                current_edges_set.remove(e)
                
                # Try to add any blocked edge
                added_any = False
                # We need to check which edges are now valid
                # An edge (a,b) is valid if (masks[a] & masks[b]) == 0
                # And (a,b) is not in current_edges_set
                # And (a,b) is not in core_set (core is already in)
                
                # To be efficient, we can iterate through potential_edges
                # But that's O(E).
                # Better: Check edges that were blocked by u or v?
                # Actually, removing (u,v) only unblocks edges (x,y) where
                # x was neighbor of u and y was neighbor of v? No.
                # Triangle check for (a,b) fails if there is a common neighbor.
                # Removing (u,v) changes the neighbor sets of u and v.
                # It doesn't change the neighbor sets of other vertices.
                # So only edges involving u or v can have their validity changed?
                # No.
                # Triangle (a,b,c) is formed if a-b, b-c, c-a exist.
                # If we remove u-v, it breaks any triangle containing u-v.
                # It does not break triangles not containing u-v.
                # So an edge (a,b) that was invalid because of a common neighbor c
                # is still invalid if c is not u or v?
                # Wait.
                # If (a,b) was invalid, it means there existed c such that (a,c) and (b,c) were edges.
                # Removing (u,v) only affects the edge set.
                # If c != u and c != v, the edges (a,c) and (b,c) are still there.
                # So (a,b) is still invalid.
                # If c == u, then (a,u) and (b,u) were edges.
                # We removed (u,v). This doesn't remove (a,u) or (b,u).
                # So (a,b) is still invalid via u.
                # Wait, if we remove (u,v), does it make (a,b) valid?
                # Only if the *only* common neighbor was... wait.
                # Common neighbor c means (a,c) and (b,c) exist.
                # Removing (u,v) doesn't remove (a,c) or (b,c).
                # So removing an edge (u,v) NEVER makes another edge (a,b) valid?
                # Let's re-verify.
                # Triangle-free condition: No 3-cliques.
                # Adding (a,b) creates a triangle if N(a) intersects N(b).
                # If (a,b) was not added, it's because N(a) & N(b) != 0.
                # Removing (u,v) removes u from N(v) and v from N(u).
                # It does NOT change N(a) or N(b) unless a or b is u or v.
                # Case 1: a=u, b=v. We just removed it.
                # Case 2: a=u, b=w.
                #   N(u) changed (lost v). N(w) unchanged.
                #   Intersection N(u) & N(w) might have shrunk.
                #   If v was in N(u) & N(w), and v is the ONLY common neighbor, then (u,w) becomes valid.
                #   So yes, removing (u,v) can make (u,w) valid if v was the only common neighbor of u and w.
                #   Similarly, it can make (v,w) valid.
                #   It cannot make (a,b) valid if a,b are not u or v.
                
                # So we only need to check edges incident to u or v.
                candidates = []
                for w in range(n):
                    if w != u and w != v:
                        if not (masks[u] & (1 << w)) and (u, w) not in current_edges_set and (u, w) not in core_set:
                            # Check if (u,w) is now valid
                            if (masks[u] & masks[w]) == 0:
                                candidates.append((u, w))
                        if not (masks[v] & (1 << w)) and (v, w) not in current_edges_set and (v, w) not in core_set:
                            if (masks[v] & masks[w]) == 0:
                                candidates.append((v, w))
                
                # Add as many candidates as possible
                # Sort candidates by some heuristic? Or just shuffle?
                random.shuffle(candidates)
                for a, b in candidates:
                    if (masks[a] & masks[b]) == 0:
                        masks[a] |= (1 << b)
                        masks[b] |= (1 << a)
                        current_edges_set.add((a, b))
                        added_any = True
                        
                if added_any:
                    # We improved
                    improved = True
                    # We don't break immediately, we continue checking other edges?
                    # Or restart the local search loop?
                    # Restarting is safer to avoid local minima.
                    break 
                else:
                    # Restore e
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    current_edges_set.add(e)
                    
        # Update best
        count = count_edges(masks)
        if count > best_count:
            best_count = count
            best_masks = masks[:]
            
    return masks_to_edges(best_masks)
