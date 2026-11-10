import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n = 16
    
    # Precompute pairs
    pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    num_pairs = len(pairs)
    
    # Bitmask utilities
    def get_bitmask(adj):
        return adj
        
    def is_triangle_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        return True
        
    def is_3_colorable(adj):
        # Fast 3-colorability check using backtracking with bitsets
        # Order vertices by degree descending to prune early
        degrees = [bin(a).count('1') for a in adj]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Find forbidden colors
            forbidden = 0
            for u in range(n):
                if (adj[v] >> u) & 1:
                    if colors[u] != -1:
                        forbidden |= (1 << colors[u])
            
            # Try available colors
            for c in range(3):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def try_add_edge(adj, u, v):
        """Returns True if edge (u,v) can be added while keeping triangle-free."""
        if (adj[u] >> v) & 1:
            return False
        if adj[u] & adj[v]:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Strategy:
    # 1. Generate random 4-chromatic triangle-free "seeds" using Mycielski-like structures or random dense graphs.
    # 2. Use a "greedy fill" to add any edge that doesn't create a triangle.
    # 3. Use "swap" local search to increase density: remove an edge, try to add two new ones.
    # 4. Crucially, ensure the graph remains 4-chromatic (not 3-colorable).
    
    best_edges_count = 0
    best_edge_list = []
    
    # Helper: Create a base 4-chromatic triangle-free graph
    # The Grötzsch graph is 11 vertices, 20 edges, 4-chromatic, triangle-free.
    # We can use this as a core and add 5 more vertices, connecting them to maximize edges.
    
    def create_groetzsch_core():
        # Vertices 0-4: Cycle C5
        # Vertices 5-9: Copies of 0-4, connected to i-1, i+1
        # Vertex 10: Connected to 5-9
        adj = [0] * n
        
        # Cycle 0-4
        for i in range(5):
            u, v = i, (i+1)%5
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Copies 5-9
        for i in range(5):
            u = 5 + i
            v1 = (i-1)%5
            v2 = (i+1)%5
            adj[u] |= (1 << v1)
            adj[v1] |= (1 << u)
            adj[u] |= (1 << v2)
            adj[v2] |= (1 << u)
            
        # Top vertex 10
        for i in range(5):
            u = 10
            v = 5 + i
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        return adj

    def random_seed():
        # Mix of Grötzsch core and random dense triangle-free
        if random.random() < 0.5:
            return create_groetzsch_core()
        else:
            adj = [0] * n
            # Random triangle-free generation
            perm = pairs[:]
            random.shuffle(perm)
            for u, v in perm:
                if not ((adj[u] >> v) & 1) and not (adj[u] & adj[v]):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            return adj

    def greedy_fill(adj):
        """Add all possible edges that maintain triangle-free property."""
        changed = True
        while changed:
            changed = False
            for u, v in pairs:
                if not ((adj[u] >> v) & 1):
                    if not (adj[u] & adj[v]):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        changed = True
        return adj

    def local_search(adj, time_limit=10.0):
        """
        Local search to maximize edges while maintaining:
        1. Triangle-free
        2. Not 3-colorable
        """
        current_adj = adj[:]
        current_edges = edge_count(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # If not 4-chromatic, we must modify it to be so, or it's invalid.
        # The task requires the returned graph to be NOT 3-colorable.
        # If the seed is 3-colorable, we can't just return it.
        # We need to force 4-chromaticity.
        
        if is_3_colorable(current_adj):
            # We need to make it 4-chromatic.
            # This is hard. Better to only start with 4-chromatic seeds.
            # Or, if we start with 3-colorable, we might need to add edges that break 3-colorability.
            # But adding edges to a 3-colorable graph might keep it 3-colorable.
            # So, if seed is 3-colorable, we discard or try to fix.
            # Let's just discard 3-colorable seeds in the main loop to save time.
            return current_adj, current_edges

        start_ls = time.time()
        no_improve_count = 0
        
        while time.time() - start_ls < time_limit:
            improved = False
            
            # Pick a random edge to remove
            edges_list = []
            for u in range(n):
                for v in range(u+1, n):
                    if (current_adj[u] >> v) & 1:
                        edges_list.append((u, v))
                        
            if not edges_list:
                break
                
            u, v = random.choice(edges_list)
            
            # Remove edge
            remove_edge(current_adj, u, v)
            current_edges -= 1
            
            # Try to add 2 new edges
            # Candidates for u: w such that (u,w) is valid
            # Candidates for v: x such that (v,x) is valid
            
            pot_u = []
            for w in range(n):
                if w != u and w != v and not ((current_adj[u] >> w) & 1):
                    if not (current_adj[u] & current_adj[w]):
                        pot_u.append(w)
                        
            pot_v = []
            for x in range(n):
                if x != v and x != u and not ((current_adj[v] >> x) & 1):
                    if not (current_adj[v] & current_adj[x]):
                        pot_v.append(x)
                        
            if pot_u and pot_v:
                # Shuffle to randomize
                random.shuffle(pot_u)
                random.shuffle(pot_v)
                
                # Try a few combinations
                found = False
                for w in pot_u[:10]:
                    for x in pot_v[:10]:
                        if w == x:
                            continue
                            
                        # Add (u, w) and (v, x)
                        # Check if this addition is valid (triangle-free)
                        # We already checked (u,w) and (v,x) against current_adj.
                        # But we need to check if adding (u,w) affects (v,x) validity?
                        # No, (v,x) validity depends on common neighbors of v and x.
                        # Adding (u,w) changes adj[u] and adj[w].
                        # It doesn't change adj[v] or adj[x] directly, unless u==v or w==x etc.
                        # But u!=v and w!=x are guaranteed.
                        # However, adding (u,w) might create a triangle involving v?
                        # Triangle (u, w, z) where z is common neighbor. We checked common neighbors of u,w.
                        # Triangle (v, x, z) where z is common neighbor. We checked common neighbors of v,x.
                        # Triangle (u, v, z)? u,v not connected (we removed it).
                        # Triangle (u, x, z)?
                        # Triangle (v, w, z)?
                        # The local check `not (current_adj[u] & current_adj[w])` ensures no common neighbor in CURRENT adj.
                        # Since we are adding (u,w) and (v,x) simultaneously, we must ensure they don't interfere.
                        # Actually, the standard triangle-free check is: for any edge (a,b), N(a) ∩ N(b) = ∅.
                        # If we add (u,w), we need N(u) ∩ N(w) = ∅ in the NEW graph.
                        # N(u) in new graph = N(u)_old ∪ {w}.
                        # N(w) in new graph = N(w)_old ∪ {u}.
                        # Intersection: (N(u)_old ∪ {w}) ∩ (N(w)_old ∪ {u})
                        # = (N(u)_old ∩ N(w)_old) ∪ (N(u)_old ∩ {u}) ∪ ({w} ∩ N(w)_old) ∪ ({w} ∩ {u})
                        # N(u)_old ∩ N(w)_old was 0.
                        # N(u)_old ∩ {u} is 0 (no self loops).
                        # {w} ∩ N(w)_old is 0.
                        # {w} ∩ {u} is 0 since u!=w.
                        # So (u,w) is safe.
                        
                        # Similarly for (v,x).
                        
                        # But do (u,w) and (v,x) create triangles with each other?
                        # E.g. if u=x? No, u!=x.
                        # If w=v? No, w!=v.
                        # If u is neighbor of x?
                        # If we add (u,w) and (v,x), and u is already neighbor of x?
                        # Then we have path u-x-v... no triangle.
                        # Triangle would require 3 mutually connected vertices.
                        # New edges are (u,w) and (v,x).
                        # Possible triangles involving new edges:
                        # 1. (u, w, z) -> checked.
                        # 2. (v, x, z) -> checked.
                        # 3. (u, v, w) -> u-v not connected.
                        # 4. (u, v, x) -> u-v not connected.
                        # 5. (u, w, v) -> u-v not connected.
                        # 6. (u, w, x) -> Is u connected to x? Is w connected to x?
                        #    If u-x and w-x are edges, then (u, w, x) is a triangle IF u-w is edge.
                        #    We are adding u-w. So if u-x and w-x exist, we create a triangle!
                        #    We must check this.
                        
                        # Check for triangle (u, w, x):
                        # u-w is new.
                        # Need u-x and w-x to be edges.
                        if (current_adj[u] >> x) & 1 and (current_adj[w] >> x) & 1:
                            continue
                            
                        # Check for triangle (u, v, w) - wait, u-v is removed.
                        # Check for triangle (v, x, u):
                        # v-x is new.
                        # Need v-u and x-u to be edges. v-u is removed.
                        
                        # Check for triangle (v, x, w):
                        # v-x is new.
                        # Need v-w and x-w to be edges.
                        if (current_adj[v] >> w) & 1 and (current_adj[x] >> w) & 1:
                            continue
                            
                        # If we pass these checks, the graph is triangle-free.
                        
                        # Apply changes
                        current_adj[u] |= (1 << w)
                        current_adj[w] |= (1 << u)
                        current_adj[v] |= (1 << x)
                        current_adj[x] |= (1 << v)
                        
                        new_edges = current_edges + 2
                        
                        # Check 3-colorability
                        if not is_3_colorable(current_adj):
                            if new_edges > best_edges:
                                best_edges = new_edges
                                best_adj = current_adj[:]
                                improved = True
                                found = True
                                break
                        else:
                            # Rollback
                            remove_edge(current_adj, u, w)
                            remove_edge(current_adj, v, x)
                            current_edges -= 2
                            continue
                            
                        # Rollback if not improved or just to try next?
                        # If we found an improvement, we keep it and break.
                        # If we didn't find an improvement (still 3-colorable or fewer edges), rollback.
                        if not found:
                            remove_edge(current_adj, u, w)
                            remove_edge(current_adj, v, x)
                            current_edges -= 2
                            
                    if found:
                        break
            else:
                # No candidates, rollback removal
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                continue
            
            if not improved:
                # Rollback the initial removal if we didn't add anything
                # Note: if we added and rolled back, current_adj is back to state after removal.
                # We need to restore the removed edge (u,v).
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
                no_improve_count += 1
                if no_improve_count > 100:
                    break
            else:
                no_improve_count = 0
                current_edges = best_edges
                # current_adj is already updated to best_adj state?
                # In the loop, if found, we break with current_adj modified.
                # So current_adj is the new best.
                
        return best_adj, best_edges

    # Main Loop
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        adj = random_seed()
        
        # Greedy fill
        adj = greedy_fill(adj)
        
        # Check if 4-chromatic
        if is_3_colorable(adj):
            continue
            
        edge_count = edge_count(adj)
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_edge_list = get_edge_list(adj)
            
        # Local search
        # Allocate more time to promising candidates
        time_remaining = deadline - time.time()
        if time_remaining < 5.0:
            break
            
        # Spend up to 5 seconds per local search
        ls_time = min(5.0, time_remaining / 2.0)
        adj, improved_count = local_search(adj, time_limit=ls_time)
        
        if improved_count > best_edges_count:
            best_edges_count = improved_count
            best_edge_list = get_edge_list(adj)

    if not best_edge_list:
        # Fallback: Grötzsch graph + isolated vertices
        adj = create_groetzsch_core()
        # Greedy fill on the rest?
        # The Grötzsch core is maximal in its own right? No, vertices 11-15 are isolated.
        # We can connect 11-15 to each other and to the core if possible.
        adj = greedy_fill(adj)
        if is_3_colorable(adj):
            # If greedy fill made it 3-colorable (unlikely for Groetzsch core), revert?
            # Groetzsch core is 4-chromatic. Adding edges usually preserves or increases chromatic number?
            # No, adding edges can make it harder to color, but if it was 4-chromatic, it stays >=4.
            # Wait, adding edges to a 4-chromatic graph keeps it 4-chromatic (or higher).
            # So if we start with Groetzsch, it is 4-chromatic.
            # Greedy fill adds edges. It remains 4-chromatic.
            # So this should be valid.
            pass
        best_edge_list = get_edge_list(adj)
        best_edges_count = edge_count(adj)

    return best_edge_list

# EVOLVE-BLOCK-END
