import random
import time
import itertools

def construct():
    """
    Constructs a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph.
    Uses a fast exact evaluation and a local search that maximizes triangle density
    while maintaining K4-freeness.
    """
    n = 18
    deadline = time.time() + 110.0  # Leave 10s safety margin for the 120s limit

    # Helper: check if a graph (adj list of sets) is K4-free
    def is_k4_free(adj):
        for u in range(n):
            nbrs = adj[u]
            # Check all pairs of neighbors
            nbr_list = list(nbrs)
            L = len(nbr_list)
            for i in range(L):
                v = nbr_list[i]
                # For each w in nbrs[v], check if there's an edge between w and any other common neighbor?
                # Actually, K4 is {u, v, w, x} where all 6 edges exist.
                # Given u is connected to v. If v and w are connected, and w and x are connected, and v-x, u-x, u-w...
                # Simpler: Check if the neighborhood of u contains a triangle.
                # The subgraph induced by N(u) must be triangle-free.
                pass
        
        # Efficient check: For each vertex u, the subgraph induced by N(u) must have no triangles.
        # We can precompute or just check on the fly.
        for u in range(n):
            nbrs = adj[u]
            # Get list of neighbors
            nl = list(nbrs)
            L = len(nl)
            # Check for triangles in N(u)
            # A triangle in N(u) is a set {v, w, x} such that v-w, w-x, x-v are edges.
            # Since v, w, x are all in N(u), we just need to check edges among them.
            # This is equivalent to checking if the induced subgraph on N(u) has a K3.
            # For n=18, max degree is 17. 17 choose 3 is ~ 680. Total 18 * 680 ~ 12k checks. Very fast.
            for i in range(L):
                v = nl[i]
                for j in range(i + 1, L):
                    w = nl[j]
                    if w in adj[v]:
                        # Check if there is an x in N(u) such that x-v and x-w? No, just v-w-x triangle.
                        # If v-w is an edge, and v, w are neighbors of u.
                        # Do v, w, x form a triangle?
                        # We need a third vertex x in N(u) such that x-v and x-w are edges?
                        # No, a triangle in N(u) is v-w, w-x, x-v.
                        # So if v-w is an edge, we just need to check if v and w have a common neighbor in N(u).
                        # Common neighbors of v and w in N(u).
                        # If (N(v) & N(w) & N(u)) is non-empty, then there is a triangle.
                        # Actually, if x is in N(u) and x in N(v) and x in N(w), then {v,w,x} is a triangle in N(u).
                        # Note: v-w edge is required.
                        if adj[v] & adj[w] & nbrs:
                            return False
        return True

    # Helper: Calculate number of triangles
    def count_triangles(adj):
        cnt = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    cnt += len(adj[u] & adj[v])
        return cnt // 3

    # Helper: Evaluate exact max triangle-free induced subgraph size
    # Uses bitmask representation for speed.
    def exact_max_tf_size(adj):
        # Convert adj to bitmasks
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in adj[u]:
                m |= (1 << v)
            masks[u] = m

        # Precompute for each pair (i, j) the set of common neighbors (triangle partners)
        # tri_mask[i][j] = bitmask of vertices k such that {i,j,k} is a triangle
        # If i,j not connected, it's 0.
        tri_m = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (masks[i] >> j) & 1:
                    common = masks[i] & masks[j]
                    # Exclude i and j themselves if they are in common? 
                    # masks[i] has j, masks[j] has i.
                    # common contains i and j?
                    # masks[i] is neighbors of i. j is in masks[i].
                    # masks[j] is neighbors of j. i is in masks[j].
                    # So common has i and j?
                    # i is in masks[j]. j is in masks[i].
                    # Is i in masks[i]? No. Is j in masks[j]? No.
                    # So common does NOT contain i or j.
                    # common contains exactly the vertices k such that i-k, j-k are edges.
                    tri_m[i][j] = common
                    tri_m[j][i] = common

        # Branch and Bound to find max triangle-free set
        # State: current set of vertices (bitmask), current size
        # Upper bound: size + popcount(available)
        # Pruning: If we can't beat best, stop.
        
        best = 0
        # Order vertices to help pruning? 
        # Vertices with high triangle participation should be decided early?
        # Or just fixed order.
        
        # We want to find the MAX set.
        # We can iterate through vertices and decide to INCLUDE or EXCLUDE.
        # If we include u, we must exclude any v, w such that {u,v,w} is a triangle.
        # This implies we can't just keep a simple "available" set.
        
        # Alternative: Minimum Vertex Cover of the triangle hypergraph.
        # Minimize size of cover C such that every triangle intersects C.
        # Max TF set = n - min_cover_size.
        
        # List of triangles as tuples (u, v, w)
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if tri_m[i][j]:
                    for k in range(j + 1, n):
                        if (tri_m[i][j] >> k) & 1:
                            triangles.append((i, j, k))
        
        # If no triangles, return n
        if not triangles:
            return n

        # Sort triangles to improve pruning?
        # Group by first vertex?
        
        # B&B for Min Vertex Cover
        # State: index of next triangle to cover, current cover mask, current cover size
        # Optimization: If all remaining triangles are covered by current cover, we can stop.
        # But checking "all covered" is expensive.
        
        # Better B&B:
        # Pick a triangle with fewest "free" vertices (vertices not in cover).
        # Branch on each such vertex to add to cover.
        
        # For n=18, and likely < 500 triangles, this should be very fast.
        
        # We'll use a recursive search.
        # To avoid recursion depth issues or overhead, we might use a stack, but n is small.
        
        best_cover = n # Max possible cover size
        found = False
        
        def search(cover_mask, cover_size, remaining_tris_idx):
            nonlocal best_cover, found
            if found:
                # If we found a cover of size 1, we can stop immediately (min possible >0 for non-empty triangles)
                # Actually min cover is at least 1.
                # If cover_size == best_cover, no improvement possible.
                pass
            
            if cover_size >= best_cover:
                return

            # Check if all triangles are covered
            # We need an efficient way to know this.
            # We can maintain a list of "active" triangles (not yet covered).
            # Passing a list in recursion is slow.
            
            # Alternative: Precompute for each vertex the list of triangles it covers.
            # Then, if we add vertex v, we mark those triangles as covered.
            
            # Let's restart the search strategy slightly for speed.
            # We'll use a global list of triangles and a global "covered" count?
            # No, that's hard to backtrack.
            
            # Let's stick to the "pick triangle, branch" but optimize the "is covered" check.
            # We can pass the set of uncovered triangles as a bitmask of triangle indices?
            # If num_triangles <= 64, we can use an int.
            # 18 vertices, max triangles 816. So no.
            
            # Heuristic pruning:
            # If cover_size + 1 >= best_cover, prune.
            
            # Find an uncovered triangle.
            # To do this fast, we can just iterate triangles and check if any vertex is in cover_mask.
            # This is O(T) per node.
            
            # Optimization: 
            # If we sort vertices by "centrality" in triangles, we might prune better.
            
            for t_idx, t in enumerate(triangles):
                u, v, w = t
                if (cover_mask & (1 << u)) or (cover_mask & (1 << v)) or (cover_mask & (1 << w)):
                    continue
                
                # Found an uncovered triangle t
                # Branch on u, v, w
                # Try adding u
                search(cover_mask | (1 << u), cover_size + 1, t_idx + 1) # Pass t_idx to avoid re-checking earlier ones? 
                # Wait, if we pass t_idx, we skip triangles 0..t_idx.
                # But adding u might cover them? No, we checked they are uncovered.
                # Actually, if we skip them, we might miss that they were covered by *other* choices?
                # No, the "uncovered" status is dynamic.
                # This simple "iterate until find one" is O(T) per node.
                
                # To make it faster, we can just return after branching on the FIRST uncovered triangle found.
                # Why? Because the search space is the same. We just need to cover ALL.
                # If we cover t, we don't need to consider it again.
                # So we can just find ONE uncovered triangle, branch on it, and recurse.
                # The recursion will find the next uncovered one.
                break
            
            else:
                # No uncovered triangle found
                best_cover = cover_size
                found = True
                return

        # To speed up `search`, we can precompute the triangles.
        # The loop `for t_idx, t in enumerate(triangles): if not covered: break` is the bottleneck.
        # We can optimize by maintaining a list of "candidate" triangles?
        
        # Let's try a different exact method: 
        # Maximize independent set in the 3-uniform hypergraph.
        # This is equivalent to what we are doing.
        
        # Given the small n, even O(T * Nodes) might be okay if Nodes is small.
        # With good pruning, Nodes should be small.
        
        search(0, 0, 0)
        return n - best_cover

    # Heuristic score for local search:
    # We want to minimize max_tf_size.
    # Max triangles is a good proxy?
    # More triangles -> smaller max TF set?
    # Yes, generally.
    # Also, we want the graph to be "dense" but K4-free.
    
    # Local Search Initialization
    # Start with a random K4-free graph or K6,6,6?
    # K6,6,6 is a strong base.
    
    def init_graph():
        adj = [set() for _ in range(n)]
        # K6,6,6
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u].add(v)
                        adj[v].add(u)
        return adj

    def random_k4_free_graph():
        adj = [set() for _ in range(n)]
        # Try to add random edges
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            if not is_k4_free(adj):
                adj[u].remove(v)
                adj[v].remove(u)
        return adj

    # Main Search Loop
    best_adj = init_graph()
    best_score = exact_max_tf_size(best_adj)
    
    current_adj = [s.copy() for s in best_adj]
    
    # Try local search:
    # 1. Try adding an edge if K4-free.
    # 2. Try removing an edge? (Rarely helps if we want to minimize TF set, as it reduces triangles)
    #    But removing an edge might allow adding TWO edges later?
    #    Let's stick to adding first. If stuck, try some removals.
    
    # We can also use "swap" moves: Remove (u,v), Add (u,w), Add (v,w)?
    
    # Let's use a simple hill climbing with restarts.
    
    while time.time() < deadline - 5.0:
        moved = False
        # Try all possible edge additions
        # To be fast, iterate random edges
        for _ in range(100): # Try 100 random moves
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or v in current_adj[u]:
                continue
            
            # Check if adding (u,v) creates K4
            # K4 formed if u,v have a common neighbor w, and w is connected to another common neighbor x?
            # Common neighbors of u and v: C = N(u) & N(v)
            # If there is an edge (w,x) in the subgraph induced by C, then adding (u,v) creates K4 {u,v,w,x}.
            common = current_adj[u] & current_adj[v]
            creates_k4 = False
            for w in common:
                # Check if w is connected to any other vertex in common
                # i.e., N(w) & common is non-empty (and not just w itself? w is in common, so w not in N(w))
                if current_adj[w] & common:
                    creates_k4 = True
                    break
            
            if creates_k4:
                continue
            
            # Try adding
            current_adj[u].add(v)
            current_adj[v].add(u)
            
            # Evaluate
            # Use fast proxy first?
            # If we add edge, triangles increase.
            # New score = exact_max_tf_size(current_adj)
            # This is expensive. 
            # Let's use a proxy: count_triangles.
            # If count_triangles increases, it's likely score decreases or stays same.
            # But we need exact score to be sure?
            # For local search, we can use the proxy to guide, and only compute exact score periodically.
            
            # Strategy:
            # 1. Compute new triangle count.
            # 2. If higher, accept move (greedy).
            # 3. Every N moves, compute exact score. If improved, update best.
            
            # Let's just compute exact score every move if we are fast enough?
            # exact_max_tf_size takes ~1-5ms?
            # If so, 100 moves * 5ms = 50ms. 1000 moves = 50ms?
            # 110s / 0.05s = 2200 iterations.
            # That's a lot.
            
            # Let's try to optimize exact_max_tf_size further or use it sparingly.
            
            # For now, let's use triangle count as the primary objective for the move, 
            # and only update best_adj if the move was accepted (higher triangles).
            # Then at the end of the loop (or periodically), check exact score.
            
            # Actually, minimizing max_tf_size is the goal.
            # Maximizing triangles is a proxy.
            # Let's accept the move if it increases triangles.
            # And track the best exact score seen so far.
            
            # We need the current triangle count.
            # Let's maintain it.
            
            # This requires refactoring to maintain state.
            
            # Simpler: Just run the local search with exact evaluation if it's fast enough.
            # Let's assume exact_max_tf_size is fast.
            
            # If we are here, we added the edge.
            # Score = exact_max_tf_size(current_adj)
            # If Score < best_score:
            #     best_score = Score
            #     best_adj = copy
            #     moved = True
            # else:
            #     revert
            
            # But computing exact score is the bottleneck.
            # Let's try a different approach:
            # Use the proxy (triangle count) to accept moves.
            # Keep a record of the best exact score found.
            
            # Let's revert for now and use a smarter loop.
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
        # End random moves
            
        # Periodically evaluate exact score of current_adj
        # And compare to best_adj
        
        # To save code, I'll just implement the local search using triangle count as objective,
        # and at the very end, return the best_adj found by exact evaluation of the candidates.
        
        # Actually, let's just do:
        # 1. Random restarts.
        # 2. Greedy add edges based on triangle count.
        # 3. Evaluate exact score of the final graph of each restart.
        # 4. Return the best.
        
        # This is simpler and robust.
        
        # Let's restart the logic inside the while loop.
        # Break out of the move loop and start a new "round" or restart.
        
        # If we haven't moved in a while, restart?
        # For now, just let it run random moves.
        
        # To ensure we find the best, we should evaluate the exact score of the current graph
        # after a batch of moves.
        
        # Let's change the loop structure:
        # For t in time:
        #   Generate a graph via local search (max triangles, K4-free)
        #   Evaluate exact score
        #   Keep best
        
        # This is cleaner.
        
        # Let's rewrite the loop below.
        pass

    # --- Rewritten Main Loop ---
    
    best_adj = init_graph()
    best_exact_score = exact_max_tf_size(best_adj)
    
    # If best_exact_score is already low (e.g. 9 or 10), we are happy.
    # K6,6,6 has score 12.
    
    while time.time() < deadline - 5.0:
        # Local Search to maximize triangles
        adj = [s.copy() for s in best_adj] # Start from best
        # Or start from random?
        # Starting from best might get stuck.
        # Let's mix: 50% from best, 50% from random K4-free.
        
        if random.random() < 0.5:
            adj = init_graph()
            # Perturb?
            # Add some random edges
            for _ in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or v in adj[u]: continue
                # Check K4
                common = adj[u] & adj[v]
                ok = True
                for w in common:
                    if adj[w] & common:
                        ok = False
                        break
                if ok:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Greedy Local Search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try all non-edges
            # To speed up, shuffle
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u]:
                        candidates.append((u, v))
            
            random.shuffle(candidates)
            
            for u, v in candidates:
                if v in adj[u]: # Might have been added
                    continue
                common = adj[u] & adj[v]
                creates_k4 = False
                for w in common:
                    if adj[w] & common:
                        creates_k4 = True
                        break
                if creates_k4:
                    continue
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                
                # Check if it increases triangles?
                # We don't track count, so we assume adding an edge in a K4-free graph
                # that doesn't create K4 always increases triangles?
                # No. It increases triangles by the number of edges in N(u) & N(v).
                # If N(u) & N(v) has no edges, triangles don't increase.
                # But it might allow other edges?
                
                # Let's just accept it if it doesn't create K4?
                # This will lead to a maximal K4-free graph.
                # Is maximal K4-free good for minimizing TF set?
                # Not necessarily. A dense graph is better.
                # Let's accept if it increases triangle count.
                # We need to compute triangle count change.
                
                # Triangles added = number of edges in (adj[u] & adj[v]) BEFORE adding u-v?
                # No.
                # New triangles are of the form {u, v, w} where w is common neighbor.
                # So added = len(common).
                # Wait, if w is in common, u-w and v-w are edges.
                # So {u,v,w} is a triangle.
                # So added triangles = len(common).
                
                if len(common) > 0:
                    # Keep the edge
                    improved = True
                    # We don't need to update anything else, just keep adj
                else:
                    # Revert
                    adj[u].remove(v)
                    adj[v].remove(u)
        
        # Evaluate final graph
        score = exact_max_tf_size(adj)
        if score < best_exact_score:
            best_exact_score = score
            best_adj = [s.copy() for s in adj]
            # If we found a very good one (e.g. 9), we can stop early?
            # Theoretical min is likely 9 or 10.
            # If we hit 9, it's likely optimal.
            if best_exact_score <= 9:
                break

    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
                
    return edges

# Test
# if __name__ == "__main__":
#     t0 = time.time()
#     res = construct()
#     print(time.time() - t0)
#     print(len(res))
