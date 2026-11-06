import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def has_triangle(adj, n):
        """Check if adjacency list contains a triangle."""
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    # Check if any neighbor of v (greater than u to avoid duplicates?) is in adj[u]
                    # Actually, just check if any common neighbor exists.
                    # For efficiency, assume adj lists are sets.
                    common = adj[u] & adj[v]
                    if common:
                        return True
        return False

    def max_independent_set_size(adj, n):
        """
        Exact branch and bound for maximum independent set size.
        Returns the size.
        Optimized for n <= 41.
        """
        # Use a simple upper bound based on greedy coloring of the complement graph?
        # Or just standard BB.
        # For n=41, exact BB can be slow if not pruned well.
        # Let's use a recursive search with pruning.
        
        # To make it faster, we can use a heuristic to find a good lower bound first.
        # Greedy independent set:
        def greedy_independent_set(vertices):
            if not vertices:
                return 0
            # Pick vertex with min degree in induced subgraph
            min_deg = float('inf')
            chosen = None
            for v in vertices:
                deg = len(adj[v] & set(vertices))
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            # Include chosen, exclude its neighbors
            remaining = set(vertices) - {chosen} - adj[chosen]
            return 1 + greedy_independent_set(remaining)
        
        best_size = greedy_independent_set(set(range(n)))
        
        # Branch and Bound
        # State: current set of vertices to consider, current independent set size
        # We want to find if we can get size > best_size
        
        # To keep it simple and fast enough for n<=41 within time limits:
        # We will use a simple recursive BB that prunes if current_size + len(vertices) <= best_size
        
        def bb(vertices, current_size):
            nonlocal best_size
            if not vertices:
                return
            # Pruning: even if we take all remaining, we can't beat best
            if current_size + len(vertices) <= best_size:
                return
            
            # Heuristic ordering: pick vertex with highest degree in induced subgraph?
            # Or lowest? Usually for MaxClique (equivalent to MaxIndSet in complement),
            # ordering by degree helps.
            
            # Let's just pick the first vertex for simplicity, but try to prune early.
            # A better heuristic: order vertices by degree in the induced subgraph (descending).
            # Vertices with high degree in the induced subgraph are less likely to be in the independent set?
            # No, in independent set, we want vertices with LOW degree in the induced subgraph.
            # So sort by degree ascending.
            
            verts_list = list(vertices)
            # Sort by degree in induced subgraph
            def deg_in_induced(v):
                return len(adj[v] & set(verts_list))
            
            # This sorting is expensive. For n<=41, maybe just pick min degree each time.
            # Let's stick to a simpler recursive approach with a good lower bound.
            
            # Actually, for n=41, a naive BB might be too slow if the graph is dense/sparse in bad ways.
            # But Andrásfai graphs are fairly regular.
            
            # Let's use a simple recursive step:
            # Pick a vertex v.
            # Option 1: Include v. New vertices = vertices \ ({v} U N(v)).
            # Option 2: Exclude v. New vertices = vertices \ {v}.
            
            # To ensure we find the MAX, we explore Include first (if promising) or Exclude first?
            # Usually, exploring the branch that keeps more options open (Exclude) is worse for finding a large set quickly?
            # Actually, if we include v, we lose neighbors. If we exclude v, we keep them.
            # To find a LARGE independent set, including a vertex with few neighbors is good.
            
            # Let's pick the vertex with the minimum degree in the induced subgraph.
            min_deg = float('inf')
            chosen = None
            for v in verts_list:
                d = len(adj[v] & set(verts_list))
                if d < min_deg:
                    min_deg = d
                    chosen = v
            
            # Branch 1: Include chosen
            new_vertices_include = set(verts_list) - {chosen} - (adj[chosen] & set(verts_list))
            bb(new_vertices_include, current_size + 1)
            
            # Branch 2: Exclude chosen
            # Prune if current_size + len(new_vertices_exclude) <= best_size
            new_vertices_exclude = set(verts_list) - {chosen}
            if current_size + len(new_vertices_exclude) > best_size:
                bb(new_vertices_exclude, current_size)
        
        bb(set(range(n)), 0)
        return best_size

    def andrasfai_graph(n, k):
        """
        Construct Andrásfai graph A_n(k).
        Vertices 0..n-1.
        Edge (u, v) if (v - u) mod n is in {1, ..., k} or {n-k, ..., n-1}.
        Triangle-free if 3k < n.
        """
        adj = [set() for _ in range(n)]
        for u in range(n):
            for d in range(1, k + 1):
                v1 = (u + d) % n
                v2 = (u - d) % n
                if v1 != u:
                    adj[u].add(v1)
                    adj[v1].add(u)
                if v2 != u:
                    adj[u].add(v2)
                    adj[v2].add(u)
        return adj

    def local_search(adj, n, time_limit=5):
        """
        Try to improve the graph (increase n or maintain validity) via local search.
        Here we focus on verifying and potentially adding edges to reduce alpha, 
        or adding vertices.
        """
        pass # Placeholder, we will do specific extensions in main loop

    best_n = 0
    best_edges = []
    
    # Strategy:
    # 1. Start with known good families: Andrásfai graphs.
    # 2. Try to extend them or find larger variants.
    # 3. Use random local search on adjacency matrices for fixed n to find valid configs.
    
    # First, check Andrásfai graphs for various n, k
    # Triangle-free condition: 3k < n
    # Independence number of A_n(k) is roughly n/(2k+1) ? 
    # Actually, alpha(A_n(k)) = floor(n/(2k+1)) is not always exact, but close.
    # We need alpha < 10.
    
    # Let's scan for valid Andrásfai graphs first.
    for n in range(10, 41):
        if time.time() > deadline - 10:
            break
        max_k = (n - 1) // 3  # 3k < n => k < n/3
        for k in range(1, max_k + 1):
            adj = andrasfai_graph(n, k)
            if has_triangle(adj, n):
                continue # Should not happen if 3k < n
            
            # Check alpha
            alpha = max_independent_set_size(adj, n)
            if alpha < 10:
                if n > best_n:
                    best_n = n
                    best_edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                best_edges.append([u, v])
    
    # Now, try to extend the best found or search for better random graphs
    # We can try to add vertices to the best graph found.
    # Or, use a random local search for each target n > best_n.
    
    for target_n in range(best_n + 1, 41):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations and local search
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with a random triangle-free graph?
            # Or start with Andrásfai and perturb?
            # Let's start with a random graph and remove triangles?
            # Better: Start with a known valid graph on target_n - 1 and add a vertex.
            
            # If we have a valid graph on target_n - 1, try to add vertex target_n - 1.
            # But we might not have the best graph on target_n - 1 stored.
            
            # Alternative: Random local search on n vertices.
            # Start with a sparse random graph.
            adj = [set() for _ in range(target_n)]
            
            # Add random edges while maintaining triangle-free
            # To ensure triangle-free, we can build it by adding vertices one by one
            # and connecting to a clique-free subset of existing vertices.
            # This is complex.
            
            # Let's use a different heuristic:
            # Start with Andrásfai graph of size target_n if valid, else perturb.
            # If Andrásfai is invalid (alpha >= 10), try to add edges to reduce alpha.
            
            max_k = (target_n - 1) // 3
            if max_k < 1:
                continue
                
            # Pick a k that gives alpha close to 10
            # alpha approx n/(2k+1)
            # We want n/(2k+1) < 10 => 2k+1 > n/10 => k > n/20 - 0.5
            # Also 3k < n => k < n/3
            
            min_k_needed = max(1, int(target_n / 20))
            max_k_allowed = (target_n - 1) // 3
            
            if min_k_needed > max_k_allowed:
                continue
                
            # Try a k in the range
            k = random.randint(min_k_needed, max_k_allowed)
            adj = andrasfai_graph(target_n, k)
            
            # Verify triangle-free
            if has_triangle(adj, target_n):
                continue
            
            # Check alpha
            alpha = max_independent_set_size(adj, target_n)
            
            # If alpha >= 10, try to reduce it by adding edges
            # Adding edges reduces alpha.
            # We can try to add random non-triangular edges until alpha < 10 or no more can be added.
            
            iterations = 0
            while alpha >= 10 and iterations < 1000:
                if time.time() > deadline - 5:
                    break
                
                # Find a pair (u, v) that is not an edge, and adding it doesn't create a triangle
                # And hopefully reduces alpha.
                # Randomly pick u, v
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v or v in adj[u]:
                    continue
                
                # Check if adding (u, v) creates a triangle
                # Triangle exists if there is w such that w in adj[u] and w in adj[v]
                common = adj[u] & adj[v]
                if common:
                    continue
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                iterations += 1
                
                # Re-evaluate alpha (expensive)
                # Only re-evaluate every few iterations or if we think it might help?
                # For n=41, BB is fast enough if we don't do it 1000 times.
                # Let's re-evaluate every 10 iterations or if we hit a limit.
                if iterations % 10 == 0:
                    alpha = max_independent_set_size(adj, target_n)
                    if alpha < 10:
                        break
            
            if alpha < 10:
                # Valid graph found
                best_n = target_n
                best_edges = []
                for u in range(target_n):
                    for v in adj[u]:
                        if u < v:
                            best_edges.append([u, v])
                found = True
                break # Move to next target_n
        
        if not found:
            # If we couldn't find a valid graph for target_n, 
            # it doesn't necessarily mean target_n+1 is impossible, 
            # but usually if we can't find one for n, we might struggle for n+1.
            # However, the problem is not monotonic in "findability" easily.
            # But if we failed to find one for n, we can continue trying n+1.
            pass

    return {'n': best_n, 'edges': best_edges}
