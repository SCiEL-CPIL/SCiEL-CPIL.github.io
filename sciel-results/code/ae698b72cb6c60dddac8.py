import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target n values to try, descending from max
    # The task says n <= 47. Known value is 46 or 47.
    # We try to construct for n=47, then 46, etc.
    
    best_result = {'n': 39, 'edges': []}
    
    # We will try to construct a C4-free graph with min degree >= n - 39.
    # For n=46, min degree >= 7.
    # For n=47, min degree >= 8.
    
    # Strategy:
    # 1. Generate a candidate graph using a deterministic algebraic construction (Polarity graph).
    # 2. If it doesn't meet the degree requirement, use local search to add edges to low-degree vertices.
    # 3. If local search fails, try random restarts with different parameters.
    
    # Polarity Graph Construction over GF(q)
    # For q=3, we have 9 points. Not enough.
    # For q=7, we have 49 points. Close to 46/47.
    # The graph has vertices (x,y) in GF(q)^2.
    # Edge if x1*x2 + y1*y2 = 1 (or similar non-degenerate quadratic form).
    # This graph is C4-free.
    
    def generate_polarity_graph(q):
        """
        Generates a polarity graph over GF(q).
        Vertices are pairs (x, y) in GF(q)^2.
        Total vertices = q^2.
        Edge between (x1, y1) and (x2, y2) if x1*x2 + y1*y2 = 1.
        This graph is C4-free.
        Degree is roughly q.
        """
        if q < 2:
            return None, None
        
        # Map vertices to 0..q^2-1
        def vid(x, y):
            return x * q + y
        
        # Check if q is prime for simple field arithmetic
        # For this task, q=7 is prime. q=3 is prime.
        # If q is not prime, we'd need GF(p^k) which is more complex.
        # Let's stick to prime q for simplicity and speed.
        
        vertices = []
        for x in range(q):
            for y in range(q):
                vertices.append((x, y))
        
        edges = []
        # To avoid duplicates and self-loops
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            for j in range(i + 1, len(vertices)):
                x2, y2 = vertices[j]
                # Quadratic form: x1*x2 + y1*y2 = 1
                # We can also use x1*x2 + y1*y2 = 0 (symplectic) but that might have triangles or C4?
                # Standard polarity graph uses non-degenerate quadratic form.
                # For q odd, x^2 + y^2 = 1 is not homogeneous.
                # The standard construction for C4-free graphs is:
                # V = GF(q)^2.
                # E = { {u, v} : u . v = 1 }
                # This graph is C4-free.
                
                dot = (x1 * x2 + y1 * y2) % q
                if dot == 1:
                    edges.append([i, j])
        
        n = len(vertices)
        return n, edges

    def is_c4_free(n, edges):
        """
        Check if graph is C4-free.
        O(E^2) or using matrix multiplication.
        For small n, O(E^2) with adjacency set is fine.
        """
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each pair of vertices (u, v), count common neighbors.
        # If common neighbors >= 2, there is a C4 (u-w1-v-w2-u).
        # Actually, if common neighbors >= 2, we have a cycle of length 4?
        # u - w1 - v - w2 - u. Yes.
        # So we need common neighbors <= 1 for all pairs.
        
        # Optimization: Only check pairs that are not adjacent?
        # No, C4 can involve non-adjacent vertices.
        # We need to check all pairs (u, v) with u < v.
        
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u].intersection(adj[v])
                if len(common) > 1:
                    return False
        return True

    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg)

    def try_construct_for_n(n_target):
        """
        Attempt to construct a C4-free graph on n_target vertices with min degree >= n_target - 39.
        """
        req_deg = n_target - 39
        if req_deg < 0:
            req_deg = 0
            
        # 1. Try Polarity Graph with q such that q^2 >= n_target
        # q=7 -> 49 vertices.
        # We can take a subset of vertices if needed, but degrees will drop.
        # Better: Start with the full polarity graph on q^2 vertices, then prune?
        # Or just use the graph as is if n_target == q^2.
        # If n_target < q^2, we can remove vertices.
        # If n_target > q^2, we need to add vertices? Hard.
        
        # Let's try q=7 (49 vertices) for n=47, 46.
        # We will generate the 49-vertex graph, then remove 49 - n_target vertices.
        # To maximize min degree, we should remove vertices with lowest degree?
        # Or random removal?
        
        candidates = []
        
        # Try q=7
        n_poly, edges_poly = generate_polarity_graph(7)
        if n_poly is not None:
            # n_poly is 49
            if n_target <= n_poly:
                # We need to select n_target vertices from 49.
                # Which ones to remove?
                # Calculate degrees in full graph.
                deg = [0] * n_poly
                for u, v in edges_poly:
                    deg[u] += 1
                    deg[v] += 1
                
                # Sort vertices by degree (ascending)
                # Removing low-degree vertices keeps high-degree ones?
                # Actually, we want the remaining graph to have high min degree.
                # If we remove a vertex, its neighbors lose degree.
                # This is a complex optimization.
                
                # Simpler heuristic: Random sampling of subsets.
                # Or: Greedy removal of vertices with lowest current degree.
                
                # Let's try a few random subsets of size n_target.
                import random
                random.seed(42) # Deterministic for reproducibility in this block
                
                best_subgraph = None
                best_md = -1
                
                # Try 100 random subsets
                for _ in range(100):
                    # Select n_target vertices
                    keep = set(random.sample(range(n_poly), n_target))
                    # Map old ids to new ids
                    old_to_new = {}
                    for new_id, old_id in enumerate(sorted(keep)):
                        old_to_new[old_id] = new_id
                    
                    new_edges = []
                    for u, v in edges_poly:
                        if u in keep and v in keep:
                            new_edges.append([old_to_new[u], old_to_new[v]])
                    
                    # Check C4-free
                    if is_c4_free(n_target, new_edges):
                        md = min_degree(n_target, new_edges)
                        if md > best_md:
                            best_md = md
                            best_subgraph = new_edges
                        
                        # If we meet the requirement, we can stop early?
                        # No, we might find a better one, but requirement is binary.
                        if md >= req_deg:
                            # Found a valid graph!
                            return {'n': n_target, 'edges': new_edges}
                
                # If we didn't find a valid one in random sampling, 
                # try local search on the best subgraph found.
                if best_subgraph is not None:
                    # Local search: Add edges to increase min degree.
                    # But adding edges might create C4.
                    # We can try to add edges between vertices with low degree, 
                    # if it doesn't create a C4.
                    
                    current_edges = best_subgraph
                    current_n = n_target
                    current_md = best_md
                    adj = [set() for _ in range(current_n)]
                    for u, v in current_edges:
                        adj[u].add(v)
                        adj[v].add(u)
                    deg = [len(adj[i]) for i in range(current_n)]
                    
                    # Iterate to improve min degree
                    improved = True
                    while improved and current_md < req_deg:
                        improved = False
                        # Find vertices with min degree
                        min_deg = min(deg)
                        low_verts = [i for i in range(current_n) if deg[i] == min_deg]
                        
                        # Try to add an edge between two low-degree vertices?
                        # Or between a low-degree and any other?
                        # To increase min degree, we must increase the degree of ALL min-degree vertices.
                        # So we need to add edges incident to each min-degree vertex.
                        
                        # Simple heuristic: For each low-degree vertex, try to connect it to a random non-neighbor.
                        for v in low_verts:
                            # Find non-neighbors
                            non_neighbors = [u for u in range(current_n) if u != v and u not in adj[v]]
                            if not non_neighbors:
                                continue
                            
                            # Try adding edge (v, u)
                            # Check if it creates C4
                            # C4 is formed if there is a common neighbor of v and u? No.
                            # C4 is formed if there is a path of length 3 between v and u?
                            # v - w1 - w2 - u.
                            # If we add (v, u), we close the cycle v-w1-w2-u-v.
                            # So we need to check if there is a common neighbor of v and u?
                            # No, that would be a triangle.
                            # We need to check if there is a vertex w such that w is neighbor of v and w is neighbor of u? No, that's triangle.
                            # We need to check if there is a vertex w such that w is neighbor of v and w is neighbor of u?
                            # Wait.
                            # Cycle: v - a - b - u - v.
                            # Edges: (v,a), (a,b), (b,u), (u,v).
                            # If we add (u,v), we need to ensure there is no path of length 3 from u to v.
                            # Path of length 3: u - x - y - v.
                            # This exists if there is an edge (x,y) such that x is neighbor of u and y is neighbor of v.
                            
                            # Efficient check:
                            # For each neighbor x of u, check if any neighbor y of x is a neighbor of v.
                            # i.e., is there y in adj[x] such that y in adj[v]?
                            # This is O(deg(u) * deg(x)).
                            
                            safe = True
                            for x in adj[u]:
                                # Check common neighbors of x and v
                                # If common neighbors > 0, then there is a path u-x-y-v.
                                # Adding u-v creates C4 u-x-y-v-u.
                                common = adj[x].intersection(adj[v])
                                if common:
                                    safe = False
                                    break
                            
                            if safe:
                                # Add edge
                                current_edges.append([v, u])
                                adj[v].add(u)
                                adj[u].add(v)
                                deg[v] += 1
                                deg[u] += 1
                                improved = True
                                # Recalculate min degree
                                current_md = min(deg)
                                if current_md >= req_deg:
                                    return {'n': current_n, 'edges': current_edges}
                    
                    # If local search didn't reach req_deg, return best found if valid
                    if best_md >= req_deg:
                         return {'n': n_target, 'edges': best_subgraph}

        # If polarity graph didn't work, try Random Graph + Repair
        # This is a fallback.
        
        return None

    # Try n=47, 46
    for n_target in [47, 46]:
        if time.time() > deadline:
            break
        result = try_construct_for_n(n_target)
        if result:
            best_result = result
            # If we found 47, we can stop? 
            # The task says maximize n. 47 is max possible.
            if n_target == 47:
                break

    return best_result

# EVOLVE-BLOCK-END
