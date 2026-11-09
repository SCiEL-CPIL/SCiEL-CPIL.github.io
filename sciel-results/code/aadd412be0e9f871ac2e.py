import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 18
    vertices = list(range(n))
    
    # Helper to check if graph is K4-free
    # Graph represented as set of edges (u, v) with u < v
    def is_k4_free(edges):
        # Precompute adjacency matrix or set for fast lookup
        # For 18 vertices, set of tuples is fine
        edge_set = set()
        for u, v in edges:
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
        
        # Check all 4-cliques
        for quad in itertools.combinations(vertices, 4):
            u, v, w, x = quad
            # Check all 6 edges
            if ((u,v) in edge_set and (u,w) in edge_set and (u,x) in edge_set and
                (v,w) in edge_set and (v,x) in edge_set and (w,x) in edge_set):
                return False
        return True

    # Helper to calculate score: largest triangle-free induced subgraph
    def calc_score(edges):
        edge_set = set()
        for u, v in edges:
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
                
        # Branch and Bound for max triangle-free set
        # State: current set of vertices (frozenset or tuple), best found
        best_size = 0
        
        # Precompute neighbors for speed?
        # adj[v] = set of neighbors
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        def has_triangle(vertices_tuple):
            # Check if induced subgraph on vertices_tuple has a triangle
            # vertices_tuple is a tuple of vertices
            # Optimization: if len < 3, return False
            if len(vertices_tuple) < 3:
                return False
            # Iterate all triples
            for i in range(len(vertices_tuple)):
                for j in range(i+1, len(vertices_tuple)):
                    for k in range(j+1, len(vertices_tuple)):
                        u, v, w = vertices_tuple[i], vertices_tuple[j], vertices_tuple[k]
                        if (v in adj[u] and w in adj[u] and w in adj[v]):
                            return True
            return False
            
        # B&B
        # We want to maximize size of subset with no triangle
        # Start with all vertices
        current = tuple(vertices)
        
        # Memoization might be tricky with changing graph, but graph is fixed for this call
        # Use a simple recursive search with pruning
        # Pruning: if len(current) <= best_size, return
        
        # To speed up, we can order vertices or use a greedy initial bound
        # Greedy: start with empty, add vertices if no triangle formed?
        # No, greedy for max independent set is hard.
        # Let's just use B&B.
        
        def search(vertices_tuple):
            nonlocal best_size
            if len(vertices_tuple) <= best_size:
                return
            
            if not has_triangle(vertices_tuple):
                best_size = len(vertices_tuple)
                return
            
            # Find a triangle to branch on
            # has_triangle returned True, so there is at least one
            # Find the first one
            tri = None
            for i in range(len(vertices_tuple)):
                for j in range(i+1, len(vertices_tuple)):
                    for k in range(j+1, len(vertices_tuple)):
                        u, v, w = vertices_tuple[i], vertices_tuple[j], vertices_tuple[k]
                        if (v in adj[u] and w in adj[u] and w in adj[v]):
                            tri = (u, v, w)
                            break
                    if tri:
                        break
                if tri:
                    break
            
            if not tri:
                # Should not happen if has_triangle is True
                best_size = max(best_size, len(vertices_tuple))
                return
                
            u, v, w = tri
            # Branch: remove u, or remove v, or remove w
            # To prune, remove the one that leaves the largest set?
            # Or just try all 3.
            
            # Remove u
            search(tuple(x for x in vertices_tuple if x != u))
            # Remove v
            search(tuple(x for x in vertices_tuple if x != v))
            # Remove w
            search(tuple(x for x in vertices_tuple if x != w))
            
        search(current)
        return best_size

    # Initialize with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    initial_edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    initial_edges.append([a, b])
                    
    best_edges = [e[:] for e in initial_edges]
    best_score = calc_score(best_edges)
    
    # Simulated Annealing
    # Temperature
    T = 1.0
    cooling = 0.999
    
    all_possible_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_possible_edges.append((u, v))
            
    # Map edge tuple to index for fast toggle?
    # Or just keep list of edges and check membership.
    # For 18 vertices, 153 edges. List membership is O(N).
    # Set membership is O(1).
    
    current_edges = [e[:] for e in best_edges]
    current_score = best_score
    
    # Precompute all K4s?
    # If we toggle an edge, we only need to check K4s containing that edge.
    # There are C(16, 2) = 120 K4s containing a specific edge?
    # No, a K4 has 6 edges.
    # If we add an edge (u,v), it creates a K4 if there exist w,x such that (u,w),(u,x),(v,w),(v,x),(w,x) are all present.
    # If we remove an edge, it cannot create a K4.
    
    # So:
    # 1. Try removing a random edge. Always K4-free.
    # 2. Try adding a random edge. Check if it creates a K4.
    
    while time.time() < deadline:
        # Pick a random edge to toggle
        u, v = random.choice(all_possible_edges)
        
        # Check if edge is currently present
        present = False
        for e in current_edges:
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                present = True
                break
        
        if present:
            # Remove edge
            new_edges = [e for e in current_edges if not ((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u))]
            # Always K4-free
            new_score = calc_score(new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [e[:] for e in current_edges]
            else:
                # Accept with probability
                delta = new_score - current_score
                if delta > 0:
                    if random.random() < pow(2.718, -delta / T):
                        current_edges = new_edges
                        current_score = new_score
                # If delta == 0, accept to move around
                elif random.random() < 0.5:
                    current_edges = new_edges
                    current_score = new_score
        else:
            # Add edge
            # Check if adding (u,v) creates a K4
            # We need to check all pairs (w,x) such that w,x != u,v
            # And (u,w),(u,x),(v,w),(v,x),(w,x) are present
            # This is expensive.
            # Optimization:
            # Only add if it doesn't create K4.
            # If it creates K4, skip this move.
            
            # Build adjacency set for current graph
            adj = [set() for _ in range(n)]
            for a, b in current_edges:
                adj[a].add(b)
                adj[b].add(a)
                
            creates_k4 = False
            # Iterate w in neighbors of u
            for w in adj[u]:
                if w == v: continue
                # Iterate x in neighbors of v
                for x in adj[v]:
                    if x == u or x == w: continue
                    # Check if w,x connected and u,x connected and v,w connected
                    if (x in adj[u] and w in adj[v] and x in adj[w]):
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if creates_k4:
                # Skip move
                continue
                
            # Add edge
            new_edges = current_edges + [[u, v]]
            new_score = calc_score(new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [e[:] for e in current_edges]
            else:
                delta = new_score - current_score
                if delta > 0:
                    if random.random() < pow(2.718, -delta / T):
                        current_edges = new_edges
                        current_score = new_score
                elif random.random() < 0.5:
                    current_edges = new_edges
                    current_score = new_score
                    
        T *= cooling
        
    return best_edges

# EVOLVE-BLOCK-END
