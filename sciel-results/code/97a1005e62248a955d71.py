"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_triangle(n, adj):
    """Check if graph with adjacency list has any triangle."""
    for i in range(n):
        for j in adj[i]:
            if j > i:
                for k in adj[j]:
                    if k > j and k in adj[i]:
                        return True
    return False

def _max_independent_set(n, adj):
    """Find maximum independent set size using branch and bound."""
    adj_sets = [set(adj[i]) for i in range(n)]
    
    # Convert to complement for independent set search
    # Independent set in G = clique in complement
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj_sets[i]:
                comp_adj[i].add(j)
    
    best = [0]
    
    def expand(clique, candidates):
        if not candidates:
            if len(clique) > best[0]:
                best[0] = len(clique)
            return
        if len(clique) + len(candidates) <= best[0]:
            return
        
        # Greedy coloring bound
        color = {}
        uncolored = list(candidates)
        color_class = 0
        while uncolored:
            color_class += 1
            color_set = set()
            for v in list(uncolored):
                if not any(u in color_set and u in comp_adj[v] for u in color_set):
                    color[v] = color_class
                    color_set.add(v)
                    uncolored.remove(v)
        
        if len(clique) + color_class <= best[0]:
            return
        
        # Choose vertex with highest color
        v = max(candidates, key=lambda x: color.get(x, 0))
        
        expand(clique + [v], candidates & comp_adj[v])
        candidates.discard(v)
    
    expand([], set(range(n)))
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with Andrasfai graph And(9): Z_26 with i ~ j iff (j - i) mod 26 is 1 mod 3
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Verify initial graph
    if _has_triangle(n, adj):
        # Fallback: return empty graph if something went wrong
        return {'n': 1, 'edges': []}
    
    alpha = _max_independent_set(n, adj)
    if alpha >= 10:
        # Fallback
        return {'n': 1, 'edges': []}
    
    best_n = n
    best_edges = edges[:]
    
    # Try to extend the graph by adding vertices
    # We'll try to add one vertex at a time, checking constraints
    current_n = n
    current_edges = edges[:]
    current_adj = adj
    
    # Try adding vertices up to n=40
    while current_n < 40:
        if time.time() > deadline:
            break
        
        # Try to add a new vertex connected to some existing vertices
        # The new vertex must not create a triangle and must not increase alpha to 10
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Choose a random subset of existing vertices to connect to
            # The new vertex v = current_n
            # It can connect to vertices that don't form triangles with each other
            # (i.e., the neighbors of v must form an independent set in G)
            
            # Try different connection patterns
            # Simple approach: connect to a random independent set
            if random.random() < 0.5:
                # Connect to no one (isolated vertex) - this increases alpha by 1
                if alpha + 1 >= 10:
                    continue
                new_adj = current_adj[:]
                new_adj.append(set())
                new_n = current_n + 1
                new_edges = current_edges[:]
                
                # Check no triangle (trivially true since no edges)
                if not _has_triangle(new_n, new_adj):
                    new_alpha = _max_independent_set(new_n, new_adj)
                    if new_alpha < 10:
                        current_n = new_n
                        current_adj = new_adj
                        current_edges = new_edges
                        alpha = new_alpha
                        found = True
                        break
            else:
                # Connect to some vertices
                # Pick a random subset of size 1-3
                conn_size = random.randint(1, 3)
                candidates = list(range(current_n))
                random.shuffle(candidates)
                neighbors = candidates[:conn_size]
                
                # Check that neighbors form an independent set (no edges between them)
                valid = True
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        if neighbors[j] in current_adj[neighbors[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Build new adjacency
                new_adj = [set(s) for s in current_adj]
                new_adj.append(set(neighbors))
                for v in neighbors:
                    new_adj[v].add(current_n)
                
                new_n = current_n + 1
                new_edges = current_edges[:]
                for v in neighbors:
                    new_edges.append([v, current_n])
                
                # Check no triangle
                if _has_triangle(new_n, new_adj):
                    continue
                
                new_alpha = _max_independent_set(new_n, new_adj)
                if new_alpha < 10:
                    current_n = new_n
                    current_adj = new_adj
                    current_edges = new_edges
                    alpha = new_alpha
                    found = True
                    break
        
        if not found:
            break
        
        best_n = current_n
        best_edges = current_edges[:]
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
