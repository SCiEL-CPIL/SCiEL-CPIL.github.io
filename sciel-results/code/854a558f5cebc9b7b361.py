import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115
    
    def is_c6_free(adj, u, v):
        # Check if adding (u, v) creates a C6.
        # Equivalent to: Is there a simple path of length 5 from u to v in G?
        # Path: u - n1 - n2 - n3 - n4 - v
        
        # Use a stack for DFS: (node, depth, visited_tuple)
        # visited_tuple is expensive. Use a bitmask or set.
        # With 40 nodes, a set is fine.
        
        # Optimization: Pre-filter.
        # n1 must be in N(u)
        # n4 must be in N(v)
        # We need n1 - n2 - n3 - n4 in G \ {u, v}
        
        # Let's iterate n1 in N(u) and n4 in N(v)
        # And check if there is a path n1 - n2 - n3 - n4 of length 3.
        
        # This might be faster than full DFS from u.
        
        # Path n1 - n2 - n3 - n4:
        # n2 in N(n1), n3 in N(n4)
        # Check if (n2, n3) is an edge.
        # And n1, n2, n3, n4 distinct and not u, v.
        
        # This is O(D^2 * D^2) = O(D^4) per (n1, n4) pair?
        # No, for each n1, n4, we check all n2 in N(n1) and n3 in N(n4).
        # If D=5, 25 checks.
        # Number of (n1, n4) pairs is D*D = 25.
        # Total 625 checks. Much faster than DFS.
        
        # Let's implement this.
        
        neighbors_u = [x for x in range(n) if adj[u][x] and x != v]
        neighbors_v = [x for x in range(n) if adj[v][x] and x != u]
        
        for n1 in neighbors_u:
            for n4 in neighbors_v:
                if n1 == n4:
                    continue
                # Check path n1 - n2 - n3 - n4
                # n2 in N(n1), n3 in N(n4)
                # Edge (n2, n3) must exist.
                # n1, n2, n3, n4 distinct.
                # n2, n3 not u, v.
                
                for n2 in adj[n1]:
                    if n2 == n1 or n2 == n4 or n2 == u or n2 == v:
                        continue
                    for n3 in adj[n4]:
                        if n3 == n4 or n3 == n1 or n3 == n2 or n3 == u or n3 == v:
                            continue
                        if adj[n2][n3]:
                            return False # C6 found
        
        return True

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    best_edges = []
    best_count = 0
    
    def run_local_search(adj, max_time):
        nonlocal best_edges, best_count
        
        current_edges = get_edges(adj)
        current_count = len(current_edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges[:]
            
        while time.time() < max_time:
            # Build list of non-edges
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i][j]:
                        non_edges.append((i, j))
                        
            if not non_edges:
                break # Complete graph (impossible for C6-free)
                
            random.shuffle(non_edges)
            added = False
            for u, v in non_edges:
                if is_c6_free(adj, u, v):
                    adj[u][v] = True
                    adj[v][u] = True
                    current_count += 1
                    added = True
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = get_edges(adj)
                    break
                    
            if not added:
                # Perturb
                if current_count > 0:
                    # Pick a random edge to remove
                    # We need a list of current edges
                    current_edges = get_edges(adj)
                    if current_edges:
                        u, v = random.choice(current_edges)
                        adj[u][v] = False
                        adj[v][u] = False
                        current_count -= 1
                else:
                    break

    # Initial: Seed
    adj = [[False]*n for _ in range(n)]
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u][v] = True
                adj[v][u] = True
                
    run_local_search(adj, deadline)
    
    # More restarts
    while time.time() < deadline:
        # Random restart: Start with a random sparse graph
        adj = [[False]*n for _ in range(n)]
        # Add some random edges
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.15:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Ensure it's C6-free? 
        # If not, local search might fail to add edges or get stuck.
        # But local search only ADDS edges. It doesn't remove C6s.
        # So if the start has a C6, we are in trouble.
        # We should start with a C6-free graph.
        # The seed is C6-free.
        # Random sparse graphs are likely C6-free.
        # Let's verify or just assume for low density.
        
        run_local_search(adj, deadline)

    return best_edges
