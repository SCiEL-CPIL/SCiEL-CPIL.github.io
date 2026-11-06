import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Fallback: 8 disjoint copies of K4
    # n = 32
    # alpha = 8 (one from each K4)
    # C5-free: K4 has no 5-cycle, disjoint union has no 5-cycle.
    def make_k4_graph(num_copies):
        n = 4 * num_copies
        edges = []
        for b in range(num_copies):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    def build_adj(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def has_c5(adj, n):
        """Check if graph contains a 5-cycle. Returns True if C5 found."""
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]: continue
                for c in range(b+1, n):
                    if not adj[b][c]: continue
                    for d in range(c+1, n):
                        if not adj[c][d]: continue
                        for e in range(d+1, n):
                            if not adj[d][e]: continue
                            if adj[e][a]:
                                return True
        return False
    
    def max_independent_set_size(adj, n):
        """Compute maximum independent set size using branch and bound."""
        # For small n, use a greedy + exact approach
        # Convert to adjacency list
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and adj[i][j]:
                    adj_list[i].append(j)
        
        # Use a simple recursive approach with pruning
        best = [0]
        
        def is_independent(vertices):
            """Check if set of vertices is independent."""
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    if adj[vertices[i]][vertices[j]]:
                        return False
            return True
        
        # Branch and bound
        def bnb(remaining, current_set, current_size):
            if time.time() > deadline:
                return
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound: current_size + len(remaining)
            if current_size + len(remaining) <= best[0]:
                return
            
            # Try including the first vertex
            v = remaining[0]
            rest = remaining[1:]
            
            # Include v
            new_remaining = [x for x in rest if not adj[v][x]]
            bnb(new_remaining, current_set + [v], current_size + 1)
            
            # Exclude v
            bnb(rest, current_set, current_size)
        
        bnb(list(range(n)), [], 0)
        return best[0]
    
    def check_valid(adj, n):
        """Check C5-free and alpha < 9."""
        if has_c5(adj, n):
            return False
        alpha = max_independent_set_size(adj, n)
        return alpha < 9
    
    # Start with 8xK4
    result = make_k4_graph(8)
    n = result['n']
    edges = result['edges']
    adj = build_adj(n, edges)
    
    # Verify base case
    if not check_valid(adj, n):
        return make_k4_graph(7)
    
    best_n = n
    best_edges = edges[:]
    
    # Try to extend to n+1, n+2, n+3
    for target_n in range(n+1, min(n+4, 36)):
        if time.time() > deadline:
            break
        
        new_adj = [[False]*target_n for _ in range(target_n)]
        for i in range(n):
            for j in range(n):
                new_adj[i][j] = adj[i][j]
        
        new_edges = edges[:]
        
        # Try all subsets of existing vertices to connect to new vertex
        # The new vertex (index n) needs to be connected to some subset
        # such that no C5 is formed and alpha < 9
        
        found = False
        # Use random sampling of connection patterns
        random.seed(42)
        
        # Try various connection densities
        for density in [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]:
            if time.time() > deadline:
                break
            if found:
                break
            
            for trial in range(50):
                if time.time() > deadline:
                    break
                
                # Randomly choose connections for new vertex
                new_conn = []
                for v in range(n):
                    if random.random() < density:
                        new_conn.append(v)
                
                # Build new adjacency
                for v in new_conn:
                    new_adj[n][v] = True
                    new_adj[v][n] = True
                    new_edges.append([n, v])
                
                if check_valid(new_adj, target_n):
                    found = True
                    best_n = target_n
                    best_edges = new_edges[:]
                    # Update adj for next iteration
                    adj = new_adj
                    n = target_n
                    break
                else:
                    # Reset
                    new_adj = [[False]*target_n for _ in range(target_n)]
                    for i in range(n):
                        for j in range(n):
                            new_adj[i][j] = adj[i][j]
                    new_edges = edges[:]
        
        if not found:
            break
    
    return {'n': best_n, 'edges': best_edges}
