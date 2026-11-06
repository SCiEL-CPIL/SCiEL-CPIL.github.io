import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Returns a dictionary with 'n' and 'edges'.
    
    Strategy: Start with 8xK4 (n=32). Add vertices one by one.
    For each new vertex, ensure it is a complete join to at least one existing K4
    to maintain alpha <= 8. Use local search to avoid C5.
    """
    deadline = time.time() + 140.0
    
    def get_edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def has_c5(adj, n):
        # Optimized C5 check
        neighbors = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
        
        for u in range(n):
            for v in neighbors[u]:
                if u >= v: continue
                # Path u-a-b-c-v
                found = False
                for a in neighbors[u]:
                    if a == v: continue
                    for c in neighbors[v]:
                        if c == a or c == u: continue
                        common = neighbors[a].intersection(neighbors[c])
                        if common:
                            for b in common:
                                if b == u or b == v or b == a or b == c:
                                    continue
                                return True
        return False

    def is_valid(adj, n):
        if has_c5(adj, n):
            return False
        # Check alpha <= 8
        # For the constructed graphs (clique unions + joins), we can check alpha
        # by checking if there is an independent set of size 9.
        # Greedy check:
        # Since we enforce the "join to K4" constraint, alpha should be small.
        # Let's do a simple backtracking for independent set of size 9.
        # Or just trust the constraint if we verified it during construction?
        # Better to verify.
        
        # Simple check for independent set of size 9
        # Use a heuristic: try to find 9 non-adjacent vertices.
        # If found, invalid.
        
        # For small n, we can use a bitset DP or just recursion.
        # Given n<=35, recursion is fine.
        
        def find_ind_set(rem, size):
            if size == 0:
                return True
            if len(rem) < size:
                return False
            # Pick a vertex u
            u = rem[0]
            # Option 1: include u
            # New rem: vertices in rem that are not adjacent to u
            new_rem = [v for v in rem[1:] if not adj[u][v]]
            if find_ind_set(new_rem, size - 1):
                return True
            # Option 2: exclude u
            if find_ind_set(rem[1:], size):
                return True
            return False

        if find_ind_set(list(range(n)), 9):
            return False
            
        return True

    # Base graph: 8xK4
    n_base = 32
    adj = [[0]*35 for _ in range(35)] # Max size 35
    
    # Initialize 8xK4
    for b in range(8):
        start = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[start+i][start+j] = 1
                adj[start+j][start+i] = 1
                
    current_n = 32
    
    # Try to add vertices 32, 33, 34
    for new_v in range(32, 35):
        if time.time() > deadline - 20:
            break
            
        # We want to add vertex new_v
        # Constraint: Must join completely to at least one of the 8 cliques.
        # Let's try each clique as the "anchor"
        found = False
        for anchor_b in range(8):
            if time.time() > deadline - 10:
                break
                
            # Initialize connections for new_v
            # Must connect to all 4 vertices in anchor_b
            adj_new = [row[:] for row in adj] # Copy
            for i in range(4):
                u = 4 * anchor_b + i
                adj_new[new_v][u] = 1
                adj_new[u][new_v] = 1
                
            # Randomize other connections
            for u in range(current_n):
                if adj_new[new_v][u] == 0:
                    if random.random() < 0.5:
                        adj_new[new_v][u] = 1
                        adj_new[u][new_v] = 1
                        
            # Local search on the edges of new_v
            best_cost = float('inf')
            best_adj = None
            
            T = 1.0
            cooling = 0.99
            min_T = 0.01
            steps = 0
            max_steps = 2000
            
            while T > min_T and steps < max_steps:
                if time.time() > deadline - 5:
                    break
                    
                # Flip a random edge of new_v
                u = random.randint(0, current_n - 1)
                # Don't flip edges to anchor clique? 
                # If we flip an edge to anchor, we might lose the alpha constraint.
                # We can allow it, but the alpha check will fail.
                # So we can restrict flipping to non-anchor vertices to be safe?
                # Or just flip any and let the cost function handle it.
                
                adj_new[new_v][u] ^= 1
                adj_new[u][new_v] ^= 1
                
                # Check validity
                if is_valid(adj_new, new_v + 1):
                    cost = 0
                else:
                    # Penalty for C5
                    if has_c5(adj_new, new_v + 1):
                        cost = 100
                    else:
                        cost = 10 # Alpha violation or other
                    
                if cost < best_cost:
                    best_cost = cost
                    best_adj = [row[:] for row in adj_new]
                    if cost == 0:
                        # Found valid graph
                        adj = best_adj
                        current_n = new_v + 1
                        found = True
                        break
                
                # SA acceptance
                # We need previous cost. 
                # This is getting complicated.
                # Let's just do random restarts.
                
                T *= cooling
                steps += 1
                
            if found:
                break
                
        if not found:
            break
            
    return {'n': current_n, 'edges': get_edges_from_adj(adj, current_n)}
