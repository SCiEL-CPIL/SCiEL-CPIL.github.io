import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for 120s limit
    
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common & (common - 1):
                    return False
        return True

    def greedy_mis_size(adj, n):
        # Greedy algorithm to find a large independent set
        # Order vertices by degree (ascending)
        degrees = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x])
        
        independent_set = 0
        available = (1 << n) - 1
        
        for v in order:
            if available & (1 << v):
                independent_set += 1
                # Remove v and its neighbors from available
                available &= ~(adj[v] | (1 << v))
                
        return independent_set

    def has_independent_set_of_size_11(adj, n):
        # First try greedy
        if greedy_mis_size(adj, n) >= 11:
            return True
        
        # If greedy fails, do a limited backtracking
        # This is only called if we are close to a solution
        # To keep it fast, we limit the recursion depth or nodes
        
        # We can use a simple recursive search with pruning
        # State: bitmask of available vertices
        # We want to find 11 vertices
        
        # If n is small, this is fast.
        # For n=43, it might be slow if the graph is dense.
        # But C4-free graphs are sparse.
        
        # Let's use a heuristic: 
        # If the number of edges is very low, alpha is high.
        # If the number of edges is high, alpha is low.
        
        # For now, rely on greedy for speed during search.
        # Only use exact check for final verification.
        return False

    def exact_alpha_le_10(adj, n):
        # Exact check for alpha <= 10
        # Equivalent to: no independent set of size 11
        # Use backtracking
        
        # Sort vertices by degree descending to prune faster?
        # Actually, for finding an independent set, we want to pick vertices that conflict with many others?
        # No, we want to pick vertices that leave many options?
        
        # Standard MIS backtracking:
        # 1. Select a vertex v.
        # 2. Branch on including v or excluding v.
        # 3. If including v, remove v and its neighbors.
        # 4. If excluding v, remove v.
        
        # We want to know if there is an independent set of size 11.
        
        def backtrack(available, current_size):
            if current_size == 11:
                return True
            
            # Pruning: if available has fewer than 11 - current_size bits, return False
            if bin(available).count('1') < 11 - current_size:
                return False
            
            # Heuristic: pick a vertex with high degree in the induced subgraph
            # to reduce the branching factor
            # Or just pick the first one
            # Let's pick the vertex with the highest degree in 'available'
            # to maximize pruning when we include it
            
            # Find vertex with max degree in available
            best_v = -1
            max_deg = -1
            avail_list = [i for i in range(n) if available & (1 << i)]
            
            if not avail_list:
                return False
                
            for v in avail_list:
                deg = bin(available & adj[v]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v
            new_available = available & ~(adj[best_v] | (1 << best_v))
            if backtrack(new_available, current_size + 1):
                return True
            
            # Branch 2: Exclude best_v
            new_available = available & ~(1 << best_v)
            if backtrack(new_available, current_size):
                return True
                
            return False

        return not backtrack((1 << n) - 1, 0)

    best_n = 30
    best_edges = []
    
    # Seed solution: 10 triangles
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
        
    # Try to find better n
    for n in range(43, 30, -1):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Try several random restarts
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize random graph
            adj = [0] * n
            # p = 0.1 is a good starting point for C4-free
            p = 0.1 
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local search
            # Objective: minimize C4 count + large alpha
            # We will just try to remove C4s and keep alpha low
            
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Check if valid
                if is_c4_free(adj, n):
                    # Check alpha
                    # Use greedy first for speed
                    if greedy_mis_size(adj, n) < 11:
                        # Verify with exact check
                        if exact_alpha_le_10(adj, n):
                            found = True
                            # Extract edges
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            best_n = n
                            best_edges = edges
                            break
                            
                if found:
                    break
                    
                # Flip a random edge
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                    
                # Flip
                if adj[i] & (1 << j):
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
            if found:
                break
                
    return {'n': best_n, 'edges': best_edges}
