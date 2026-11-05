import random

def construct():
    n = 20
    
    def is_valid_add(adj, u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        # 1. Pair (u,v) common neighbours must be <= 2
        if len(adj[u] & adj[v]) > 2:
            return False
            
        # 2. For all w in N(v) \ {u}, v becomes a new common neighbour of u and w.
        #    So |N(u) ∩ N(w)| must be <= 1
        for w in adj[v]:
            if w == u:
                continue
            if len(adj[u] & adj[w]) > 1:
                return False
                
        # 3. For all w in N(u) \ {v}, u becomes a new common neighbour of v and w.
        #    So |N(v) ∩ N(w)| must be <= 1
        for w in adj[u]:
            if w == v:
                continue
            if len(adj[v] & adj[w]) > 1:
                return False
                
        return True

    def check_validity(adj):
        for i in range(n):
            for j in range(i+1, n):
                if len(adj[i] & adj[j]) > 2:
                    return False
        return True

    best_edges = -1
    best_adj = None
    
    # Use a fixed seed for reproducibility
    random.seed(12345)
    
    # Perform multiple random restarts
    for restart in range(100):
        adj = [set() for _ in range(n)]
        
        # Random initialization with varying density
        p = 0.3 + 0.4 * random.random()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Repair invalid graph by removing random edges
        # This is a simple heuristic. More sophisticated repair could be used.
        valid = False
        while not valid:
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if len(adj[i] & adj[j]) > 2:
                        valid = False
                        # Find all edges to remove from
                        edges = [(u, v) for u in range(n) for v in range(u+1, n) if v in adj[u]]
                        if edges:
                            # Remove a random edge
                            u, v = random.choice(edges)
                            adj[u].remove(v)
                            adj[v].remove(u)
                        break
                if not valid:
                    break
                    
        # Greedy addition of edges
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        if is_valid_add(adj, i, j):
                            adj[i].add(j)
                            adj[j].add(i)
                            improved = True
                            
        current_edges = sum(len(s) for s in adj) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [set(s) for s in adj]
            
    # Convert best_adj to list of edges
    final_edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                final_edges.append([i, j])
                
    return final_edges
