import random
import time

def construct():
    n = 20
    best_edges = 0
    best_adj = None
    deadline = time.time() + 110  # Safety margin
    
    def is_safe_addition(u, v, adj):
        # Check if adding (u,v) creates a K3,3
        # Assumes u, v are NOT currently connected
        nv = adj[v] - {u}
        nu = adj[u] - {v}
        
        nv_list = list(nv)
        for i in range(len(nv_list)):
            for j in range(i + 1, len(nv_list)):
                a1 = nv_list[i]
                a2 = nv_list[j]
                common = adj[a1] & adj[a2]
                intersection = common & nu
                if len(intersection) >= 2:
                    return False
        return True

    def repair(adj):
        # Remove edges to fix violations
        # A violation is a K3,3 subgraph.
        # We can just try to remove edges randomly until no violation.
        # To find a violation, we can check all triples?
        # Or, we can just check if the graph is K3,3 free.
        # If not, find one.
        
        # Efficient way to find violation:
        # Iterate all triples (u,v,w). Check if |N(u) & N(v) & N(w)| >= 3.
        # If so, return the triple and the common neighbors.
        
        while True:
            violation = None
            for u in range(n):
                for v in range(u + 1, n):
                    for w in range(v + 1, n):
                        common = adj[u] & adj[v] & adj[w]
                        if len(common) >= 3:
                            violation = (u, v, w, list(common))
                            break
                    if violation: break
                if violation: break
            
            if not violation:
                break
            
            u, v, w, common = violation
            # Remove a random edge incident to one of u,v,w and one of common
            # This edge is part of the K3,3
            # Actually, any edge in the K3,3 works.
            # The K3,3 edges are between {u,v,w} and common.
            # Pick a random edge from these 3*3 = 9 edges?
            # Wait, the K3,3 is defined by the bipartite connections.
            # The edges present in the graph between {u,v,w} and common form a subgraph of K3,3.
            # If len(common) >= 3, it means at least 3 vertices in common are connected to ALL of u,v,w.
            # So the edges (u, c), (v, c), (w, c) for c in common are all present.
            # We can remove any of these.
            c = random.choice(common)
            # Remove edge (u, c)
            adj[u].remove(c)
            adj[c].remove(u)
            
    def greedy_add(adj):
        # Try to add all possible edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        if is_safe_addition(u, v, adj):
                            adj[u].add(v)
                            adj[v].add(u)
                            changed = True
                            if time.time() > deadline:
                                return adj
        return adj

    def local_search(adj):
        # Try to improve by removing an edge and re-running greedy?
        # Or just swap: remove e1, add e2.
        # Simple hill climbing:
        # Try removing each edge, see if we can add more.
        # This is expensive.
        # Let's stick to random restarts with repair + greedy.
        # It's usually sufficient for n=20.
        
        # However, local search can escape local maxima of greedy.
        # Let's do a simple swap:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing a random edge
            if not adj:
                break
            u = random.randint(0, n - 1)
            if not adj[u]:
                continue
            v = random.choice(list(adj[u]))
            
            # Remove (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Greedy add
            new_adj = greedy_add({i: set(adj[i]) for i in range(n)})
            
            if len([1 for i in range(n) for j in new_adj[i] if i < j]) > len([1 for i in range(n) for j in adj[i] if i < j]):
                adj = new_adj
                improved = True
            else:
                # Revert
                adj[u].add(v)
                adj[v].add(u)
                
        return adj

    # Main loop
    while time.time() < deadline:
        # Random initialization
        # p=0.5 is a good start
        adj = {i: set() for i in range(n)}
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Repair
        if time.time() < deadline:
            repair(adj)
            
        # Greedy Add
        if time.time() < deadline:
            greedy_add(adj)
            
        # Local Search
        if time.time() < deadline:
            local_search(adj)
            
        # Count edges
        edge_count = sum(len(adj[u]) for u in range(n)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = {i: set(adj[i]) for i in range(n)}

    # Convert to list of edges
    edges = []
    if best_adj:
        for u in range(n):
            for v in best_adj[u]:
                if u < v:
                    edges.append([u, v])
    else:
        # Fallback to seed
        edges = [[a, b] for a in range(2) for b in range(2, 20)]
        
    return edges
