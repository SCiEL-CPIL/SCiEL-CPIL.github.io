import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def get_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_triangle_free(n, edges):
        adj = get_adj(n, edges)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u].intersection(adj[v]):
                        return False
        return True

    def find_independent_set(n, adj, size=10):
        # Find an independent set of given size
        # Returns the set or None
        candidates = list(range(n))
        
        def bnb(current, available):
            if len(current) == size:
                return current
            if not available:
                return None
            
            v = available[0]
            remaining = available[1:]
            
            # Try including v
            new_available = [u for u in remaining if u not in adj[v]]
            res = bnb(current + [v], new_available)
            if res:
                return res
            
            # Try excluding v
            res = bnb(current, remaining)
            if res:
                return res
            return None

        return bnb([], candidates)

    def try_find(n, max_restarts=20):
        for _ in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Generate random triangle-free graph
            edges = []
            adj = [set() for _ in range(n)]
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if not adj[u].intersection(adj[v]):
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Local search to reduce alpha
            for step in range(50):
                if time.time() > deadline:
                    return None
                
                indep_set = find_independent_set(n, adj, 10)
                if not indep_set:
                    return edges # Valid!
                
                # Try to add an edge within indep_set
                added = False
                for i in range(len(indep_set)):
                    for j in range(i+1, len(indep_set)):
                        u = indep_set[i]
                        v = indep_set[j]
                        if v not in adj[u]:
                            if not adj[u].intersection(adj[v]):
                                edges.append((u, v))
                                adj[u].add(v)
                                adj[v].add(u)
                                added = True
                                break
                    if added:
                        break
                if not added:
                    break # Stuck
            
        return None

    best_n = 26
    best_edges = None
    
    for n in range(39, 29, -1):
        if time.time() > deadline:
            break
        edges = try_find(n)
        if edges is not None:
            best_n = n
            best_edges = edges
            break
            
    if best_edges is None:
        n = 26
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        best_n = n
        
    return {'n': best_n, 'edges': [[u, v] for u, v in best_edges]}
