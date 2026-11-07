import time
import random

def construct():
    deadline = time.time() + 55  # 55s safety margin (90s limit)
    
    n = 15
    
    def can_add_edge(adj, deg, u, v):
        """Check if adding edge (u,v) would create a forbidden 3-path."""
        # For edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}
        # where s != t (simple path)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    return False
        return True
    
    def add_edge(adj, deg, edges, u, v):
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
        edges.append([u, v])
    
    def remove_edge(adj, deg, edges, u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        deg[u] -= 1
        deg[v] -= 1
        edges.remove([u, v])
    
    def validate_edges(edges):
        """Check if the graph defined by edges satisfies the constraint."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u:
                            continue
                        if s == t:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def greedy_add(adj, deg, edges, candidates, deadline):
        """Try to add as many candidate edges as possible greedily."""
        added = 0
        for u, v in candidates:
            if time.time() > deadline:
                break
            if can_add_edge(adj, deg, u, v):
                add_edge(adj, deg, edges, u, v)
                added += 1
        return added
    
    def local_search(adj, deg, edges, deadline):
        """Local search: try adding edges, and 1-remove-2-add moves."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding all possible edges
            all_edges_set = set()
            for u, v in edges:
                all_edges_set.add((min(u,v), max(u,v)))
            
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in all_edges_set:
                        continue
                    if can_add_edge(adj, deg, u, v):
                        add_edge(adj, deg, edges, u, v)
                        all_edges_set.add((u, v))
                        improved = True
            
            # Try 1-remove-2-add moves
            if time.time() > deadline:
                break
            for i in range(len(edges)):
                if time.time() > deadline:
                    break
                ru, rv = edges[i]
                
                # Remove edge (ru, rv)
                remove_edge(adj, deg, edges, ru, rv)
                
                # Try to add two new edges
                added = 0
                all_edges_set = set()
                for u, v in edges:
                    all_edges_set.add((min(u,v), max(u,v)))
                
                for u in range(n):
                    for v in range(u+1, n):
                        if (u, v) in all_edges_set:
                            continue
                        if can_add_edge(adj, deg, u, v):
                            add_edge(adj, deg, edges, u, v)
                            all_edges_set.add((u, v))
                            added += 1
                            if added == 2:
                                break
                    if added == 2:
                        break
                
                if added >= 1:
                    improved = True
                else:
                    # Restore the removed edge
                    add_edge(adj, deg, edges, ru, rv)
        
        return edges
    
    best_edges = []
    best_count = 0
    
    for seed in range(50):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Initialize with K_{7,8}
        adj = [[] for _ in range(n)]
        deg = [0] * n
        edges = []
        
        for u in range(7):
            for v in range(7, 15):
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
        
        # Generate candidates: intra-partition edges
        candidates = []
        for i in range(7):
            for j in range(i+1, 7):
                candidates.append((i, j))
        for i in range(7, 15):
            for j in range(i+1, 15):
                candidates.append((i, j))
        
        random.shuffle(candidates)
        greedy_add(adj, deg, edges, candidates, deadline)
        
        # Local search
        local_search(adj, deg, edges, deadline)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Final validation
    if not validate_edges(best_edges):
        # Fallback to just K_{7,8}
        best_edges = []
        for u in range(7):
            for v in range(7, 15):
                best_edges.append([u, v])
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
