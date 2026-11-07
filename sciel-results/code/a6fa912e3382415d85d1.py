import time
import random

def construct():
    deadline = time.time() + 85  # 85s safety margin (90s limit)
    
    n = 15
    
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
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def can_add(adj, deg, u, v):
        """Check if edge (u,v) can be added without violating constraints."""
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
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
            if can_add(adj, deg, u, v):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
                added += 1
        return added
    
    def local_search(adj, deg, edges, deadline):
        """Local search: try adding edges, and 1-remove-1-add swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding all possible missing edges
            edge_set = set()
            for u, v in edges:
                edge_set.add((min(u,v), max(u,v)))
            
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    if can_add(adj, deg, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        deg[u] += 1
                        deg[v] += 1
                        edges.append([u, v])
                        edge_set.add((u, v))
                        improved = True
            
            # Try 1-remove-1-add swaps
            if time.time() > deadline:
                break
            
            # Build list of current edges
            current_edges = [(u, v) for u, v in edge_set]
            
            for ru, rv in current_edges:
                if time.time() > deadline:
                    break
                
                # Try removing (ru, rv) and adding some other edge
                # Remove temporarily
                adj[ru].remove(rv)
                adj[rv].remove(ru)
                deg[ru] -= 1
                deg[rv] -= 1
                edges.remove([ru, rv]) if [ru, rv] in edges else None
                edge_set.discard((ru, rv))
                edge_set.discard((rv, ru))
                
                added_any = False
                for u in range(n):
                    for v in range(u+1, n):
                        if (u, v) in edge_set:
                            continue
                        if can_add(adj, deg, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            deg[u] += 1
                            deg[v] += 1
                            edges.append([u, v])
                            edge_set.add((u, v))
                            added_any = True
                            break
                    if added_any:
                        break
                
                if not added_any:
                    # Restore the removed edge
                    adj[ru].add(rv)
                    adj[rv].add(ru)
                    deg[ru] += 1
                    deg[rv] += 1
                    edges.append([ru, rv])
                    edge_set.add((min(ru,rv), max(ru,rv)))
        
        return edges
    
    best_edges = []
    
    for seed in range(30):
        if time.time() > deadline - 5:
            break
        
        random.seed(seed)
        
        # Initialize with K_{7,8}
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        for u in range(7):
            for v in range(7, 15):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
        
        # Generate candidates: intra-part edges
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
        
        if len(edges) > len(best_edges):
            best_edges = edges[:]
    
    # Final validation
    if best_edges and not validate_edges(best_edges):
        # Fallback to just K_{7,8}
        best_edges = []
        for u in range(7):
            for v in range(7, 15):
                best_edges.append([u, v])
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
