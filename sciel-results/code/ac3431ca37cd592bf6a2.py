import time
import random

def construct():
    deadline = time.time() + 55  # 55s safety margin (90s limit)
    
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
                # For edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}
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
            # Check if (u,v) can be added
            ok = True
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u:
                        continue
                    if deg[s] == deg[t]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
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
                    # Check if can add
                    ok = True
                    for s in adj[u]:
                        if s == v:
                            continue
                        for t in adj[v]:
                            if t == u:
                                continue
                            if deg[s] == deg[t]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        adj[u].append(v)
                        adj[v].append(u)
                        deg[u] += 1
                        deg[v] += 1
                        edges.append([u, v])
                        all_edges_set.add((u, v))
                        improved = True
            
            # Try 1-remove-2-add moves
            if time.time() > deadline:
                break
            for i, (ru, rv) in enumerate(edges):
                if time.time() > deadline:
                    break
                # Remove edge (ru, rv)
                adj[ru].remove(rv)
                adj[rv].remove(ru)
                deg[ru] -= 1
                deg[rv] -= 1
                edges.pop(i)
                
                # Try to add two new edges
                added = 0
                all_edges_set = set()
                for u, v in edges:
                    all_edges_set.add((min(u,v), max(u,v)))
                
                for u in range(n):
                    for v in range(u+1, n):
                        if (u, v) in all_edges_set:
                            continue
                        # Check if can add
                        ok = True
                        for s in adj[u]:
                            if s == v:
                                continue
                            for t in adj[v]:
                                if t == u:
                                    continue
                                if deg[s] == deg[t]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            adj[u].append(v)
                            adj[v].append(u)
                            deg[u] += 1
                            deg[v] += 1
                            edges.append([u, v])
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
                    adj[ru].append(rv)
                    adj[rv].append(ru)
                    deg[ru] += 1
                    deg[rv] += 1
                    edges.append([ru, rv])
        
        return edges
    
    best_edges = []
    
    for seed in range(20):
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
        
        # Generate candidates
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
    if not validate_edges(best_edges):
        # Fallback to just K_{7,8}
        best_edges = []
        for u in range(7):
            for v in range(7, 15):
                best_edges.append([u, v])
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
