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
    
    def can_add_edge(u, v, adj, deg):
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
    
    def try_add_all_possible(adj, deg, edges, deadline):
        """Try to add as many missing edges as possible. Returns count added."""
        all_edges_set = set()
        for u, v in edges:
            all_edges_set.add((min(u,v), max(u,v)))
        
        added = 0
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in all_edges_set:
                    continue
                if can_add_edge(u, v, adj, deg):
                    adj[u].append(v)
                    adj[v].append(u)
                    deg[u] += 1
                    deg[v] += 1
                    edges.append([u, v])
                    all_edges_set.add((u, v))
                    added += 1
        return added
    
    def local_search(adj, deg, edges, deadline):
        """Local search: try adding edges, 1-remove-2-add, and 2-remove-3-add moves."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try adding all possible edges
            added = try_add_all_possible(adj, deg, edges, deadline)
            if added > 0:
                improved = True
            
            # Phase 2: Try 1-remove-2-add moves
            if time.time() > deadline:
                break
            n_edges = len(edges)
            for i in range(n_edges):
                if time.time() > deadline:
                    break
                ru, rv = edges[i]
                # Remove edge (ru, rv)
                adj[ru].remove(rv)
                adj[rv].remove(ru)
                deg[ru] -= 1
                deg[rv] -= 1
                edges.pop(i)
                
                # Try to add up to two new edges
                added = 0
                all_edges_set = set()
                for u, v in edges:
                    all_edges_set.add((min(u,v), max(u,v)))
                
                for u in range(n):
                    for v in range(u+1, n):
                        if (u, v) in all_edges_set:
                            continue
                        if can_add_edge(u, v, adj, deg):
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
            
            # Phase 3: Try 2-remove-3-add moves
            if time.time() > deadline:
                break
            n_edges = len(edges)
            for i in range(n_edges):
                if time.time() > deadline:
                    break
                for j in range(i+1, n_edges):
                    if time.time() > deadline:
                        break
                    ru1, rv1 = edges[i]
                    ru2, rv2 = edges[j]
                    # Remove both edges
                    adj[ru1].remove(rv1)
                    adj[rv1].remove(ru1)
                    deg[ru1] -= 1
                    deg[rv1] -= 1
                    adj[ru2].remove(rv2)
                    adj[rv2].remove(ru2)
                    deg[ru2] -= 1
                    deg[rv2] -= 1
                    edges.pop(j)
                    edges.pop(i)
                    
                    # Try to add up to three new edges
                    added = 0
                    all_edges_set = set()
                    for u, v in edges:
                        all_edges_set.add((min(u,v), max(u,v)))
                    
                    for u in range(n):
                        for v in range(u+1, n):
                            if (u, v) in all_edges_set:
                                continue
                            if can_add_edge(u, v, adj, deg):
                                adj[u].append(v)
                                adj[v].append(u)
                                deg[u] += 1
                                deg[v] += 1
                                edges.append([u, v])
                                all_edges_set.add((u, v))
                                added += 1
                                if added == 3:
                                    break
                        if added == 3:
                            break
                    
                    if added >= 2:
                        improved = True
                    else:
                        # Restore the two removed edges
                        adj[ru1].append(rv1)
                        adj[rv1].append(ru1)
                        deg[ru1] += 1
                        deg[rv1] += 1
                        adj[ru2].append(rv2)
                        adj[rv2].append(ru2)
                        deg[ru2] += 1
                        deg[rv2] += 1
                        edges.append([ru1, rv1])
                        edges.append([ru2, rv2])
            
            # Final pass: try adding any remaining possible edges
            if time.time() < deadline:
                added = try_add_all_possible(adj, deg, edges, deadline)
                if added > 0:
                    improved = True
        
        return edges
    
    best_edges = []
    
    # Try multiple partition sizes and random perturbations
    partition_sizes = [(7, 8), (6, 9), (8, 7), (5, 10), (10, 5)]
    
    for seed in range(30):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Choose a partition size
        if seed < len(partition_sizes):
            split = partition_sizes[seed]
        else:
            split = partition_sizes[random.randint(0, len(partition_sizes)-1)]
        
        a, b = split
        
        # Initialize with K_{a,b}
        adj = [[] for _ in range(n)]
        deg = [0] * n
        edges = []
        
        for u in range(a):
            for v in range(a, n):
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
        
        # Generate candidates: all edges within the two parts
        candidates = []
        for i in range(a):
            for j in range(i+1, a):
                candidates.append((i, j))
        for i in range(a, n):
            for j in range(i+1, n):
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
